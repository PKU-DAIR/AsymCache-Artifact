from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

USE_ORG = os.getenv("USE_ORG") == "1"
AUTO_SPLIT = os.getenv("AUTO_SPLIT") == "1"
_FA2_LIBRARY_DIR = Path(__file__).resolve().parents[3] / ("vllm_flash_attn_org" if USE_ORG else "vllm_flash_attn")
_FA2_LIBRARIES = list(_FA2_LIBRARY_DIR.glob("_vllm_fa2_C*.so"))
if len(_FA2_LIBRARIES) != 1:
    raise ImportError(f"Expected one FA2 library in {_FA2_LIBRARY_DIR}")
else:
    print(f"Loading FA2 library from {_FA2_LIBRARIES[0]}")
torch.ops.load_library(str(_FA2_LIBRARIES[0]))


@dataclass
class FAPodAttentionMetadata:
    cache_seqlens_int32: torch.Tensor = None
    max_seq_len_q: int = 1
    max_seq_len_k: int = 0
    cu_seqlens_q: torch.Tensor = None
    page_table: torch.Tensor = None


class FAPodAttentionBackend(AttentionBackend):
    """Paged MHA backend using the vLLM FlashAttention v2 operators."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__()

        assert not model_runner.use_mla_backend, "fapod does not support MLA"
        assert not model_runner.model_config.is_encoder_decoder, (
            "fapod does not support cross attention"
        )
        assert model_runner.page_size == 64, "fapod requires page size 64"
        assert getattr(model_runner, "attention_chunk_size", None) is None, (
            "fapod does not support local attention"
        )
        assert model_runner.server_args.speculative_algorithm is None, (
            "fapod does not support speculative decoding"
        )
        assert model_runner.server_args.kv_cache_dtype == "auto", (
            "fapod does not support FP8 KV cache"
        )

        self.device = model_runner.device
        self.max_context_len = model_runner.model_config.context_len
        self.page_size = model_runner.page_size
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.forward_metadata: Optional[FAPodAttentionMetadata] = None
        self.cuda_graph_metadata = {}
        self.pod_attn = None
        if os.getenv("USE_POD") == "1":
            self.pod_attn = importlib.import_module("pod_attn")

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        assert forward_batch.spec_info is None, (
            "fapod does not support speculative decoding"
        )
        assert forward_batch.encoder_lens is None, (
            "fapod does not support cross attention"
        )

        metadata = FAPodAttentionMetadata()
        seq_lens = forward_batch.seq_lens.to(torch.int32).contiguous()
        metadata.cache_seqlens_int32 = seq_lens
        metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
        raw_page_table = forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, : metadata.max_seq_len_k
        ]

        if forward_batch.forward_mode.is_decode_or_idle():
            metadata.cu_seqlens_q = torch.arange(
                0,
                forward_batch.batch_size + 1,
                dtype=torch.int32,
                device=seq_lens.device,
            )
        else:
            assert forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
            if any(forward_batch.extend_prefix_lens_cpu):
                extend_seq_lens = forward_batch.extend_seq_lens.to(torch.int32)
                metadata.max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
                metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
            else:
                metadata.max_seq_len_q = metadata.max_seq_len_k
                metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )

        num_pages = (metadata.max_seq_len_k + self.page_size - 1) // self.page_size
        metadata.page_table = torch.empty(
            (forward_batch.batch_size, num_pages),
            dtype=torch.int32,
            device=seq_lens.device,
        )
        metadata.page_table.copy_(raw_page_table[:, :: self.page_size])
        metadata.page_table.div_(self.page_size, rounding_mode="floor")
        self.forward_metadata = metadata

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        max_pages = (self.max_context_len + self.page_size - 1) // self.page_size
        self.cuda_graph_metadata = {
            "cache_seqlens": torch.zeros(
                max_bs, dtype=torch.int32, device=self.device
            ),
            "cu_seqlens_q": torch.arange(
                0, max_bs + 1, dtype=torch.int32, device=self.device
            ),
            "page_table": torch.zeros(
                max_bs, max_pages, dtype=torch.int32, device=self.device
            ),
            "strided_indices": torch.arange(
                0, self.max_context_len, self.page_size, device=self.device
            ),
        }

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info,
    ):
        assert forward_mode.is_decode_or_idle()
        assert encoder_lens is None and spec_info is None

        cache_seqlens = self.cuda_graph_metadata["cache_seqlens"][:bs]
        cache_seqlens.copy_(seq_lens[:bs])
        metadata = FAPodAttentionMetadata(
            cache_seqlens_int32=cache_seqlens,
            max_seq_len_q=1,
            max_seq_len_k=seq_lens.max().item(),
            cu_seqlens_q=self.cuda_graph_metadata["cu_seqlens_q"][: bs + 1],
            page_table=self.cuda_graph_metadata["page_table"][:bs],
        )
        self.cuda_graph_metadata[bs] = metadata
        self.forward_metadata = metadata

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info,
        seq_lens_cpu: Optional[torch.Tensor],
        out_cache_loc: Optional[torch.Tensor] = None,
    ):
        assert forward_mode.is_decode_or_idle()
        assert encoder_lens is None and spec_info is None
        assert seq_lens_cpu is not None

        metadata = self.cuda_graph_metadata[bs]
        seq_lens = seq_lens[:bs]
        req_pool_indices = req_pool_indices[:bs]
        metadata.cache_seqlens_int32.copy_(seq_lens)
        metadata.max_seq_len_k = seq_lens_cpu[:bs].max().item()
        max_pages = (
            metadata.max_seq_len_k + self.page_size - 1
        ) // self.page_size
        page_indices = self.req_to_token[
            req_pool_indices[:, None],
            self.cuda_graph_metadata["strided_indices"][:max_pages][None, :],
        ]
        metadata.page_table[:, :max_pages].copy_(page_indices // self.page_size)
        self.forward_metadata = metadata

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    @staticmethod
    def _window_size(layer: RadixAttention) -> tuple[int, int]:
        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            return (layer.sliding_window_size, 0)
        return (-1, -1)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        assert not layer.is_cross_attention, "fapod does not support cross attention"
        assert layer.v_head_dim == layer.head_dim, (
            "fapod requires identical key and value head dimensions"
        )
        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer,
                    forward_batch.out_cache_loc,
                    k,
                    v,
                    layer.k_scale,
                    layer.v_scale,
                )

        metadata = self.forward_metadata
        key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
            layer.layer_id
        )
        key_cache = key_cache.view(
            -1, self.page_size, layer.tp_k_head_num, layer.head_dim
        )
        value_cache = value_cache.view(
            -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
        )
        window_size = self._window_size(layer)

        if self.pod_attn is not None and forward_batch.forward_mode.is_mixed():
            output = self._forward_mixed_pod(
                q,
                key_cache,
                value_cache,
                layer,
                forward_batch,
                window_size,
            )
            if output is not None:
                return output

        output, _ = torch.ops._vllm_fa2_C.varlen_fwd(
            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            key_cache,
            value_cache,
            None,
            metadata.cu_seqlens_q,
            torch.empty_like(metadata.cu_seqlens_q),
            metadata.cache_seqlens_int32,
            None,
            metadata.page_table,
            None,
            metadata.max_seq_len_q,
            metadata.max_seq_len_k,
            0.0,
            layer.scaling,
            False,
            True,
            window_size[0],
            window_size[1],
            layer.logit_cap,
            False,
            None,
        )
        return output.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _forward_mixed_pod(
        self,
        q: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        window_size: tuple[int, int],
    ) -> Optional[torch.Tensor]:
        prefill_count = forward_batch.mixed_prefill_count
        assert prefill_count is not None, "mixed batch is missing its prefill boundary"
        decode_count = forward_batch.batch_size - prefill_count
        if prefill_count == 0 or decode_count == 0:
            return None

        prefill_query_lens = forward_batch.extend_seq_lens_cpu[:prefill_count]
        prefill_query_len = prefill_query_lens[0]
        if any(length != prefill_query_len for length in prefill_query_lens):
            return None

        prefill_tokens = prefill_count * prefill_query_len
        q = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        q_prefill = q[:prefill_tokens].view(
            prefill_count,
            prefill_query_len,
            layer.tp_q_head_num,
            layer.head_dim,
        )
        q_decode = q[prefill_tokens:].view(
            decode_count, 1, layer.tp_q_head_num, layer.head_dim
        )

        metadata = self.forward_metadata
        
        if AUTO_SPLIT:
            kwargs = {}
        else:
            kwargs = {
                "num_splits_p": 1,
                "num_splits_d": 1,
            }
        
        output_prefill, output_decode = self.pod_attn.true_fused_attn_with_kvcache(
            q_prefill,
            key_cache,
            value_cache,
            q_decode,
            key_cache,
            value_cache,
            cache_seqlens_p=metadata.cache_seqlens_int32[:prefill_count],
            cache_seqlens_d=metadata.cache_seqlens_int32[prefill_count:],
            block_table_p=metadata.page_table[:prefill_count],
            block_table_d=metadata.page_table[prefill_count:],
            softmax_scale=layer.scaling,
            causal=True,
            window_size=window_size,
            softcap=layer.logit_cap,
            **kwargs,
        )
        return torch.cat(
            (output_prefill.flatten(0, 1), output_decode.flatten(0, 1)), dim=0
        ).view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        assert not layer.is_cross_attention, "fapod does not support cross attention"
        assert layer.v_head_dim == layer.head_dim, (
            "fapod requires identical key and value head dimensions"
        )
        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer,
                    forward_batch.out_cache_loc,
                    k,
                    v,
                    layer.k_scale,
                    layer.v_scale,
                )

        metadata = self.forward_metadata
        key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
            layer.layer_id
        )
        key_cache = key_cache.view(
            -1, self.page_size, layer.tp_k_head_num, layer.head_dim
        )
        value_cache = value_cache.view(
            -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
        )
        window_size = self._window_size(layer)

        output, _ = torch.ops._vllm_fa2_C.fwd_kvcache(
            q.contiguous().view(-1, 1, layer.tp_q_head_num, layer.head_dim),
            key_cache,
            value_cache,
            None,
            None,
            metadata.cache_seqlens_int32,
            None,
            None,
            None,
            None,
            metadata.page_table,
            None,
            None,
            layer.scaling,
            True,
            window_size[0],
            window_size[1],
            layer.logit_cap,
            True,
            0,
        )
        return output.view(-1, layer.tp_q_head_num * layer.v_head_dim)