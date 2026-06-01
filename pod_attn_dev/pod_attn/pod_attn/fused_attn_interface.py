
from typing import Optional, Union

import torch
import torch.nn as nn

# isort: off
# We need to import the CUDA kernels after importing torch
import fused_attn as fused_attn_cuda
import pod_attn.flash_attn_interface as fa

def true_fused_attn_with_kvcache(
    q_p,
    k_cache_p,
    v_cache_p,
    q_d,
    k_cache_d,
    v_cache_d,
    k=None, # Only for decode
    v=None, # Only for decode
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens_p: Optional[Union[(int, torch.Tensor)]] = None,
    cache_seqlens_d: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    block_table_p: Optional[torch.Tensor] = None,
    block_table_d: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits_p=0,
    num_splits_d=0,
    return_softmax_lse=False,
    fused_params=15,
):
    # If only one set of params is sent, call vanilla flash attention instead
    if q_p == None:
        return None, fa.flash_attn_with_kvcache(
            q_d,
            k_cache_d,
            v_cache_d,
            k=k,
            v=v,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            cache_seqlens=cache_seqlens_d,
            block_table=block_table_d,
            cache_batch_idx=cache_batch_idx,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,  # -1 means infinite context window
            rotary_interleaved=rotary_interleaved,
            alibi_slopes=alibi_slopes,
            num_splits=num_splits_d,
        )
    elif q_d == None:
        return fa.flash_attn_with_kvcache(
            q_p,
            k_cache_p,
            v_cache_p,
            k=k,
            v=v,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            cache_seqlens = cache_seqlens_p,
            block_table=block_table_p,
            cache_batch_idx=cache_batch_idx,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,  # -1 means infinite context window
            rotary_interleaved=rotary_interleaved,
            alibi_slopes=alibi_slopes,
            num_splits=num_splits_p,
        ), None
    # Prepare prefill data first
    assert k_cache_p.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache_p.stride(-1) == 1, "v_cache must have contiguous last dimension"
    maybe_contiguous = lambda x: x.contiguous() if x is not None and x.stride(-1) != 1 else x
    q_p = maybe_contiguous(q_p)
    if softmax_scale is None:
        softmax_scale = q_p.shape[-1] ** (-0.5)
    
    if cache_seqlens_p is not None and isinstance(cache_seqlens_p, int):
        cache_seqlens_p = torch.full(
            (k_cache_p.shape[0],), cache_seqlens_p, dtype=torch.int32, device=k_cache_p.device
        )
        cache_seqlens_p = maybe_contiguous(cache_seqlens_p)
    
    if cache_seqlens_d is not None and isinstance(cache_seqlens_d, int):
        cache_seqlens_d = torch.full(
            (k_cache_d.shape[0],), cache_seqlens_d, dtype=torch.int32, device=k_cache_d.device
        )
        cache_seqlens_d = maybe_contiguous(cache_seqlens_d)
    
    cache_batch_idx = maybe_contiguous(cache_batch_idx)
    block_table_p = maybe_contiguous(block_table_p)
    block_table_d = maybe_contiguous(block_table_d)

    # Now prepare decode data
    assert k_cache_d.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache_d.stride(-1) == 1, "v_cache must have contiguous last dimension"
    q_d, k, v = [maybe_contiguous(x) for x in (q_d, k, v)]

    out_prefill, out_decode = fused_attn_cuda.true_fused_fwd_kvcache(
        q_p,
        k_cache_p,
        v_cache_p,
        cache_seqlens_p,
        q_d,
        k_cache_d,
        v_cache_d,
        cache_seqlens_d,
        k,
        v,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        cache_leftpad,
        block_table_p,
        block_table_d,
        alibi_slopes,
        None,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        rotary_interleaved,
        num_splits_p,
        num_splits_d,
        fused_params,
    )
    return out_prefill, out_decode

def true_fused_nseg_attn_varlen_func(
    q_p,
    k_p,
    v_p,
    q_d,
    k_d,
    v_d,
    cu_seqlens_q_p,
    max_seqlen_q_p,
    max_seqlen_k_p,
    cu_seqlens_q_d,
    max_seqlen_q_d,
    max_seqlen_k_d,

    cu_seqlens_k_p=None,
    cu_seqlens_k_d=None,
    block_table_p: Optional[torch.Tensor] = None,
    block_table_d: Optional[torch.Tensor] = None,

    dropout_p=0.0,    
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    return_softmax=False,
    rotary_interleaved=True,
    alibi_slopes=None,
    leftpad_k=None,
    out_prefill=None,
    out_decode=None,
    seqused_k_p=None,
    seqused_k_d=None,
    num_splits_p=0,
    num_splits_d=0,
    fused_params=15,
    seqlens_q_block_p=None,
    seqused_k_block_p=None,
    blocks_in_batch_p=None,
):
        #     q_p,
        # k_p,
        # v_p,
        # q_d,
        # k_d,
        # v_d,
        # cu_seqlens_q_p,
        # max_seqlen_q_p,
        # max_seqlen_k_p,
        # cu_seqlens_q_d,
        # max_seqlen_q_d,
        # max_seqlen_k_d,

        # cu_seqlens_k_p=cu_seqlens_k_p,
        # seqused_k_d=seqused_k_d,
        # block_table_p=block_table_p,
        # block_table_d=block_table_d,
        # causal=True,
        # seqlens_q_block_p=seqlens_q_block_p,
        # seqused_k_block_p=seqused_k_block_p,
        # blocks_in_batch_p=blocks_in_batch_p,
    # with open("/jizhicfs/marvinhjia/sca/vllm-dev/examples/test/tmp.txt", "w") as f:
    #     torch.set_printoptions(profile="full")
    #     f.write(f'q_p={q_p.shape}'\
    #             f'k_p={k_p.shape}'\
    #             f'v_p={v_p.shape}'\
    #             f'q_d={q_d.shape}'\
    #             f'k_d={k_d.shape}'\
    #             f'v_d={v_d.shape}'\
    #             f'cu_q_p={cu_seqlens_q_p}'\
    #             f'mx_q_p={max_seqlen_q_p}'\
    #             f'mx_k_p={max_seqlen_k_p}'\
    #             f'cu_q_d={cu_seqlens_q_d}'\
    #             f'mx_q_d={max_seqlen_q_d}'\
    #             f'mx_k_d={max_seqlen_k_d}'\
    #             f'cu_k_p={cu_seqlens_k_p}'\
    #             f'seqused_k_d={seqused_k_d}'\
    #             f'block_table_p={block_table_p}'\
    #             f'block_table_d={block_table_d}'\
    #             f'seqlens_q_block_p={seqlens_q_block_p}'\
    #             f'seqused_k_block_p={seqused_k_block_p}'\
    #             f'blocks_in_batch_p={blocks_in_batch_p}\n')
    #     torch.set_printoptions(profile="default")
    # exit(0)
    if q_p is None:
        assert False
    elif q_d is None:
        assert False
        return fa.flash_nseg_attn_varlen_func(
            q_p,
            k_p,
            v_p,
            max_seqlen_q=max_seqlen_q_p,
            cu_seqlens_q=cu_seqlens_q_p,
            max_seqlen_k=max_seqlen_k_p,
            cu_seqlens_k=cu_seqlens_k_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            block_table=block_table_p,
            out=out_prefill,
            seqused_k=seqused_k_p,
            seqlens_q_block=seqlens_q_block_p,
            seqused_k_block=seqused_k_block_p,
            blocks_in_batch=blocks_in_batch_p,
        ), None
    maybe_contiguous = lambda x: x.contiguous() if x is not None and x.stride(-1) != 1 else x
    q_p, k_p, v_p = [maybe_contiguous(x) for x in (q_p, k_p, v_p)]
    q_d, k_d, v_d = [maybe_contiguous(x) for x in (q_d, k_d, v_d)]
    block_table_p = maybe_contiguous(block_table_p)
    block_table_d = maybe_contiguous(block_table_d)
    if softmax_scale is None:
        softmax_scale = q_p.shape[-1] ** (-0.5)
    assert k_p.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_p.stride(-1) == 1, "v_cache must have contiguous last dimension"
    assert k_d.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_d.stride(-1) == 1, "v_cache must have contiguous last dimension"
    return fused_attn_cuda.true_fused_nseg_varlen_fwd(
        q_p,
        k_p,
        v_p,
        cu_seqlens_q_p,
        cu_seqlens_k_p,
        max_seqlen_q_p,
        max_seqlen_k_p,
        q_d,
        k_d,
        v_d,
        cu_seqlens_q_d,
        cu_seqlens_k_d,
        max_seqlen_q_d,
        max_seqlen_k_d,
        leftpad_k,
        seqused_k_p,
        seqused_k_d,
        block_table_p,
        block_table_d,
        alibi_slopes,
        out_prefill,
        out_decode,
        dropout_p,
        softmax_scale,
        False,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        return_softmax,
        None,
        num_splits_p,
        num_splits_d,
        fused_params,
        seqlens_q_block_p,
        seqused_k_block_p,
        blocks_in_batch_p,
    )

def true_fused_nseg_attn_with_kvcache(
    q_p,
    k_cache_p,
    v_cache_p,
    q_d,
    k_cache_d,
    v_cache_d,
    k=None, # Only for decode
    v=None, # Only for decode
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens_p: Optional[Union[(int, torch.Tensor)]] = None,
    cache_seqlens_d: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    block_table_p: Optional[torch.Tensor] = None,
    block_table_d: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits_p=0,
    num_splits_d=0,
    return_softmax_lse=False,
    fused_params=15,
    seqlens_q_block_p=None,
    seqused_k_block_p=None,
    blocks_in_batch_p=None,
    seqlens_q_block_d=None,
    seqused_k_block_d=None,
    blocks_in_batch_d=None,
):
    # If only one set of params is sent, call vanilla flash attention instead
    if q_p == None:
        return None, fa.flash_nseg_attn_with_kvcache(
            q_d,
            k_cache_d,
            v_cache_d,
            k=k,
            v=v,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            cache_seqlens=cache_seqlens_d,
            block_table=block_table_d,
            cache_batch_idx=cache_batch_idx,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,  # -1 means infinite context window
            rotary_interleaved=rotary_interleaved,
            alibi_slopes=alibi_slopes,
            num_splits=num_splits_d,
            seqlens_q_block=seqlens_q_block_d,
            seqused_k_block=seqused_k_block_d,
            blocks_in_batch=blocks_in_batch_d,
        )
    elif q_d == None:
        return fa.flash_nseg_attn_with_kvcache(
            q_p,
            k_cache_p,
            v_cache_p,
            k=k,
            v=v,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            cache_seqlens = cache_seqlens_p,
            block_table=block_table_p,
            cache_batch_idx=cache_batch_idx,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,  # -1 means infinite context window
            rotary_interleaved=rotary_interleaved,
            alibi_slopes=alibi_slopes,
            num_splits=num_splits_p,
            seqlens_q_block=seqlens_q_block_p,
            seqused_k_block=seqused_k_block_p,
            blocks_in_batch=blocks_in_batch_p,
        ), None
    # Prepare prefill data first
    assert k_cache_p.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache_p.stride(-1) == 1, "v_cache must have contiguous last dimension"
    maybe_contiguous = lambda x: x.contiguous() if x is not None and x.stride(-1) != 1 else x
    q_p = maybe_contiguous(q_p)
    if softmax_scale is None:
        softmax_scale = q_p.shape[-1] ** (-0.5)
    
    if cache_seqlens_p is not None and isinstance(cache_seqlens_p, int):
        cache_seqlens_p = torch.full(
            (k_cache_p.shape[0],), cache_seqlens_p, dtype=torch.int32, device=k_cache_p.device
        )
        cache_seqlens_p = maybe_contiguous(cache_seqlens_p)
    
    if cache_seqlens_d is not None and isinstance(cache_seqlens_d, int):
        cache_seqlens_d = torch.full(
            (k_cache_d.shape[0],), cache_seqlens_d, dtype=torch.int32, device=k_cache_d.device
        )
        cache_seqlens_d = maybe_contiguous(cache_seqlens_d)
    
    cache_batch_idx = maybe_contiguous(cache_batch_idx)
    block_table_p = maybe_contiguous(block_table_p)
    block_table_d = maybe_contiguous(block_table_d)

    # Now prepare decode data
    assert k_cache_d.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache_d.stride(-1) == 1, "v_cache must have contiguous last dimension"
    q_d, k, v = [maybe_contiguous(x) for x in (q_d, k, v)]

    out_prefill, out_decode = fused_attn_cuda.true_fused_nseg_fwd_kvcache(
        q_p,
        k_cache_p,
        v_cache_p,
        cache_seqlens_p,
        q_d,
        k_cache_d,
        v_cache_d,
        cache_seqlens_d,
        k,
        v,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        cache_leftpad,
        block_table_p,
        block_table_d,
        alibi_slopes,
        None,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        rotary_interleaved,
        num_splits_p,
        num_splits_d,
        fused_params,
        seqlens_q_block_p,
        seqused_k_block_p,
        blocks_in_batch_p,
        seqlens_q_block_d,
        seqused_k_block_d,
        blocks_in_batch_d,
    )
    return out_prefill, out_decode
def fused_attn_with_kvcache(
    q_p,
    k_cache_p,
    v_cache_p,
    q_d,
    k_cache_d,
    v_cache_d,
#    k=None,
#    v=None,
    rotary_cos=None,
    rotary_sin=None,
#    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,
    return_softmax_lse=False,
    ratio=0,
):
    # TODO: Add support for k, v insertion later
    k = None
    v = None
    cache_seqlens = None
    # Prepare prefill data first
    assert k_cache_p.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache_p.stride(-1) == 1, "v_cache must have contiguous last dimension"
    maybe_contiguous = lambda x: x.contiguous() if x is not None and x.stride(-1) != 1 else x
    q_p, k, v = [maybe_contiguous(x) for x in (q_p, k, v)]
    if softmax_scale is None:
        softmax_scale = q_p.shape[-1] ** (-0.5)
    '''
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (k_cache.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )
        cache_seqlens = maybe_contiguous(cache_seqlens)
    '''
    cache_batch_idx = maybe_contiguous(cache_batch_idx)
    block_table = maybe_contiguous(block_table)


    # Now prepare decode data
    assert k_cache_d.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache_d.stride(-1) == 1, "v_cache must have contiguous last dimension"
    maybe_contiguous = lambda x: x.contiguous() if x is not None and x.stride(-1) != 1 else x
    q_d, k, v = [maybe_contiguous(x) for x in (q_d, k, v)]

    out, softmax_lse = fused_attn_cuda.fused_fwd_kvcache(
        q_p,
        k_cache_p,
        v_cache_p,
        q_d,
        k_cache_d,
        v_cache_d,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        cache_leftpad,
        block_table,
        alibi_slopes,
        None,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        rotary_interleaved,
        num_splits,
        ratio,
    )
    return (out, softmax_lse) if return_softmax_lse else out
