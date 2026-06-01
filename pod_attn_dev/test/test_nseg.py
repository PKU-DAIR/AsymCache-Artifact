import pod_attn
import torch
batch_size_p = 2
seqlen_p = 1
seqlen_kv_p = 1
nheads = 32
kv_nheads = 8
headdim = 128
q_p = torch.randn((batch_size_p, seqlen_p, nheads, headdim), device="cuda").to(torch.float16)
k_cache_p = torch.randn((batch_size_p, seqlen_kv_p, kv_nheads, headdim), device="cuda").to(torch.float16)
v_cache_p = torch.randn((batch_size_p, seqlen_kv_p, kv_nheads, headdim), device="cuda").to(torch.float16)

batch_size_d = 1
seqlen_kv_d = 10
q_d = torch.randn((batch_size_d, 1, nheads, headdim), device="cuda").to(torch.float16)
k_cache_d = torch.randn((batch_size_d, seqlen_kv_d, kv_nheads, headdim), device="cuda").to(torch.float16)
v_cache_d = torch.randn((batch_size_d, seqlen_kv_d, kv_nheads, headdim), device="cuda").to(torch.float16)

seqlens_q_block_p=torch.tensor([4, 4], dtype=torch.int32, device="cuda")
seqused_k_block_p=torch.tensor([4, 4], dtype=torch.int32, device="cuda")
blocks_in_batch_p=torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
def get_result_pod_attn():
    out_p, out_d = pod_attn.true_fused_nseg_attn_with_kvcache(
        q_p, k_cache_p, v_cache_p, q_d, k_cache_d, v_cache_d, k=None, v=None,
        rotary_cos=None, rotary_sin=None, cache_seqlens_p = None,
        cache_seqlens_d = None, cache_batch_idx = None,
        softmax_scale=None, causal=False, window_size=(-1, -1), rotary_interleaved=True, fused_params=15,
        seqlens_q_block_p = seqlens_q_block_p,
        seqused_k_block_p= seqused_k_block_p,
        blocks_in_batch_p=blocks_in_batch_p,
        seqlens_q_block_d=None,
        seqused_k_block_d=None,
        blocks_in_batch_d=None,)
    # out_p, out_d = pod_attn.true_fused_nseg_attn_with_kvcache(
    #     q_p, k_cache_p, v_cache_p, q_d, k_cache_d, v_cache_d, k=None, v=None,
    #     rotary_cos=None, rotary_sin=None, cache_seqlens_p = None,
    #     cache_seqlens_d = None, cache_batch_idx = None,
    #     softmax_scale=None, causal=False, window_size=(-1, -1), rotary_interleaved=True, fused_params=15,
    #     seqlens_q_block_p = seqlens_q_block_p,
    #     seqused_k_block_p= seqused_k_block_p,
    #     blocks_in_batch_p=blocks_in_batch_p,
    #     seqlens_q_block_d=seqlens_q_block_d,
    #     seqused_k_block_d=seqused_k_block_d,
    #     blocks_in_batch_d=blocks_in_batch_d,)
    return out_p, out_d

# import flash_attn
import pod_attn.flash_attn_interface as fa
def get_result_flash_attn():
    out_p = fa.flash_attn_with_kvcache(
        q_p, k_cache_p, v_cache_p, k=None, v=None,
        rotary_cos=None, rotary_sin=None)
    # out_p_first = flash_attn.flash_attn_with_kvcache(
    #     q_p[:, :256, :, :], k_cache_p[:, :512, :, :], v_cache_p[:, :512, :, :], k=None, v=None,
    #     cache_seqlens = None,
    #     rotary_cos=None, rotary_sin=None
    # )
    # out_p_second = flash_attn.flash_attn_with_kvcache(
    #     q_p[:, 256:, :, :], k_cache_p[:, :2048, :, :], v_cache_p[:, :2048, :, :], k=None, v=None,
    #     cache_seqlens = None,
    #     rotary_cos=None, rotary_sin=None
    # )
    out_d = fa.flash_attn_with_kvcache(
        q_d, k_cache_d, v_cache_d, k=None, v=None,
        rotary_cos=None, rotary_sin=None)
    return out_p, out_d
    # return torch.cat((out_p_first, out_p_second), dim=1), out_d

# print(get_result_pod_attn())
# print(get_result_flash_attn())
r1 = get_result_pod_attn()
r2 = get_result_flash_attn()
print(torch.allclose(r1[0], r2[0], 1e-3, 1e-3))
print(torch.allclose(r1[1], r2[1], 1e-3, 1e-3))

# print("first part:", torch.allclose(r1[0][:, :256, :, :], r2[0][:, :256, :, :], 1e-3, 1e-3))

# print("second part:", torch.allclose(r1[0][:, 256:, :, :], r2[0][:, 256:, :, :], 1e-3, 1e-3))
# print(r1[0][:1, 254:260, :1, :1])
# print(r2[0][:1, 254:260, :1, :1])