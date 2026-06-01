import pod_attn
import torch
import pod_attn.flash_attn_interface as fa
import random
import numpy as np
import os
# 设置随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
os.environ["CUDA_VISIBLE_DEVICES"]="7"
head_num=32
kv_head_num=32
block_size=64
block_num=50

prefill = 10
decode = 10
print(f"decode:{decode}")
q_p = torch.randn((prefill, head_num, 128), device="cuda").to(torch.float16)
k_p = torch.randn((block_num, block_size, kv_head_num, 128), device="cuda").to(torch.float16)
# k_p_2 = k_p.detach().clone()
v_p = torch.randn((block_num, block_size, kv_head_num, 128), device="cuda").to(torch.float16)
# v_p_2 = v_p.detach().clone()
cu_seqlens_q_p = torch.tensor([0, 2, 4, 6, 8, 10], device="cuda", dtype=torch.int32)
max_seqlen_q_p = 2
cu_seqlens_k_p = torch.tensor([2, 2, 2, 2, 2], device="cuda", dtype=torch.int32)
max_seqlen_k_p = 2

q_d = torch.randn((decode, head_num, 128), device="cuda").to(torch.float16)
k_d = k_p#torch.randn((block_num, block_size, head_num, 128), device="cuda").to(torch.float16)
# k_d_2 = k_d.detach().clone()
v_d = v_p#torch.randn((block_num, block_size, head_num, 128), device="cuda").to(torch.float16)
# v_d_2 = v_d.detach().clone()
cu_seqlens_q_d = torch.tensor(list(range(0,decode+1)), device="cuda", dtype=torch.int32)
max_seqlen_q_d = 1

seqused_k_d = torch.tensor([10]*decode, device="cuda", dtype=torch.int32)#([9, 9, 9, 9, 8, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4,3, 3, 3, 3], device="cuda", dtype=torch.int32)
max_seqlen_k_d = 10

block_table_p = torch.tensor([[28], [29], [30], [31], [32]], device="cuda", dtype=torch.int32)
block_table_d = torch.tensor([[i] for i in range(decode)], device="cuda", dtype=torch.int32)

out_p = torch.randn((prefill, head_num, 128), device="cuda", dtype=torch.float16)
out_d = torch.randn((decode, head_num, 128), device="cuda", dtype=torch.float16)

out_p_f = torch.randn((prefill, head_num, 128), device="cuda", dtype=torch.float16)
out_d_f = torch.randn((decode, head_num, 128), device="cuda", dtype=torch.float16)

ret_0 = torch.randn((prefill+decode, head_num, 128), device="cuda", dtype=torch.float16)
ret_1 = torch.randn((prefill+decode, head_num, 128), device="cuda", dtype=torch.float16)

seqlens_q_block_p=torch.tensor([2, 2, 2, 2, 2], dtype=torch.int32, device="cuda")
seqused_k_block_p=torch.tensor([2, 2, 2, 2, 2], dtype=torch.int32, device="cuda")
blocks_in_batch_p=torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32, device="cuda")
def get_result_pod_attn():
    pod_attn.true_fused_nseg_attn_varlen_func(
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

        cu_seqlens_k_p=cu_seqlens_k_p,
        seqused_k_d=seqused_k_d,
        block_table_p=block_table_p,
        block_table_d=block_table_d,
        causal=True,
        out_prefill=out_p,
        out_decode=out_d,
        seqlens_q_block_p=seqlens_q_block_p,
        seqused_k_block_p=seqused_k_block_p,
        blocks_in_batch_p=blocks_in_batch_p,
    )
    ret_0[:prefill].copy_(out_p)
    ret_0[prefill:].copy_(out_d)

def get_result_flash_attn(): 
    out_p_f = fa.flash_attn_with_kvcache(
        q_p.reshape(5, 2, head_num, 128), k_d[28:28+prefill,:2,:,:], v_d[28:28+prefill,:2,:,:], k=None, v=None, causal=True,
        rotary_cos=None, rotary_sin=None).reshape(10, head_num, 128)
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
    out_d_f = fa.flash_attn_with_kvcache(
        q_d.unsqueeze(1), k_d[0:decode,:10,:,:], v_d[0:decode,:10,:,:], k=None, v=None, causal=True,
        rotary_cos=None, rotary_sin=None).squeeze(1)
    # fa.flash_attn_varlen_func(
    #     q_d,
    #     k_d,
    #     v_d,
    #     max_seqlen_q=max_seqlen_q_d,
    #     cu_seqlens_q=cu_seqlens_q_d,
    #     max_seqlen_k=max_seqlen_k_d,
    #     seqused_k=seqused_k_d,
    #     causal=True,
    #     block_table=block_table_d,
    #     out=out_d_f,
    # )
    # fa.flash_nseg_attn_varlen_func(
    #     q_p,
    #     k_p,
    #     v_p,
    #     max_seqlen_q=max_seqlen_q_p,
    #     cu_seqlens_q=cu_seqlens_q_p,
    #     max_seqlen_k=max_seqlen_k_p,
    #     cu_seqlens_k=cu_seqlens_k_p,
    #     causal=True,
    #     block_table=block_table_p,
    #     out=out_p_f,
    #     seqlens_q_block=seqlens_q_block_p,
    #     seqused_k_block=seqused_k_block_p,
    #     blocks_in_batch=blocks_in_batch_p,
    # )
    ret_1[:prefill].copy_(out_p_f)
    ret_1[prefill:].copy_(out_d_f)

# print(get_result_pod_attn())
# print(get_result_flash_attn())
get_result_pod_attn()
get_result_flash_attn()

print(ret_0)
torch.save(ret_0, 'ret_0.pt')
torch.save(ret_1, 'ret_1.pt')
print(torch.allclose(ret_0, ret_1, 1e-3, 1e-3))