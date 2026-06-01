import pod_attn
import torch
import pandas as pd

test_repeat = 1

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
def test_true_fused_attn_with_kvcache(
    q_p, k_cache_p, v_cache_p, q_d, k_cache_d, v_cache_d, cache_seqlens_p, cache_seqlens_d, k_new, v_new):
    # out_p, out_d = pod_attn.true_fused_attn_with_kvcache(
    #     q_p, k_cache_p, v_cache_p, q_d, k_cache_d, v_cache_d, k=k_new, v=v_new, causal=True,
    #     cache_seqlens_p = cache_seqlens_p, cache_seqlens_d = cache_seqlens_d, 
    #     cache_batch_idx = None,
    #     fused_params=15)
    start.record()
    for _ in range(test_repeat):
        out_p, out_d = pod_attn.true_fused_attn_with_kvcache(
            q_p, k_cache_p, v_cache_p, q_d, k_cache_d, v_cache_d, k=k_new, v=v_new, causal=True,
            cache_seqlens_p = cache_seqlens_p, cache_seqlens_d = cache_seqlens_d, 
            cache_batch_idx = None,
            fused_params=15)
    end.record()
    torch.cuda.synchronize()
    duration = round(start.elapsed_time(end) / test_repeat, 3)
    return out_p, out_d, duration

clen = 16384 # 2^14
# d_len = 256
nheads = 32
nheads_kv = 8
head_dim = 128

d_len_list = [2**i for i in range(7, 16)]
d_batch_sizes = [1, 2, 4, 8, 16, 32, 64]
chunk_sizes = [2**i for i in range(7,14)]

tps = [1, 2, 4, 8]
count = 0
for d_len in d_len_list:
    for tp in tps:
        results = {}
        for bs in d_batch_sizes:
            results[bs] = {}
            for chunk_size in chunk_sizes:
                results[bs][chunk_size] = {}
                for chunk_idx in range(clen // chunk_size):
                    cache_seqlen = (chunk_idx + 1) * chunk_size

                    tot = 0
                    for tp_idx in range(tp):
                        q_p = torch.randn(1, chunk_size, nheads//tp, head_dim, device="cuda", dtype=torch.float16)
                        k_cache_p = torch.randn(1,cache_seqlen, nheads_kv//tp, head_dim, device="cuda", dtype=torch.float16)
                        v_cache_p = torch.randn(1,cache_seqlen, nheads_kv//tp, head_dim, device="cuda", dtype=torch.float16)
                        cache_seqlens_p = torch.tensor([cache_seqlen], device="cuda", dtype=torch.int32)

                        if bs > 0:
                            q_d = torch.randn(bs, 1, nheads//tp, head_dim, device="cuda", dtype=torch.float16)
                            k_cache_d = torch.randn(bs, d_len, nheads_kv//tp, head_dim, device="cuda", dtype=torch.float16)
                            v_cache_d = torch.randn(bs, d_len, nheads_kv//tp, head_dim, device="cuda", dtype=torch.float16)
                            cache_seqlens_d = torch.tensor([d_len -1] * bs, device="cuda", dtype=torch.int32)
                        else:
                            q_d = None
                            k_cache_d = None
                            v_cache_d = None
                            cache_seqlens_d = None

                        k_new = None
                        v_new = None

                        fu_prefill, fu_decode, fused_latency = test_true_fused_attn_with_kvcache(
                            q_p, k_cache_p, v_cache_p,
                            q_d, k_cache_d, v_cache_d,
                            cache_seqlens_p, cache_seqlens_d, k_new, v_new)
                        count += 1
                        tot += fused_latency
                        break

                    results[bs][chunk_size][chunk_idx+1] = tot
                    # print(fused_latency)
import sys
print(f"count={count}", file=sys.stderr)

# tot = []
# with open("/home/infer/chenyilei/vllm_pod/vattention/benchmark/threadblocks/threadblocks.txt") as f:
#     for s in f:
#         assert len(s) > 0
#         tot.append(s)

# print(f"tot={len(tot)}", file=sys.stderr)
# ptr = 0
# for tp in tps:
#     results = {}
#     for bs in d_batch_sizes:
#         results[bs] = {}
#         for chunk_size in chunk_sizes:
#             results[bs][chunk_size] = {}
#             for chunk_idx in range(clen // chunk_size):
#                 cache_seqlen = (chunk_idx + 1) * chunk_size
#                 results[bs][chunk_size][chunk_idx+1]=tot[ptr]
#                 ptr+=1
#     all_data = []
#     # 为每个bs创建并保存表格
#     for bs, chunk_data in results.items():
#         df = pd.DataFrame.from_dict(chunk_data, orient='index')
#         df = df.sort_index(axis=0)  # 行按chunk_size排序
#         df = df.sort_index(axis=1)  # 列按chunk_num排序
#         df.index.name = 'chunk_size'
#         df.columns.name = 'chunk_num'

#         # filename = f"latency_bs{bs}.csv"
#         # df.to_csv(filename)
#         marker_row = pd.DataFrame([[bs] + [""] * (df.shape[1] - 1)])
#         combined = pd.concat([marker_row, df], ignore_index=True)

#         all_data.append(combined)
#     final_df = pd.concat(all_data, ignore_index=True)
#     final_df.to_csv(f"threadblocks_tp={tp}.csv", index=False, header=False)





