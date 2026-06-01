import pod_attn
import torch
import torch.nn as nn
import pandas as pd
test_repeat = 32

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

hidden_dim = 32 * 128
hidden_dim_kv = 8 * 128
clen = 16384 # 2^14
d_len = 256
nheads = 32
nheads_kv = 8
head_dim = 128

d_batch_sizes = list(range(0, 65))
chunk_sizes = [2**i for i in range(7,14)]


def test_true_fused_attn_with_kvcache(
    cache_seqlen, input_p, input_p_kv, input_d, k_cache_p, v_cache_p, fc_q, fc_k, cache_seqlens_p, k_cache_d, v_cache_d, cache_seqlens_d):

    q_p = fc_q(input_p).view(1, -1, nheads, head_dim)
    if chunk_idx > 0:
        tmp_k = fc_k(input_p_kv).view(1, -1, nheads_kv, head_dim)
        k_cache_p = torch.cat((k_cache_p, tmp_k), dim=1)
        tmp_v = fc_k(input_p_kv).view(1, -1, nheads_kv, head_dim)
        v_cache_p = torch.cat((v_cache_p, tmp_v), dim=1)
    else:
        tmp_k = fc_k(input_p_kv).view(1, -1, nheads_kv, head_dim)
        k_cache_p = tmp_k
        tmp_v = fc_k(input_p_kv).view(1, -1, nheads_kv, head_dim)
        v_cache_p = tmp_v

    cache_seqlens_p = torch.tensor([cache_seqlen], device="cuda", dtype=torch.int32)

    if bs > 0:
        q_d = fc_q(input_d).view(bs, 1, nheads, head_dim)
    else:
        q_d = None
        k_cache_d = None
        v_cache_d = None
        cache_seqlens_d = None

    k_new = None
    v_new = None

    out_p, out_d = pod_attn.true_fused_attn_with_kvcache(
        q_p, k_cache_p, v_cache_p, q_d, k_cache_d, v_cache_d, k=k_new, v=v_new, causal=True,
        cache_seqlens_p = cache_seqlens_p, cache_seqlens_d = cache_seqlens_d, 
        cache_batch_idx = None,
        fused_params=15)
    # print(out_p.shape)
    if out_p is not None:
        out_p = fc_q(torch.relu(fc_q(out_p.view(1, -1, hidden_dim))))
    if out_d is not None:
        out_d = fc_q(torch.relu(fc_q(out_d.view(1, -1, hidden_dim))))


def test(bs, chunk_size, chunk_idx):
    cache_seqlen = (chunk_idx + 1) * chunk_size
    input_p = torch.randn(1, chunk_size, hidden_dim, device="cuda", dtype=torch.float16)
    input_p_kv = torch.randn(1, chunk_size, hidden_dim_kv, device="cuda", dtype=torch.float16)
    input_d = torch.randn(bs, 1, hidden_dim, device="cuda", dtype=torch.float16)
    if chunk_idx > 0:
        k_cache_p = torch.randn(1,cache_seqlen-chunk_size, nheads_kv, head_dim, device="cuda", dtype=torch.float16)
        v_cache_p = torch.randn(1,cache_seqlen-chunk_size, nheads_kv, head_dim, device="cuda", dtype=torch.float16)
    else:
        k_cache_p = None
        v_cache_p = None
    fc_q = nn.Linear(hidden_dim, hidden_dim, device="cuda", dtype=torch.float16)
    fc_k = nn.Linear(hidden_dim_kv, hidden_dim_kv, device="cuda", dtype=torch.float16)
    cache_seqlens_p = torch.tensor([cache_seqlen], device="cuda", dtype=torch.int32)

    if bs > 0:        
        k_cache_d = torch.randn(bs, d_len, nheads_kv, head_dim, device="cuda", dtype=torch.float16)
        v_cache_d = torch.randn(bs, d_len, nheads_kv, head_dim, device="cuda", dtype=torch.float16)
        cache_seqlens_d = torch.tensor([d_len -1] * bs, device="cuda", dtype=torch.int32)
    else:
        k_cache_d = None
        v_cache_d = None
        cache_seqlens_d = None

    test_true_fused_attn_with_kvcache(cache_seqlen, input_p, input_p_kv, input_d, k_cache_p, v_cache_p, fc_q, fc_k, cache_seqlens_p, k_cache_d, v_cache_d, cache_seqlens_d)
    start.record()
    for _ in range(test_repeat):
        test_true_fused_attn_with_kvcache(cache_seqlen, input_p, input_p_kv, input_d, k_cache_p, v_cache_p, fc_q, fc_k, cache_seqlens_p, k_cache_d, v_cache_d, cache_seqlens_d)
    end.record()
    torch.cuda.synchronize()
    duration = round(start.elapsed_time(end) / test_repeat, 3)
    return duration
results = {}
for bs in d_batch_sizes:
    results[bs] = {}
    for chunk_size in chunk_sizes:
        results[bs][chunk_size] = {}
        for chunk_idx in range(clen // chunk_size):
            fused_latency = test(bs, chunk_size, chunk_idx)

            results[bs][chunk_size][chunk_idx+1] = fused_latency
            print(fused_latency)
all_data = []
# 为每个bs创建并保存表格
for bs, chunk_data in results.items():
    df = pd.DataFrame.from_dict(chunk_data, orient='index')
    df = df.sort_index(axis=0)  # 行按chunk_size排序
    df = df.sort_index(axis=1)  # 列按chunk_num排序
    df.index.name = 'chunk_size'
    df.columns.name = 'chunk_num'

    # filename = f"latency_bs{bs}.csv"
    # df.to_csv(filename)
    marker_row = pd.DataFrame([[bs] + [""] * (df.shape[1] - 1)])
    combined = pd.concat([marker_row, df], ignore_index=True)

    all_data.append(combined)
final_df = pd.concat(all_data, ignore_index=True)
final_df.to_csv(f"latency_transformer_block.csv", index=False, header=False)





