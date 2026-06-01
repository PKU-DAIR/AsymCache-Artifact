import pod_attn
import torch
import pandas as pd
import os
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
import sys
tot = []
with open("threadblocks.txt") as f:
    for s in f:
        assert len(s) > 0
        tot.append(s)

print(f"tot={len(tot)}", file=sys.stderr)
ptr = 0
for d_len in d_len_list:
    dir = f"../d_len={d_len}"
    if not os.path.exists(dir):
        os.makedirs(dir)
    for tp in tps:
        results = {}
        for bs in d_batch_sizes:
            results[bs] = {}
            for chunk_size in chunk_sizes:
                results[bs][chunk_size] = {}
                for chunk_idx in range(clen // chunk_size):
                    cache_seqlen = (chunk_idx + 1) * chunk_size
                    if bs == 0:
                        results[bs][chunk_size][chunk_idx+1]=0
                    else:
                        results[bs][chunk_size][chunk_idx+1]=tot[ptr]
                        ptr+=1
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
        final_df.to_csv(f"{dir}/threadblocks_tp={tp}.csv", index=False, header=False)

