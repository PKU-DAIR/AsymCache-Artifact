import pandas as pd
import re

def load_data(path = '../results/summary.csv', v2=False):
    data_file = path
    df = pd.read_csv(data_file, header=None)

    dataset = {
        "longbench": {
            "8B": {},
            "70B": {},
        },
        "loogle": {
            "8B": {},
            "70B": {},
        },
    }


    for index, row in df.iterrows():
        first_col_value = str(row[0])
        
        # ignore cases
        first_col_value = first_col_value.lower()
        
        if v2 and first_col_value.startswith('300_'):
            first_col_value = first_col_value[4:]
        
        if not (first_col_value.startswith('longbench_') or first_col_value.startswith('loogle_')):
            continue
        
        if first_col_value.endswith('.jsonl'):
            first_col_value = first_col_value[:-6]
        
        tokens = first_col_value.split('_')
        dataset_name = tokens[0].lower()
        model = tokens[-1].strip().upper()
        if v2 and not model in ['8B', '70B'] and len(tokens) >= 6:
            model = tokens[5].strip().upper()
        
        entry = dataset[dataset_name][model]
        
        pattern_frac = r'frac(\d+)'
        match_frac = re.search(pattern_frac, first_col_value)
        if not match_frac:
            print(f"[line {index}] 无法从字符串中提取 frac: {first_col_value}")
            continue
        frac = match_frac.group(1)
        
        if frac not in entry:
            entry[frac] = {}
        entry = entry[frac]
        
        
        approach = tokens[2].lower()
        if approach == "mwf":
            approach = 'minexp'
        elif 'pensi' in approach:
            if v2:
                continue
            approach = 'pensi'
        elif v2 and approach == 'no':
            approach = 'pensi'
        
        if approach not in entry:
            entry[approach] = {}
        entry = entry[approach]
        
        # extract qps
        pattern = r'qps(0\.\d+)'
        match = re.search(pattern, first_col_value)
        if not match:
            print(f"[line {index}] 无法从字符串中提取 qps: {first_col_value}")
            continue
        qps = float(match.group(1))
        # print(f"[line {index}] 提取到 qps: {qps} from {first_col_value}")
        
        if qps in entry:
            print(f"[line {index}] 警告: 重复的 qps 条目: {first_col_value} (qps={qps})")
            raise ValueError(f"重复的 qps 条目: {first_col_value} (qps={qps})")
        
        res = {}
        
        ttft, tpot = float(row[3]), float(row[4])
        hit_items = int(row[7])
        hit_rate = float(row[10])
        total_time, tpt = float(row[13]), float(row[14])
        
        res["ttft"] = ttft
        res["tpot"] = tpot
        res["hit_items"] = hit_items
        res["hit_rate"] = hit_rate
        res["total_time"] = total_time
        res["tpt"] = tpt
        
        print(f"Loaded line {index}: {first_col_value} {frac} ({res})")
        
        entry[qps] = res
    return dataset

def load_data_sense(path = './data/实验结果.csv'):
    data_file = path
    df = pd.read_excel(data_file, sheet_name='sense-sheet', header=None)
    
    dataset = {i: [] for i in range(7)}
    
    def parse_row(row):
        res = {}
        
        ttft, tpot = float(row[3]), float(row[4])
        hit_items = int(row[7])
        hit_rate = float(row[10])
        total_time, tpt = float(row[13]), float(row[14])
        
        res["ttft"] = ttft
        res["tpot"] = tpot
        res["hit_items"] = hit_items
        res["hit_rate"] = hit_rate
        res["total_time"] = total_time
        res["tpt"] = tpt
        return res
    
    
    status = 0
    for index, row in df.iterrows():
        first_col_value = str(row[0]).lower()
        
        if 'lru' in first_col_value:
            res = parse_row(row)
            
            dataset[0].append({'ttft': res['ttft'], 'tpot': res['tpot'], 'hit_rate': res['hit_rate']})
            continue
        if first_col_value.startswith('subplot'):
            # new subplot
            status += 1
            continue
        if not first_col_value.startswith('longbench'):
            continue
        try:
            res = parse_row(row)
        except Exception as e:
            print(f"[line {index}] 解析行时出错: {e}")
            print(f"Row content: {row}")
            raise e
        dataset[status].append(res)
        
    dataset2 = {i: [] for i in range(1, 7)}
    dataset2[0] = dataset[0]  # LRU 数据不变
    for i in range(1, 7):
        # ttft
        dataset2[i].append([entry['ttft'] for entry in dataset[i]])
        # tpot
        dataset2[i].append([entry['tpot'] for entry in dataset[i]])
        # hit_rate
        dataset2[i].append([entry['hit_rate'] for entry in dataset[i]])
        
    return dataset2

def load_data_bfcl(path = '../vllm-dev/examples/test/e2e-exp/reproduce/final_bfcl_24_summary.csv'):
    data_file = path
    df = pd.read_csv(data_file, header=None)
    pattern = r'2x_(.*)_qps(\d+\.\d+)'
    
    dataset = {
        '70b': {},
        '8b': {},
    }

    def parse_row(row):
        res = {}
        
        ttft, tpot = float(row[3]), float(row[4])
        hit_items = int(row[7])
        hit_rate = float(row[10])
        job_avg, job_p90  = float(row[14]), float(row[15])
        total_time, tpt = float(row[19]), float(row[20])
        
        res["ttft"] = ttft
        res["tpot"] = tpot
        res["hit_items"] = hit_items
        res["hit_rate"] = hit_rate * 100  # 转换为百分比
        res["job_avg"] = job_avg
        res["job_p90"] = job_p90
        res["total_time"] = total_time
        res["tpt"] = tpt
        return res
    
    current_dataset = None
    for index, row in df.iterrows():
        first_col_value = str(row[0]).lower()
        if first_col_value in dataset.keys():
            current_dataset = dataset[first_col_value]
            continue
        
        assert current_dataset is not None, f"在解析行 {index} 时，尚未确定当前数据集 (70b 或 8b)"
        m = re.match(pattern, first_col_value)
        if not m:
            continue
        
        approach = m.group(1)
        qps = m.group(2) # string
        
        if not qps in current_dataset:
            current_dataset[qps] = {}
        
        if approach in current_dataset[qps]:
            print(f"[line {index}] 警告: 重复的条目: {first_col_value} (qps={qps}, approach={approach})")
            raise ValueError(f"重复的条目: {first_col_value} (qps={qps}, approach={approach})")
        
        try:
            res = parse_row(row)
        except Exception as e:
            print(f"[line {index}] 解析行时出错: {e}")
            print(f"Row content: {row}")
            raise e
        
        current_dataset[qps][approach] = res
        print(f"Loaded line {index}: {first_col_value} (qps={qps}, approach={approach}, res={res})")
        
    return dataset

if __name__ == "__main__":
    # dataset = load_data_sense()
    
    # import json
    # with open('./data/parsed_data.json', 'w') as f:
    #     json.dump(dataset, f, indent=4)
    
    dataset = load_data_bfcl()
    import json
    with open('./data/parsed_data_bfcl.json', 'w') as f:
        json.dump(dataset, f, indent=4)
    