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
        
        if 'hotprefix' in first_col_value:
            continue
        
        if v2 and first_col_value.startswith('300_'):
            first_col_value = first_col_value[4:]
        
        if not (first_col_value.startswith('longbench_') or first_col_value.startswith('loogle_')):
            continue
        if first_col_value.endswith(" new"):
            first_col_value = first_col_value[:-4]
        if first_col_value.endswith('.jsonl'):
            first_col_value = first_col_value[:-6]
        
        tokens = first_col_value.split('_')
        dataset_name = tokens[0].lower()
        model = tokens[-1].strip().upper()
        if v2 and (not model in ['8B', '70B']) and len(tokens) >= 6:
            # print(f"[line {index}] 非标准模型名称: {first_col_value}")
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

def load_new_data(dataset, path_prefix='./data/summary-'):
    for apporach in ('sglang', 'hotprefix'):
        df = pd.read_csv(f"{path_prefix}{apporach}.csv")
        for index, row in df.iterrows():
            first_col_value = str(row[0]).lower()
            # Process each row as needed
            
            pattern = r'300_(longbench|loogle)_(sglang_fcfs|hotprefix)_(8b|70b)_overall_qps(\d+\.\d+)_frac(\d+)_cv0.25_gamma'
            match = re.search(pattern, first_col_value)
            if not match:
                print(f"[line {index}] 无法从字符串中提取信息: {first_col_value}")
                continue
            dataset_name, _, model_size, qps, frac = match.groups()
            
            model_size = model_size.upper()
            qps = float(qps)
            
            res = {}
                    
            ttft, tpot = float(row[6]), float(row[7])
            hit_items = int(row[10])
            hit_rate = float(row[13]) if row[13][-1] != '%' else float(row[13][:-1]) / 100
            total_time, tpt = float(row[16]), float(row[17])
            
            res["ttft"] = ttft
            res["tpot"] = tpot
            res["hit_items"] = hit_items
            res["hit_rate"] = hit_rate
            res["total_time"] = total_time
            res["tpt"] = tpt
            
            print(f"Loaded line {index}: {first_col_value} {frac} ({res})")
            
            if dataset_name not in dataset:
                dataset[dataset_name] = {}
            if model_size not in dataset[dataset_name]:
                dataset[dataset_name][model_size] = {}
            if frac not in dataset[dataset_name][model_size]:
                dataset[dataset_name][model_size][frac] = {}
            if apporach not in dataset[dataset_name][model_size][frac]:
                dataset[dataset_name][model_size][frac][apporach] = {}
            
            entry = dataset[dataset_name][model_size][frac][apporach]
            
            if qps in entry:
                print(f"[line {index}] Duplicate qps entry: {qps}")
                continue
            entry[qps] = res
    
    return dataset

def load_data_tracea(path='./data/tracea.csv'):
    data_file = path
    df = pd.read_csv(data_file, header=None)

    dataset = {
        "tracea": {
            "8B": {},
            "70B": {},
        },
    }
    
    for index, row in df.iterrows():
        first_col_value = str(row[0])
        
        # ignore cases
        first_col_value = first_col_value.lower()
        
        if not first_col_value.startswith('tracea'):
            continue
        # Process the row for tracea dataset
        ele = first_col_value.split('_')
        apporach, model_size, qps = ele[3], ele[4].upper(), float(ele[5][3:])
        
        if apporach == 'pensieve':
            apporach = 'pensi'
        elif apporach == 'mwf':
            apporach = 'minexp'
        
        
        if model_size not in dataset["tracea"]:
            dataset["tracea"][model_size] = {}
        if apporach not in dataset["tracea"][model_size]:
            dataset["tracea"][model_size][apporach] = {}
        if qps not in dataset["tracea"][model_size][apporach]:
            dataset["tracea"][model_size][apporach][qps] = {}

        entry = dataset["tracea"][model_size][apporach]

        ttft, tpot = float(row[3]), float(row[10])
        hit_items = int(row[13])
        hit_rate = float(row[7]) if row[7][-1] != '%' else float(row[7][:-1]) / 100
        total_time, tpt = float(row[16]), float(row[17])

        res = {}
        
        res["ttft"] = ttft
        res["tpot"] = tpot
        res["hit_items"] = hit_items
        res["hit_rate"] = hit_rate
        res["total_time"] = total_time
        res["tpt"] = tpt
        
        print(f"[line {index}] model_size: {model_size}, approach: {apporach}, qps: {qps}. Processed entry: {res}")
        entry[qps] = res
    
    return dataset

def load_data_mixed(path='./data/data-mixed.csv'):
    data_file = path
    df = pd.read_csv(data_file, header=None)

    dataset = {
        "mixed": {
            "8B": {},
            "70B": {},
        },
    }
    
    for index, row in df.iterrows():
        first_col_value = str(row[0])
        
        # ignore cases
        first_col_value = first_col_value.lower()
        
        if not first_col_value.startswith('bfcl_tracea'):
            continue
        # Process the row for tracea dataset
        ele = first_col_value.split('_')
        apporach, model_size = ele[4], ele[5].upper()
        
        if apporach == 'pensieve':
            apporach = 'pensi'
        elif apporach == 'mwf':
            apporach = 'minexp'
        
        
        if model_size not in dataset["mixed"]:
            dataset["mixed"][model_size] = {}
        if apporach not in dataset["mixed"][model_size]:
            dataset["mixed"][model_size][apporach] = {}

        entry = dataset["mixed"][model_size]

        ttft, tpot = float(row[3]), float(row[10])
        hit_items = int(row[13])
        hit_rate = float(row[7]) if row[7][-1] != '%' else float(row[7][:-1]) / 100
        total_time, tpt = float(row[16]), float(row[17])

        res = {}
        
        res["ttft"] = ttft
        res["tpot"] = tpot
        res["hit_items"] = hit_items
        res["hit_rate"] = hit_rate
        res["total_time"] = total_time
        res["tpt"] = tpt
        
        print(f"[line {index}] model_size: {model_size}, approach: {apporach}. Processed entry: {res}")
        entry[apporach] = res
    
    return dataset

def load_data_sense(path = './data/实验结果.xlsx'):
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

def load_data_bfcl(path = './data/实验结果.xlsx'):
    data_file = path
    df = pd.read_excel(data_file, sheet_name='BFCLv4', header=None)
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
    