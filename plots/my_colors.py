color_plattes = []

color_plattes.append({
    'lru': '#1f77b4',
    'bs': '#ff7f0e',       # 橙色
    'pensi': '#2ca02c',    # 绿色
    'minexp': '#d62728',   # 红色
    'shed': '#9467bd',   # 紫色
})

color_plattes.append({
    'lru': '#cdb4db',
    'sglang': "#fae8b2",
    'bs': '#ffc8dd',
    'pensi': '#ffafcc',
    'hotprefix': "#fcd2ae",
    'minexp': "#bffebd",
    'shed': '#a2d2ff',
})

color_plattes.append({
    'lru': "#604b6b",
    'sglang': "#c1aa64",
    'bs': "#cc87a1",
    'pensi': "#be6284",
    'hotprefix': "#e7aa74",
    'minexp': "#77c474",
    'shed': "#6094c5",
})

color_plattes.append({
    'lru': "#d3c5e1",
    'sglang': "#f9ecc9",
    'bs': "#ffd8e7",
    'pensi': "#ffc3d9",
    'hotprefix': "#f9dfca",
    'minexp': "#d7fcd6",
    'shed': "#c2def9",
})

label_map = {
    'lru': 'vLLM-LRU',
    'sglang': 'SGLang',
    'bs': 'Max-Score',
    'pensi': 'Pensieve + MSA',
    'hotprefix': 'Min-Hotness',
    'minexp': 'AsymCache w/o Scheduler',
    'shed': 'AsymCache',
}

def get_color_bool(approach: str, id: int, replace_minexp: bool) -> str:
    if replace_minexp:
        # 非 LongBench 0.04 的情况，minexp就是shed
        if approach == 'minexp':
            approach = 'shed'

    return color_plattes[id % len(color_plattes)].get(approach, '#000000')

def get_color(approach: str, id: int,
              dataset=None,
              qps=None) -> str:
    if dataset != 'longbench' or qps != 0.04:
        return get_color_bool(approach, id, replace_minexp=True)

    return get_color_bool(approach, id, replace_minexp=False)

agent_color_palette = {
    'lru': '#cdb4db',
    'mwf': '#a2d2ff',
    'continuum_mwf': "#F8BF92",
    'continuum': "#fae3a8",
}

agent_text_color_palette = {
    'lru': '#cdb4db',
    'mwf': '#6094c5',
    'continuum_mwf': "#F09347",
    'continuum': "#f1bb34",
}

agent_label_map = {
    'lru': 'vLLM-LRU',
    'mwf': 'AsymCache',
    'continuum': 'Continuum',
    'continuum_mwf': 'Continuum + AsymCache',
}
