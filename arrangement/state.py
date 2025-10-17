# arrangement/state.py
import streamlit as st

def init_state():
    """初始化所有必要的会话状态变量"""
    # 测试相关状态
    test_states = {
        'k_value': 5,
        'test_results': None,
        'labels': None,
        'train_indices': None,
        'test_indices': None
    }
    
    # 数据与处理相关状态
    data_states = {
        'raw_data': None,
        'processed_data': None,
        'peaks': None,
        'train_test_split_ratio': 0.8,
        'arrangement_results': [],
        'selected_arrangement': None,
        'arrangement_details': {},
        'algorithm_permutations': [],
        'current_algorithms': {
            'baseline': '无',
            'baseline_params': {},
            'scaling': '无',
            'scaling_params': {},
            'filtering': '无',
            'filtering_params': {},
            'squashing': '无',
            'squashing_params': {}
        },
        'filtered_perms': [],
        'selected_perm_idx': 0,
        'show_arrangements': False,
        'process_method': None
    }
    
    # 合并并初始化状态
    all_states = {** test_states, **data_states}
    for key, value in all_states.items():
        if key not in st.session_state:
            st.session_state[key] = value
