# arrangement/utils/permutations.py
import itertools

def generate_permutations(algorithms):
    """生成算法排列组合"""
    algorithm_list = [
        (1, "基线校准", algorithms['baseline']),
        (2, "缩放", algorithms['scaling']),
        (3, "滤波", algorithms['filtering']),
        (4, "挤压", algorithms['squashing'])
    ]

    all_permutations = []
    all_permutations.append([])  # 无预处理

    # 1种算法
    for algo in algorithm_list:
        if algo[2] != "无":
            all_permutations.append([algo])

    # 2种算法
    for perm in itertools.permutations(algorithm_list, 2):
        if perm[0][2] != "无" and perm[1][2] != "无":
            all_permutations.append(list(perm))

    # 3种算法
    for perm in itertools.permutations(algorithm_list, 3):
        if perm[0][2] != "无" and perm[1][2] != "无" and perm[2][2] != "无":
            all_permutations.append(list(perm))

    # 4种算法
    for perm in itertools.permutations(algorithm_list, 4):
        if (perm[0][2] != "无" and perm[1][2] != "无" and
                perm[2][2] != "无" and perm[3][2] != "无"):
            all_permutations.append(list(perm))

    # 格式化结果
    formatted_perms = []
    for perm in all_permutations:
        perm_dict = {
            "name": "",
            "order": [],
            "details": perm,
            "count": len(perm),
            "first_step_type": "未知"
        }
        if not perm:
            perm_dict["name"] = "无预处理（原始光谱）"
            perm_dict["first_step_type"] = "无预处理"
        else:
            first_step_type = perm[0][1] if perm else "未知"
            perm_dict["first_step_type"] = first_step_type
            perm_details = [f"{step[0]}.{step[1]}({step[2]})" for step in perm]
            perm_dict["name"] = " → ".join(perm_details)
            perm_dict["order"] = [step[0] for step in perm]
        formatted_perms.append(perm_dict)
    return formatted_perms
