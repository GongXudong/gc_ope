import re


def extract_line_data(line):
    """
    从单行文本中提取[]内的3个浮点数和with score后面的浮点数
    """
    # 提取[]中的三个浮点数
    array_pattern = r'\[([^\]]+)\]'
    array_matches = re.findall(array_pattern, line)

    # 第一个[]包含三个浮点数（点坐标）
    point_data = []
    if len(array_matches) > 0:
        # 使用正则表达式匹配所有浮点数（包括科学计数法）
        numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', array_matches[0])
        point_data = [float(num) for num in numbers[:3]]  # 只取前三个

    # 提取with score后面的浮点数
    score_pattern = r'with score\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
    score_match = re.search(score_pattern, line)
    score_value = float(score_match.group(1)) if score_match else None

    return point_data, score_value


def process_file(
    file_path,
    sample_goal_log_begin_strs: list[str] = ["sample from omega random", "find min", "find max"],
):
    """
    处理整个文件，提取所有行的数据
    """
    all_timestamps = []
    all_points = []
    all_scores = []
    current_timestamps = 0

    with open(file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            try:
                # 提取total_timesteps的值
                if (tmp_tt := line.find("total_timesteps")) != -1:
                    timestamps_matches = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line[tmp_tt:])
                    current_timestamps = timestamps_matches[0]

                # 提取desired_goal和采样到该desired_goal的概率
                for begin_str in sample_goal_log_begin_strs:
                    tmp_idx = line.find(begin_str)
                    if tmp_idx != -1:
                        line = line[tmp_idx:]

                        point_data, score_value = extract_line_data(line)

                        if len(point_data) == 3 and score_value is not None:
                            all_timestamps.append(current_timestamps)
                            all_points.append(point_data)
                            all_scores.append(score_value)
                        else:
                            print(f"警告: 第{line_num}行数据格式不正确或数据不完整, points: {point_data}, score: {score_value}")

            except Exception as e:
                print(f"错误: 处理第{line_num}行时发生错误: {e}")
                print(f"行内容: {line}")

    return all_timestamps, all_points, all_scores
