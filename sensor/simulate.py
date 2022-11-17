"""
模拟传感器数据，产生 三相(Phase A、B、C)电流+功率瓦数(watts) 共4条曲线
思路：
1. 通过预定义的区间产生随机数，确定每个阶段的持续时间和过程、最终值
2. 通过1的结果，确定每个阶段的关键点
3. 通过关键点插值，产生符合采样率的平滑三项电流曲线
4. 通过电流曲线计算得出功率瓦数曲线
"""

import numpy as np
try:
    from .config import SUPPORTED_SAMPLE_TYPES, SAMPLE_RATE
    from .utils import *
except:
    from config import SUPPORTED_SAMPLE_TYPES, SAMPLE_RATE
    from utils import *

RUNNING_CURRENT_LIMIT = 2  # 单项最大电流保护阈值


def generate_stage1(durations, values):
    """产生第一阶段的关键点"""

    duration = durations["start_delay"] + \
        durations["rise_to_max_duration"] + \
        durations["down_to_normal_duration"]  # 第一阶段持续时间
    key_points = [(0, 0), (durations["start_delay"], 0),
                  (durations["start_delay"] +
                   durations["rise_to_max_duration"], values["stage1_max_val"]),
                  (duration, values["stage1_final_val"])]  # 第一阶段关键点
    # print(key_points)
    global result
    result += key_points  # 将关键点加入结果集
    #draw_line(*tansform_to_plt(key_points), "Stage 1", "Current(A)")
    # draw_line(*interpolate(*tansform_to_plt(key_points)),
    #          "Stage 1 interpolated", "Current(A)")
    return duration


def generate_stage2(durations, values, start_timestamp, is_phase_down=False):
    """产生第二阶段的关键点"""

    duration = durations["stage2_stable_duration"] + \
        durations["stage2_decrease_duration"]  # 第二阶段持续时间
    key_points = [(start_timestamp, values["stage1_final_val"]),
                  (start_timestamp +
                   durations["stage2_stable_duration"], values["stage1_final_val"]),
                  (start_timestamp+duration, values["stage2_final_val"] if not is_phase_down else 0)]  # 第二阶段关键点
    # print(key_points)
    global result
    result += key_points[1:]  # 将关键点加入结果集（去掉第一个是避免重复）
    #draw_line(*tansform_to_plt(key_points), "Stage 2", "Current(A)")
    return duration


def generate_stage3(durations, values,  start_timestamp, is_phase_down=False, add_end_zeros=True):
    """产生第三阶段的关键点"""

    # 适配论文中某项掉电的情况
    stage2_final_val = values["stage2_final_val"] if not is_phase_down else 0

    duration = durations["stage3_stable_duration"] + \
        durations["stage3_decrease_duration"] +\
        durations["stage3_end_zeros_duration"]  # 第三阶段持续时间
    key_points = [(start_timestamp, stage2_final_val),
                  (start_timestamp +
                   durations["stage3_stable_duration"], stage2_final_val),
                  (start_timestamp + durations["stage3_stable_duration"] +
                   durations["stage3_decrease_duration"], 0),
                  (start_timestamp+duration, 0)]  # 第三阶段关键点
    # print(key_points)
    global result
    result += key_points[1:]  # 将关键点加入结果集（去掉第一个是避免重复）
    #draw_line(*tansform_to_plt(key_points), "Stage 3", "Current(A)")
    return duration


def generate_single_current(durations, values, phase_name, type="normal"):
    """产生电流曲线"""
    global result
    result = []
    segmentations = []  # 分阶段
    time_elipsed = 0  # 时间戳
    is_phase_down = (
        phase_name == values["phase_down"]) if type != "F4" else False  # 适配论文中某项掉电的情况
    duration = generate_stage1(durations, values)  # 产生第一阶段
    time_elipsed += duration  # 时间戳增加
    segmentations.append(time_elipsed)  # 记录分段点

    duration = generate_stage2(
        durations, values, time_elipsed, is_phase_down)  # 产生第二阶段
    time_elipsed += duration  # 时间戳增加
    segmentations.append(time_elipsed)  # 记录分段点

    duration = generate_stage3(
        durations, values, time_elipsed, is_phase_down)  # 产生第三阶段
    time_elipsed += duration  # 时间戳增加

    values["segmentations"] = segmentations

    print(f"phase {phase_name} keypoints: ", result)
    print(f"phase {phase_name} time_elipsed: ", time_elipsed)
    result = interpolate(*tansform_to_plt(result))  # 插值到满足采样率
    #draw_line(*result, "", "Current(A)")

    x, y = map(lambda x: list(x), result)  # 转换为list

    fault_features = {
        "H2": {
            "start": find_nearest(x, segmentations[0]),
            "end": find_nearest(x, segmentations[1]),
            "noise_level": 0.3,
            "percentage": 0.2
        },
        "H4": {
            "start": find_nearest(x, segmentations[0]),
            "end": find_nearest(x, segmentations[1]),
            "noise_level": (0.3, 0.6),
            "percentage": 0.05},
        "H5": {
            "start": find_nearest(x, segmentations[1]),
            "end": len(x),
            "noise_level": 0.3,
            "percentage": 0.2},
    }  # 几种故障的波动特征，后面需要对其加入噪声来模拟

    if type in fault_features:  # 如果是需要加入噪声的情况
        fault_feature = fault_features[type]  # 获取故障特征
        start, end = fault_feature["start"], fault_feature["end"]  # 获取区间
        x_seg, y_seg = add_noise(
            x[start:end], y[start:end],
            noise_level=fault_feature["noise_level"],
            percentage=fault_feature["percentage"])  # 加入噪声
        result = x[:start] + x_seg + \
            x[end:], y[:start] + y_seg + y[end:]  # 重新组合

    if type == "F1":  # 适配论文中某项掉电的情况
        if phase_name == values["phase_down"]:
            return x, np.zeros(len(y))  # 直接返回0
        result = add_noise(*result, noise_level=0.01,
                           percentage=0.1)  # 加入较少的噪声
    else:
        result = add_noise(*result)  # 加入默认大小噪声
    return result


def generate_durations_and_values(type="normal"):
    """
    产生各阶段的持续时间和最终、过程值，将通过这些值确定关键点
    支持各种故障模拟
    以下内容基于论文中的经验总结
    """
    if type not in SUPPORTED_SAMPLE_TYPES:
        raise Exception("type error")  # 异常输入

    durations = {
        #######  Stage 1 starts  ########
        # 开始延迟，开始时电流为0，需要等待一段时间才开始上升
        "start_delay": random_float(0.1, 0.25),
        # 电机电流上升到最大值的时间（启动时的瞬间）
        "rise_to_max_duration": random_float(0.05, 0.1),
        # 电机从瞬间大电流下降到正常的时间
        "down_to_normal_duration": random_float(0.05, 0.1),

        #######  Stage 2 starts  ########
        # 很长一段平台期，电流几乎保持不变
        "stage2_stable_duration": random_float(3, 5),
        # 结束时电流下降过程的用时
        "stage2_decrease_duration": random_float(0, 0.1),

        #######  Stage 3 starts  ########
        # 也有一段时间电流几乎保持不变，但比stage 2短一些
        "stage3_stable_duration": random_float(1.5, 2.5),
        # 结束时电流下降过程的用时
        "stage3_decrease_duration": random_float(0, 0.1),
        # 结尾延迟
        "stage3_end_zeros_duration": random_float(0.2, 0.3),
    }
    values = {
        "stage1_max_val": random_float(4.5, 5.5),  # 电机启动电流峰值
        # stage 1最终值，也就是stage 2长时间稳定平台期的电流值，请结合图看
        "stage1_final_val": random_float(1, RUNNING_CURRENT_LIMIT-0.5),
        # stage 2最终值，stage 3平台期的电流值，请结合图看
        "stage2_final_val": random_float(0.5, 0.9),
        # 分段点
        "segmentations": [],
        # 适配论文中某项掉电的情况
        "phase_down": random.choice(["A", "B", "C"])
    }
    #print("durations: ", durations)
    #print("values: ", values)
    if type == "normal":  # 正常状态
        pass
    elif type == "H1":  # H1故障（论文中的hidden fault #1）
        """Overlong duration in Stage 2"""
        durations["stage2_stable_duration"] = random_float(12, 15)
    elif type == "H2":
        """Abnormal fluctuations of the current series"""
        pass  # 模拟此故障无需修改关键点
    elif type == "H3":
        """"Current exceeds limit in Stage 2"""
        values["stage1_final_val"] = random_float(
            RUNNING_CURRENT_LIMIT, RUNNING_CURRENT_LIMIT+1)
    elif type == "H4":
        """
        Abrupt change of current in Stage 2
        需要在曲线生成后修改
        """
        pass
    elif type == "H5":
        """Overlong duration in Stage 3"""
        durations["stage3_stable_duration"] = random_float(3.5, 5)
    elif type == "H6":
        """Double or half current value in Stage 3"""
        values["stage2_final_val"] = random_float(
            0.5, 0.9)*random.choice([0.5, 2])
    elif type == "F1":
        """Current and power staying at a low level"""
        temp = random_float(0.3, 0.5)
        values["stage1_max_val"] = temp
        values["stage1_final_val"] = temp
        values["stage2_final_val"] = temp
        durations["stage2_stable_duration"] = random_float(12, 15)
        durations["stage3_stable_duration"] = 0
    elif type == "F2":
        """Current and power are always zero"""
        values["stage1_max_val"] = 0
        values["stage1_final_val"] = 0
        values["stage2_final_val"] = 0
        durations["start_delay"] = random_float(0.8, 1)
        durations["stage2_stable_duration"] = 0
        durations["stage2_decrease_duration"] = 0
        durations["stage3_stable_duration"] = 0
        durations["stage3_decrease_duration"] = 0
    elif type == "F3":
        """Current and power drop to zero in Stage 2"""
        values["stage2_final_val"] = 0
        durations["stage2_stable_duration"] = random_float(0.5, 0.8)
        durations["stage3_stable_duration"] = random_float(0.3, 0.5)
        durations["stage3_decrease_duration"] = 0
    elif type == "F4":
        """Current and power rise in Stage 2"""
        durations["stage3_stable_duration"] = random_float(10, 15)
        values["stage2_final_val"] = values["stage1_final_val"] + \
            random_float(0.3, 0.6)
    elif type == "F5":
        """Current and power drop to zero in Stage 3"""
        values["stage2_final_val"] = 0

    return durations, values


def generate_current_series(type="normal", show_plt=False):
    """产生三相电流曲线"""
    durations, values = generate_durations_and_values(type)  # 生成模拟需要用到的参数
    current_results = {}
    time_elipsed = 0
    for phase in ["A", "B", "C"]:
        result = generate_single_current(
            durations, values, phase, type)  # 生成单相电流曲线
        current_results[phase] = result
        time_elipsed = result[0][-1]  # 计算总用时
        if show_plt:  # debug usage
            plt.plot(*result, label=f"Phase {phase}")
    print("durations: ", durations)
    print("values: ", values)
    if show_plt:  # debug usage
        plt.title(f"{type.capitalize()} Current Series")
        plt.xlabel("Time(s)")
        plt.ylabel("Current(A)")
        plt.show()

    segmentations = values["segmentations"]
    if type == "F1":
        segmentations = time_elipsed - \
            durations["stage3_end_zeros_duration"], None
    if type == "F2":
        segmentations = None, None
    if type in "F4":
        segmentations[1] = time_elipsed - \
            durations["stage3_end_zeros_duration"]

    return current_results, segmentations


def generate_sample(type="normal", show_plt=False):
    """产生单个样本(4条曲线)"""
    current_series, segmentations = generate_current_series(
        type, False)  # 产生三相电流曲线
    power_series = generate_power_series(
        current_series, show_plt=False)  # 产生瓦数曲线
    result = current_series
    result["power"] = power_series  # 将瓦数曲线加入结果
    if show_plt:  # debug usage
        show_sample(result, type)
    return result, segmentations


if __name__ == "__main__":
    for sample_type in SUPPORTED_SAMPLE_TYPES:
        #current_series = generate_current_series(sample_type, show_plt=True)
        #generate_power_series(current_series, show_plt=True)
        generate_sample(sample_type, show_plt=True)
