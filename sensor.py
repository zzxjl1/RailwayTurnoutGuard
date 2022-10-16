"""
模拟传感器数据，产生 三相(Phase A、B、C)电流+功率瓦数(watts) 共4条曲线
思路：
1. 通过预定义的区间产生随机数，确定每个阶段的持续时间和过程、最终值
2. 通过1的结果，确定每个阶段的关键点
3. 通过关键点插值，产生符合采样率的平滑三项电流曲线
4. 通过电流曲线计算得出功率瓦数曲线
"""
import random
import matplotlib.pyplot as plt
import scipy.interpolate
import numpy as np


SAMPLE_RATE = 60  # 采样率 per second
RUNNING_CURRENT_LIMIT = 2  # 单项最大电流保护阈值
SUPPORTED_SAMPLE_TYPES = ["normal", "H1", "H2", "H3",
                          "H4", "H5", "H6", "F1", "F2", "F3", "F4", "F5"]


def random_float(a, b, n=2):
    """产生a和b之间随机小数,保留n位"""
    return round(random.uniform(a, b), n)


def tansform_to_plt(points):
    """将坐标点(x,y)转换为plt可用的格式(x),(y)'
    eg: [(9,0),(1,5),(2,4)] -> ([9,1,2],[0,5,4])
    """
    return list(zip(*points))


def generate_stage1(durations, values):
    """产生第一阶段的关键点"""
    duration = durations["start_delay"] + \
        durations["rise_to_max_duration"] + \
        durations["down_to_normal_duration"]

    key_points = [(0, 0), (durations["start_delay"], 0),
                  (durations["start_delay"] +
                   durations["rise_to_max_duration"], values["stage1_max_val"]),
                  (duration, values["stage1_final_val"])]
    # print(key_points)
    global result
    result += key_points
    #draw_line(*tansform_to_plt(key_points), "Stage 1", "Current(A)")
    # draw_line(*interpolate(*tansform_to_plt(key_points)),
    #          "Stage 1 interpolated", "Current(A)")
    return duration


def generate_stage2(durations, values, start_timestamp, is_phase_down=False):
    """产生第二阶段的关键点"""

    duration = durations["stage2_stable_duration"] + \
        durations["stage2_decrease_duration"]
    key_points = [(start_timestamp, values["stage1_final_val"]),
                  (start_timestamp +
                   durations["stage2_stable_duration"], values["stage1_final_val"]),
                  (start_timestamp+duration, values["stage2_final_val"] if not is_phase_down else 0)]
    # print(key_points)
    global result
    result += key_points[1:]
    #draw_line(*tansform_to_plt(key_points), "Stage 2", "Current(A)")
    return duration


def generate_stage3(durations, values,  start_timestamp, is_phase_down=False):
    """产生第三阶段的关键点"""
    stage2_final_val = values["stage2_final_val"] if not is_phase_down else 0
    duration = durations["stage3_stable_duration"] + \
        durations["stage3_decrease_duration"]
    key_points = [(start_timestamp, stage2_final_val),
                  (start_timestamp +
                   durations["stage3_stable_duration"], stage2_final_val),
                  (start_timestamp+duration, 0)]
    # print(key_points)
    global result
    result += key_points[1:]
    #draw_line(*tansform_to_plt(key_points), "Stage 3", "Current(A)")
    return duration


def generate_normal_current(durations, values, phase_name, type="normal"):
    """产生电流曲线（正常状态）"""
    global result
    result = []
    segmentations = []
    time_elipsed = 0
    is_phase_down = (
        phase_name == values["phase_down"]) if type != "F4" else False
    duration = generate_stage1(durations, values)
    time_elipsed += duration
    segmentations.append(time_elipsed)

    duration = generate_stage2(
        durations, values, time_elipsed, is_phase_down)
    time_elipsed += duration
    segmentations.append(time_elipsed)

    duration = generate_stage3(durations, values, time_elipsed, is_phase_down)
    time_elipsed += duration

    values["segmentations"][phase_name] = segmentations

    print(f"phase {phase_name} keypoints: ", result)
    print(f"phase {phase_name} time_elipsed: ", time_elipsed)
    result = interpolate(*tansform_to_plt(result))
    #draw_line(*result, "", "Current(A)")

    x, y = map(lambda x: list(x), result)

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    fault_features = {
        "H2": {
            "start": find_nearest(x, segmentations[0])+1,
            "end": find_nearest(x, segmentations[1]),
            "noise_level": 0.5,
            "percentage": 0.2
        },
        "H4": {
            "start": find_nearest(x, segmentations[0])+1,
            "end": find_nearest(x, segmentations[1]),
            "noise_level": 0.7,
            "percentage": 0.01},
        "H5": {
            "start": find_nearest(x, segmentations[1])+1,
            "end": len(x),
            "noise_level": 0.5,
            "percentage": 0.2},
    }

    if type in fault_features:
        fault_feature = fault_features[type]
        start, end = fault_feature["start"], fault_feature["end"]
        x_seg, y_seg = add_noise(
            x[start:end], y[start:end],
            noise_level=fault_feature["noise_level"],
            percentage=fault_feature["percentage"])
        result = x[:start] + x_seg + x[end:], y[:start] + y_seg + y[end:]

    if type == "F1":
        if phase_name == values["phase_down"]:
            return x, np.zeros(len(y))
        result = add_noise(*result, noise_level=0.01, percentage=0.1)
    else:
        result = add_noise(*result)
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
        "rise_to_max_duration": random_float(0.05, 0.2),
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
        "stage3_decrease_duration": random_float(0, 0.1)
    }
    values = {
        "stage1_max_val": random_float(4.5, 5.5),  # 电机启动电流峰值
        # stage 1最终值，也就是stage 2长时间稳定平台期的电流值，请结合图看
        "stage1_final_val": random_float(1, RUNNING_CURRENT_LIMIT-0.5),
        # stage 2最终值，stage 3平台期的电流值，请结合图看
        "stage2_final_val": random_float(0.5, 0.9),
        "segmentations": {"A": [], "B": [], "C": []},
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
        durations["stage3_stable_duration"] = random_float(18, 22)
        values["stage2_final_val"] = values["stage1_final_val"] + \
            random_float(0.3, 0.6)
    elif type == "F5":
        """Current and power drop to zero in Stage 3"""
        values["stage2_final_val"] = 0

    return durations, values


def generate_current_series(type="normal", show_plt=False):
    """产生三相电流曲线"""
    durations, values = generate_durations_and_values(type)
    current_results = {}
    for phase in ["A", "B", "C"]:
        result = generate_normal_current(durations, values, phase, type)
        current_results[phase] = result
        if show_plt:
            plt.plot(*result, label=f"Phase {phase}")
    print("durations: ", durations)
    print("values: ", values)
    if show_plt:
        plt.title(f"{type.capitalize()} Current Series")
        plt.xlabel("Time(s)")
        plt.ylabel("Current(A)")
        plt.show()
    return current_results


def generate_power_series(current_series, power_factor=0.8, show_plt=False):
    """
    产生瓦数曲线，采用直接计算的方式，需传入三项电流曲线
    P (kW) = I (Amps) × V (Volts) × PF(功率因数) × 1.732
    """
    x, _ = current_series['A']
    length = len(x)
    result = np.zeros(length)
    for phase in ["A", "B", "C"]:
        for i in range(length):
            _, current = current_series[phase]
            result[i] += current[i]*220*power_factor*1.732
    if show_plt:
        plt.plot(x, result)
        plt.title("Power Series")
        plt.xlabel("Time(s)")
        plt.ylabel("Power(W)")
        plt.show()
    return x, result


def generate_sample(type="normal", show_plt=False):
    current_series = generate_current_series(type, False)
    power_series = generate_power_series(current_series, show_plt=False)
    result = current_series
    result["power"] = power_series
    if show_plt:
        fig = plt.figure(dpi=150, figsize=(9, 2))
        ax1 = fig.subplots()
        ax2 = ax1.twinx()
        for phase in ["A", "B", "C"]:
            ax1.plot(*result[phase], label=f"Phase {phase}")
        ax2.plot(*result["power"], 'b--', label="Power")
        plt.title(f"Sample {type.capitalize()}")
        ax1.set_xlabel("Time(s)")
        ax1.set_ylabel("Current(A)")
        ax2.set_ylabel('Power(W)')

        ax1.set_ylim(bottom=0, top=5)
        ax2.set_ylim(bottom=0, top=5000)

        plt.xlim(0, None)
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines + lines2, labels + labels2, loc='best')
        plt.show()
    return result


def add_noise(x, y, noise_level=0.05, percentage=0.3):
    """加入抖动噪声"""
    if isinstance(noise_level, float):
        noice_range = (-noise_level/2, noise_level/2)
    elif isinstance(noise_level, tuple):
        noice_range = noise_level
    n = [random_float(*noice_range) for _ in range(len(x))]
    for i in range(len(x)):
        if random.random() > percentage:  # 按概率加入噪声
            continue
        if y[i] == 0:  # 值为0的点不加噪声
            continue
        y[i] += n[i]
    return correct_curve(x, y)


def correct_curve(x, y):
    """修正曲线,小于0的值设为0"""
    return x, [0 if i < 0 else i for i in y]


def interpolate(x, y):
    """根据关键点插值到固定采样率"""
    interper = scipy.interpolate.interp1d(x, y, kind='slinear')
    time_elipsed = max(x)-min(x)
    x = np.linspace(min(x), max(x), round(time_elipsed*SAMPLE_RATE/2))
    y = interper(x)

    x_new = np.linspace(min(x), max(x), round(time_elipsed*SAMPLE_RATE))
    interper = scipy.interpolate.interp1d(x, y, kind='cubic')  # 三次就是cubic
    y_new = interper(x_new)
    #y_new = scipy.interpolate.make_interp_spline(x, y)(x_new)

    return correct_curve(x_new, y_new)


def draw_line(x, y, title="", y_label=""):
    """绘制曲线（debug usage）"""
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("Time(s)")
    plt.ylabel(y_label)
    plt.show()


if __name__ == "__main__":
    for sample_type in SUPPORTED_SAMPLE_TYPES:
        #current_series = generate_current_series(sample_type, show_plt=True)
        #generate_power_series(current_series, show_plt=True)
        generate_sample(sample_type, show_plt=True)
