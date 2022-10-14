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


def generate_stage2(durations, values, start_timestamp, is_phase_c=False):
    """产生第二阶段的关键点"""
    if is_phase_c:
        values["stage2_final_val"] = 0

    duration = durations["stage2_stable_duration"] + \
        durations["stage2_decrease_duration"]
    key_points = [(start_timestamp, values["stage1_final_val"]),
                  (start_timestamp +
                   durations["stage2_stable_duration"], values["stage1_final_val"]),
                  (start_timestamp+duration, values["stage2_final_val"])]
    # print(key_points)
    global result
    result += key_points[1:]
    #draw_line(*tansform_to_plt(key_points), "Stage 2", "Current(A)")
    return duration


def generate_stage3(durations, values,  start_timestamp):
    """产生第三阶段的关键点"""
    duration = durations["stage3_stable_duration"] + \
        durations["stage3_decrease_duration"]
    key_points = [(start_timestamp, values["stage2_final_val"]),
                  (start_timestamp +
                   durations["stage3_stable_duration"], values["stage2_final_val"]),
                  (start_timestamp+duration, 0)]
    # print(key_points)
    global result
    result += key_points[1:]
    #draw_line(*tansform_to_plt(key_points), "Stage 3", "Current(A)")
    return duration


def generate_normal_current(durations, values, phase_name="A"):
    """产生电流曲线（正常状态）"""
    global result
    result = []
    time_elipsed = 0
    duration = generate_stage1(durations, values)
    time_elipsed += duration
    duration = generate_stage2(
        durations, values, time_elipsed, is_phase_c=(phase_name == "C"))
    time_elipsed += duration
    duration = generate_stage3(durations, values, time_elipsed)
    time_elipsed += duration

    print(f"phase {phase_name} keypoints: ", result)
    result = interpolate(*tansform_to_plt(result))
    #draw_line(*result, "", "Current(A)")
    return result


def generate_durations_and_values(type="normal"):
    """
    产生各阶段的持续时间和最终、过程值，将通过这些值确定关键点
    以下内容基于论文中的经验总结
    """
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
        "stage1_max_val": random_float(4, 6),  # 电机启动电流峰值
        # stage 1最终值，也就是stage 2长时间稳定平台期的电流值，请结合图看
        "stage1_final_val": random_float(1, 1.5),
        # stage 2最终值，stage 3平台期的电流值，请结合图看
        "stage2_final_val": random_float(0.5, 0.9),
    }
    if type == "normal":  # 正常状态
        return durations, values
    elif type == "H1":  # H1故障（论文中的hidden fault #1）
        durations["stage2_stable_duration"] = random_float(12, 15)
        return durations, values

    raise Exception("type error")  # 异常输入


def generate_current_series(type="normal", show_plt=False):
    """产生三相电流曲线（正常状态）"""
    durations, values = generate_durations_and_values(type)
    current_results = {}
    for phase in ["A", "B", "C"]:
        result = generate_normal_current(durations, values, phase)
        current_results[phase] = result
        plt.plot(*result, label=f"Phase {phase}")
    if show_plt:
        plt.title("Normal Current Series")
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
        plt.title("Normal Power Series")
        plt.xlabel("Time(s)")
        plt.ylabel("Power(W)")
        plt.show()
    return x, result


def add_noise(x, y, noise_level=0.015):
    """加入抖动噪声"""
    return x, y+np.random.normal(0, noise_level, len(y))


def correct_curve(x, y):
    """修正曲线,小于0的值设为0"""
    return x, [0 if i < 0 else i for i in y]


def interpolate(x, y):
    """根据关键点插值到固定采样率"""
    interper = scipy.interpolate.interp1d(x, y, kind='slinear')
    time_elipsed = max(x)-min(x)
    x = np.linspace(min(x), max(x), round(time_elipsed*SAMPLE_RATE/2))
    y = interper(x)
    x, y = add_noise(x, y)

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
    current_series = generate_current_series("normal", show_plt=True)
    #generate_power_series(current_series, show_plt=True)
