"""
模拟传感器数据，产生 三相(Phase A、B、C)电流+功率瓦数(watts) 共4条曲线
"""
from functools import total_ordering
import random
from unittest import result
import matplotlib.pyplot as plt
from sqlalchemy import TIME

SAMPLE_RATE = 100  # 采样率 per second

# 产生a和b之间随机小数,保留n位


def random_float(a, b, n=2):
    return round(random.uniform(a, b), n)


def tansform_to_plt(points):
    return list(zip(*points))


def generate_stage1(max_val=random_float(4, 6), final_val=random_float(1, 1.5)):
    start_delay = random_float(0, 0.2)
    print("start_delay: ", start_delay)
    rise_to_max_duration = random_float(0, 0.2)
    print("rise_to_max_duration: ", rise_to_max_duration)
    down_to_normal_duration = random_float(0, 0.1)
    print("down_to_normal_duration: ", down_to_normal_duration)
    duration = start_delay + rise_to_max_duration + down_to_normal_duration

    key_points = [(0, 0), (start_delay, 0),
                  (start_delay+rise_to_max_duration, max_val),
                  (duration, final_val)]
    print(key_points)
    global result
    result += key_points
    #draw_line(*tansform_to_plt(key_points), "Stage 1", "Current(A)")
    return duration, final_val


def generate_stage2(start_timestamp, stable_val, final_val=random_float(0.5, 1)):
    stable_duration = random_float(4, 6)
    decrease_duration = random_float(0, 0.1)
    duration = stable_duration + decrease_duration
    key_points = [(start_timestamp, stable_val),
                  (start_timestamp+stable_duration, stable_val),
                  (start_timestamp+duration, final_val)]
    print(key_points)
    global result
    result += key_points
    #draw_line(*tansform_to_plt(key_points), "Stage 2", "Current(A)")
    return duration, final_val


def generate_stage3(start_timestamp, start_val):
    stable_duration = random_float(1.5, 2.5)
    decrease_duration = random_float(0, 0.1)
    duration = stable_duration + decrease_duration
    key_points = [(start_timestamp, start_val),
                  (start_timestamp+stable_duration, start_val),
                  (start_timestamp+duration, 0)]
    print(key_points)
    global result
    result += key_points
    #draw_line(*tansform_to_plt(key_points), "Stage 3", "Current(A)")
    return duration


def generate_normal():
    global result
    result = []
    time_elipsed = 0
    duration, final_val = generate_stage1()
    time_elipsed += duration
    duration, final_val = generate_stage2(time_elipsed, final_val)
    time_elipsed += duration
    duration = generate_stage3(time_elipsed, final_val)
    time_elipsed += duration

    print("result: ", result)
    draw_line(*tansform_to_plt(result), "", "Current(A)")


def draw_line(x, y, title="", y_label=""):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("Time(s)")
    plt.ylabel(y_label)
    plt.show()


if __name__ == "__main__":
    generate_normal()
