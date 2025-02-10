"""
对曲线进行分割，得到分段点
"""
from sklearn.neighbors import LocalOutlierFactor as LOF
from gru_score import get_score_by_time, time_to_index, GRUScore, model_input_parse
from gru_score import predict as gru_predict_score
from matplotlib import patches, pyplot as plt
from sensor import SAMPLE_RATE, SUPPORTED_SAMPLE_TYPES, get_sample
from scipy.signal import savgol_filter, find_peaks
import numpy as np


plt.rcParams['font.sans-serif'] = ['SimSun']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

SEGMENT_POINT_1_THRESHOLD = 30


def get_d(s, smooth=True, show_plt=False, name=""):
    """其实就是算斜率，具体请见论文"""
    x, y = s
    if smooth:
        """
        平滑滤波
        window_length：窗口长度，该值需为正奇整数
        k值：polyorder为对窗口内的数据点进行k阶多项式拟合，k的值需要小于window_length
        """
        y = savgol_filter(y, window_length=7, polyorder=3)
        y = savgol_filter(y, window_length=5, polyorder=1)
    if show_plt:  # debug usage
        plt.figure(dpi=150, figsize=(9, 2))
        plt.plot(*s, label='original values')
        plt.plot(x, y, label="curve after filtering")
        plt.legend(loc='best')
        plt.title(f"{name} input")
        plt.xlabel("Time(s)")
        plt.show()
    assert len(x) > 2  # 算法要求至少需要2个点
    result = []
    for i in range(len(x)-1):  # 计算曲线的斜率
        t = (y[i+1]-y[i])/(x[i+1]-x[i])
        result.append(t)
    assert len(result) == len(x)-1  # 斜率数组的个数比点的个数少1
    if show_plt:  # debug usage
        draw_line(x, result+[0],
                  title=f"{name} output", y_label="Result Value")
    return x, result+[0]  # 返回斜率


def remove_duplicate_points(points):
    """去除重复的分段点"""
    result = []
    for i in points:
        if i not in result:
            result.append(i)
    return result


def find_segmentation_point_1(x, y, threshold=SEGMENT_POINT_1_THRESHOLD):
    """寻找第一个分段点（between stage 1 and stage 2）"""
    peak_idx, _ = find_peaks(y, height=threshold)
    if threshold == 0:  # 递归中止条件，山高度阈值为0还找不到分段点，说明分段点不存在
        print("segmentation point 1 not found")
        return None, None
    if len(peak_idx) < 2:  # 找到的点不够，说明阈值太高，降低阈值再找
        threshold -= 1  # 降低"自适应阈值"
        print("applying adaptive threshhold: ", threshold)
        return find_segmentation_point_1(x, y, threshold)
    # print("peak_point_available: ", np.array(x)[peak_idx])
    index = peak_idx[1]  # 点的索引
    result = x[index]  # 点的x值（时间）
    # print("segmentation point 1: ", result)
    return index, result


def find_segmentation_point_2(x, y, original_series, segmentation_point_1_index, gru_score):
    """寻找第二个分段点（between stage 2 and stage 3）"""
    _, series_y = original_series
    # 切掉stage 1
    series_y = series_y[segmentation_point_1_index:]
    x, y = x[segmentation_point_1_index:], y[segmentation_point_1_index:]
    peak_idx, properties = find_peaks(y, prominence=0)  # 寻找峰值
    prominences = properties["prominences"]  # 峰值的详细参数
    assert len(peak_idx) == len(prominences)  # 峰值的个数和峰值的详细参数个数相同
    if len(peak_idx) == 0 or len(prominences) == 0:  # 没有找到峰值，说明分段点不存在
        print("segmentation point 2 not found")
        return None, None
    # print("peak_point_available: ", np.array(x)[peak_idx])
    scores = []  # 用于存储每个峰值的分数
    for i in range(len(prominences)):
        index = peak_idx[i]
        time_in_sec = x[index]  # 峰值的时间
        stage2_avg = np.mean(series_y[:index])  # stage 2的平均值
        stage3_avg = np.mean(series_y[index:])  # stage 3的平均值
        score = get_score_by_time(gru_score, time_in_sec) * prominences[i] * \
            (abs(y[index] - stage2_avg)/abs(y[index]-stage3_avg))
        scores.append(score)
        #print(time_in_sec, prominences[i], score)
    index = np.argmax(scores)  # 找到得分最高，返回第几个峰的索引
    index = peak_idx[index]  # 点的索引
    result = x[index]  # 峰值的x值（时间）
    # print("segmentation point 2: ", result)
    return index, result


def draw_line(x=None, y=None, title="", y_label="", is_dot=False):
    """绘制曲线（debug usage）"""
    assert y is not None
    if x is None:  # 如果没有x值，就用y值的索引作为x值
        x = [i for i in range(len(y))]
    plt.figure(dpi=150, figsize=(9, 2))
    plt.plot(x, y, "o" if is_dot else "b")
    plt.title(title)
    plt.xlabel("Time(s)")
    plt.ylabel(y_label)
    plt.show()


def calc_segmentation_points_single_series(series, gru_score, name="", show_plt=False):
    """计算单条曲线的分段点"""
    x, y = series
    duration = x[-1]  # 曲线的总时长

    d1_result = get_d(series, smooth=True, show_plt=False,
                      name=f"{name} d1")  # 计算一阶导数
    d2_result = get_d(d1_result, smooth=True,
                      show_plt=False, name=f"{name} d2")  # 计算二阶导数
    segmentation_point_1_index, segmentation_point_1_x = find_segmentation_point_1(
        *d2_result)  # 寻找第一个分段点
    _, segmentation_point_2_x = find_segmentation_point_2(
        *d2_result, series, segmentation_point_1_index, gru_score)  # 寻找第二个分段点
    if show_plt:  # debug usage
        fig = plt.figure(dpi=150, figsize=(8, 3))
        ax = fig.subplots()
        ax.set_xlim(0, duration)
        ax.set_yticks([])  # 不显示y轴
        ax_new = ax.twinx().twiny()
        ax_new.set_yticks([])  # 不显示y轴
        ax_new.set_xticks([])  # 不显示x轴
        ax_new.pcolormesh(gru_score[:time_to_index(duration)].reshape(
            1, -1), cmap="Reds", alpha=0.7)
        
        ax1 = ax.twinx()  # 生成第二个y轴
        ax2 = ax.twinx()  # 生成第三个y轴
        
        ax2.plot(*d2_result, label="纯数值差分方案", color="red",
                 linewidth=1, alpha=0.2)
        ax1.plot(x, y, label="原始电流序列", color="blue")
        
        # 设置左右y轴的标签
        ax1.set_ylabel("电流 (A)")
        ax2.set_ylabel("二阶差分值")

        # 将ax1移到左边
        ax1.spines['left'].set_position(('outward', 0))
        ax1.spines['right'].set_visible(False)
        ax1.yaxis.set_label_position('left')
        ax1.yaxis.set_ticks_position('left')
        
        
        # 画竖线
        if segmentation_point_1_x is not None:
            plt.axvline(x=segmentation_point_1_x, color='r',
                       linestyle='--', label="分段点")
        if segmentation_point_2_x is not None:
            plt.axvline(x=segmentation_point_2_x, color='r',
                       linestyle='--')
                       
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        heatmap_patch = patches.Rectangle(
            (0, 0), 1, 1, fc="r", alpha=0.7)
        plt.legend(lines+[heatmap_patch] + lines2, labels +
                  ["GRU置信度得分热力图"] + labels2, loc='upper right')  # 显示图例
        ax.set_xlabel("时间 (s)")
        plt.tight_layout()
        #plt.show()
        plt.savefig("./screenshots/segmentation_result.png",dpi=300)

    return segmentation_point_1_x, segmentation_point_2_x


def calc_segmentation_points(sample, show_plt=False):
    """计算整个样本（4条线）的分段点"""
    model_input = model_input_parse(sample)
    # print("model_input: ", model_input.shape)
    gru_score = gru_predict_score(model_input)
    # print("gru_score: ", gru_score)
    # print(gru_score.shape)

    result = {}
    for name, series in sample.items():  # 遍历每条曲线
        if name == "power":  # power曲线不作分段依据，因为感觉会起反作用
            continue
        result[name] = calc_segmentation_points_single_series(
            series, gru_score=gru_score, name=name, show_plt=show_plt)  # 计算分段点
    # print(result)
    # 做了一个融合，不同曲线算出的分段点可能不同，因此需要取最佳的分段点
    pt1, pt2 = [i[0] for i in result.values()], [i[1] for i in result.values()]
    # 去除None
    pt1, pt2 = [i for i in pt1 if i is not None], [
        i for i in pt2 if i is not None]
    # 去除离群点

    def remove_outlier(pt):
        if not pt:
            return pt
        pt = np.array(pt).reshape(-1, 1)
        result = LOF(n_neighbors=1).fit_predict(pt)
        # print(result)
        return [pt[i] for i in range(len(pt)) if result[i] == 1]

    pt1 = remove_outlier(pt1)
    pt2 = remove_outlier(pt2)
    #print(pt1, pt2)
    # 求平均值
    final_result = np.mean(pt1) if pt1 else None, np.mean(pt2) if pt2 else None
    # 特殊情况：如果第二个分段点小于等于第一个分段点，丢弃
    if final_result[0] and final_result[1] and final_result[1] <= final_result[0]:
        final_result = final_result[0], None
    print("segmentation final result: ", final_result)
    return final_result


if __name__ == "__main__":
    # sample, segmentations = generate_sample()
    # calc_segmentation_points(sample)

    
    # import matplotlib.colors as mcolors
    # from sensor.utils import find_nearest
    # from matplotlib.lines import Line2D

    # plt.figure(figsize=(15, 8), dpi=150)
    # ax = plt.subplot(projection='3d')
    # # 12种颜色
    # colors = ["teal", "purple", "royalblue", "gold", "darkslategrey", "darkviolet",
    #           "purple", "olivedrab", "dodgerblue", "slategray", "deepskyblue", "seagreen"]
    # for type_index, type in enumerate(SUPPORTED_SAMPLE_TYPES):
    #     sample, segmentations = get_sample(type)
    #     pt1, pt2 = segmentations
    #     A, B, C = sample["A"], sample["B"], sample["C"]
    #     color = colors[type_index]
    #     for x, y in [A]:
    #         ax.plot(x, y, zs=type_index, zdir='y', c=color)
    #         # 标出分段点
    #         for pt in [pt1, pt2]:
    #             if pt is None:
    #                 continue
    #             # 获取分段点的索引
    #             pt_index = find_nearest(x, pt)
    #             ax.scatter(pt, type_index, y[pt_index], c="r", marker="o")
    # ax.set_xlabel("时间（s）")
    # ax.set_ylabel("样本标签")
    # ax.set_yticklabels(SUPPORTED_SAMPLE_TYPES)
    # ax.set_yticks(range(len(SUPPORTED_SAMPLE_TYPES)))
    # ax.set_zlabel("电流（A）")
    # # Z轴范围
    # ax.set_zlim(0, 6)
    # ax.set_ylim(0, len(SUPPORTED_SAMPLE_TYPES)-1)
    # ax.set_xlim(0, 20)
    # ax.set_xticks(range(0, 21, 2))
    # plt.title("Segmentation Points of All Fault Types")
    # plt.tight_layout()
    # #plt.savefig("./screenshots/Segmentation Points of All Fault Types.png")
    # plt.show()
    

    # for type in SUPPORTED_SAMPLE_TYPES:
    #     sample, segmentations = get_sample(type)
    #     gru_score = gru_predict_score(model_input_parse(sample))
    #     print(sample.keys())
    #     name = "A"
    #     series = sample[name]
    #     result = calc_segmentation_points_single_series(
    #         series, gru_score, name=f"{name} ({type}) ", show_plt=True)
    #     print("🎁comparison", segmentations, result)
    #     input()

    """
    分段电流序列作为先验知识输入，其准确性直接影响下游分类器的性能表现。然而，只基于二阶差分的分段点检测算法在处理由故障和环境电磁干扰引起的波动信号时存在明显局限性。如图3-7所示，在H4故障样本中，由于开关电路接触不良导致第二阶段电流发生突变，二阶差分最大值的方法错误地识别了分段点 P_2。
    """
    
    """
    展示H4故障样本中P2识别错误的问题
    """
    sample, segmentations = get_sample("H4")
    
    # 获取A相电流数据
    series = sample["A"]
    x, y = series
    duration = x[-1]
    
    # 计算二阶导数
    d1_result = get_d(series, smooth=True)
    d2_x, d2_y = get_d(d1_result, smooth=True)
    
    # 创建图形
    fig = plt.figure(dpi=150, figsize=(7, 3))
    
    # 创建两个y轴共用一个x轴的图
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # 绘制原始电流
    line1 = ax1.plot(x, y, 'b-', label='原始电流序列', linewidth=2)
    ax1.set_ylabel('电流 (A)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # 绘制二阶导数
    line2 = ax2.plot(d2_x, d2_y, 'r-', label='二阶差分序列', linewidth=1, alpha=0.7)
    ax2.set_ylabel('二阶差分值', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # 找出所有峰值点
    peak_idx, properties = find_peaks(d2_y, prominence=0)  # 使用prominence而不是height来找到所有峰值
    prominences = properties["prominences"]
    
    # 标注真实分段点
    real_pt1, real_pt2 = segmentations
    if real_pt1:
        ax1.axvline(x=real_pt1, color='g', linestyle='--', label='真实分段点P₁')
    
    # 找出P₁之后且在转换阶段内的峰值点
    later_peaks = [p for p in peak_idx if real_pt1 + 0.1 < d2_x[p] < real_pt2]
    if later_peaks:
        # 找到最大峰值（错误识别的P₂）
        wrong_p2_idx = later_peaks[np.argmax([d2_y[p] for p in later_peaks])]
        ax2.scatter(d2_x[wrong_p2_idx], d2_y[wrong_p2_idx], 
                   color='red', s=100, marker='*', label='错误识别的P2')
        
    if real_pt2:
        ax1.axvline(x=real_pt2, color='g', linestyle='--', label='真实分段点')

    # 标注其他峰值点（只标注转换阶段内的峰值点）
    other_peaks = [p for p in later_peaks if p != wrong_p2_idx]
    if other_peaks:
        ax2.scatter(np.array(d2_x)[other_peaks], np.array(d2_y)[other_peaks], 
                   color='gray', s=50, alpha=0.5, label='候选峰值点')
    
    # 添加波动区域的阴影
    if real_pt1 and real_pt2:
        ax1.axvspan(real_pt1, real_pt2, color='yellow', alpha=0.1, label='转换阶段')
    
    # 设置x轴标签和范围
    ax1.set_xlabel('时间 (s)')
    ax1.set_xlim(0, duration)
    
    # 添加网格
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    
    # 添加散点图例
    scatter_legend = [
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
                   markersize=12, label=r'错误识别的$P_{2}$'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=8, alpha=0.5, label='候选峰值点'),
        plt.Line2D([0], [0], color='g', linestyle='--', label='真实分段点'),
    ]
    
    # 合并所有图例
    plt.legend(lines + scatter_legend, 
              labels + [l.get_label() for l in scatter_legend], 
              loc='upper right',
              framealpha=1,  # 设置图例背景不透明
              )     # 确保图例在最上层

    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig("./screenshots/segmentation_p2_error_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()