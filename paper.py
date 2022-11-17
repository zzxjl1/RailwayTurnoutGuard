
import pandas as pd
from sensor import interpolate, generate_power_series, show_sample
from bp_classification import get_label_from_result_pretty, predict, BP_Net, FusedFuzzyDeepNet, FuzzyLayer


def get_paper_sample(type="normal", show_plt=False):
    df = pd.read_excel("paper.xlsx", sheet_name=type)
    # print(df)
    result = {}
    for name in ["A", "B", "C"]:
        # 获取名为 Phase $name的列的序号
        index = df.columns.get_loc(f"Phase {name}")
        # 获取该列的数据
        x = df.iloc[1:, index:index+1].values
        # 去除空值
        x = x[~pd.isnull(x)].reshape(-1)

        y = df.iloc[1:, index+1:index+2].values
        y = y[~pd.isnull(y)].reshape(-1)

        #print(name, x, y)
        assert len(x) == len(y)
        result[name] = interpolate(x, y)

    result["power"] = generate_power_series(result)

    if show_plt:
        show_sample(result, type)
    return result


def get_all_subsheets():
    df = pd.read_excel("paper.xlsx", sheet_name=None)
    result = []
    for name, sheet in df.items():
        result.append(name)
    return result


if __name__ == "__main__":
    for name in get_all_subsheets():
        sample = get_paper_sample(type=name, show_plt=True)
        # print(calc_features(sample))
        result = predict(sample)
        print(result)
        label = get_label_from_result_pretty(result)  # 获取预测结果标签字符串
        print(label)
