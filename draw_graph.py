import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import torch
from auto_encoder import model_input_parse, BP_AE
from sensor import SUPPORTED_SAMPLE_TYPES, get_sample
from extract_features import calc_features
from gru_score import GRUScore


def graph_1():
    results = {}
    #SUPPORTED_SAMPLE_TYPES = ["F1", "F3"]
    for _ in range(10):
        for type in SUPPORTED_SAMPLE_TYPES:
            sample, _ = get_sample(type)
            features = calc_features(sample)
            for k, v in features.items():
                if k not in results:
                    results[k] = [(v, type)]
                else:
                    results[k].append((v, type))
    print(results)

    for k, v in results.items():
        result = {}
        for type in SUPPORTED_SAMPLE_TYPES:
            result[type] = []
        for i, j in v:
            result[j].append(i)
        results[k] = result
    print(results)

    data = np.random.lognormal(size=(37, 4), mean=1.5, sigma=1.75)

    # 只留下XX开头的特征
    results = {k: v for k, v in results.items() if k.startswith("A_stage1")}
    # 去除xxx开头的特征
    results = {k: v for k, v in results.items() if not k.startswith("power")}
    print(len(results))
    selection = [""]

    count = 1
    height, width = 5, 3
    #height, width = 3, 5
    for name, v in results.items():
        if count > height*width:
            break
        data = np.array(list(v.values())).T
        plt.subplot(height, width, count)
        #plt.boxplot(data, labels=SUPPORTED_SAMPLE_TYPES, notch=True)
        plt.violinplot(data,
                       showmeans=False,
                       showmedians=True,
                       vert=False
                       )
        plt.yticks(range(1, len(SUPPORTED_SAMPLE_TYPES)+1),
                   SUPPORTED_SAMPLE_TYPES)
        plt.tick_params(axis='y', labelsize=8)
        plt.title(name, fontsize=10)
        count += 1

    plt.tight_layout()
    plt.show()


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    #cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=False,
                   labeltop=False, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def graph_2():
    FILE_PATH = './models/auto_encoder/'  # 模型保存路径
    DEVICE = torch.device('cpu')

    plt.figure(figsize=(7, 6), dpi=150)
    height, width = 2, 1

    result_1, result_2 = [], []
    for type in SUPPORTED_SAMPLE_TYPES:
        sample, _ = get_sample(type)
        x = model_input_parse(sample)
        model_path = f"{FILE_PATH}{type}.pth"
        model = torch.load(model_path, map_location=DEVICE).to(DEVICE)
        model(x)
        t = model.bottle_neck_output.squeeze().detach().numpy()
        result_1.append(t)

    for type in SUPPORTED_SAMPLE_TYPES:
        sample, _ = get_sample(type)
        x = model_input_parse(sample)
        model_path = f"{FILE_PATH}normal.pth"
        model = torch.load(model_path, map_location=DEVICE).to(DEVICE)
        model(x)
        t = model.bottle_neck_output.squeeze().detach().numpy()
        result_2.append(t)

    ax = plt.subplot(height, width, 1)

    result_1, result_2 = np.array(result_1), np.array(result_2)

    im, _ = heatmap(result_1,
                    SUPPORTED_SAMPLE_TYPES,
                    ["" for _ in range(result_1.shape[1])],
                    ax=ax,
                    cmap=plt.get_cmap("PiYG", 7),
                    cbarlabel="Loss")
    #annotate_heatmap(im, valfmt="{x:.1f} t")
    ax = plt.subplot(height, width, 2)

    im, _ = heatmap(result_2,
                    SUPPORTED_SAMPLE_TYPES,
                    ["" for _ in range(result_2.shape[1])],
                    ax=ax,
                    cmap="RdYlBu",
                    cbarlabel="Loss")
    plt.show()


if __name__ == "__main__":
    # graph_1()
    graph_2()
