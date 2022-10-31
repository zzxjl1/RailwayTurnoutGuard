"""pca lda demo"""
from dnn_classification import generate_data
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from extract_features import calc_features
from paper import get_paper_sample
from sensor import SUPPORTED_SAMPLE_TYPES

SAMPLE_COUNT = 1000
x, y = generate_data(SAMPLE_COUNT)
y = [i.index(1) for i in y]  # 将one-hot编码转换为索引

#x = MinMaxScaler().fit_transform(x)
x = StandardScaler().fit_transform(x)


def pca_1():
    """PCA降维到1维"""
    global x, y
    pca = PCA(n_components=1)
    newX = pca.fit_transform(x)
    print(newX)
    print(pca.explained_variance_ratio_)

    # 画图
    plt.scatter(newX, newX, c=y)
    plt.title("pca_1")
    plt.show()


def pca_2():
    """PCA降维到2维"""
    global x, y
    pca = PCA(n_components=2)
    newX = pca.fit_transform(x)
    print(newX)
    print(pca.explained_variance_ratio_)

    # 画图
    plt.scatter(newX[:, 0], newX[:, 1], c=y)
    plt.title("pca_2")
    plt.show()


def pca_3():
    """PCA降维到3维"""
    global x, y
    pca = PCA(n_components=3)
    newX = pca.fit_transform(x)
    print(newX)
    print(pca.explained_variance_ratio_)

    # 画图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(newX[:, 0], newX[:, 1], newX[:, 2], c=y)
    plt.title("pca_3")
    plt.show()


def lda_2():
    """LDA降维到2维"""
    global x, y
    clf = LinearDiscriminantAnalysis(n_components=2)
    newX = clf.fit_transform(x, y)
    print(newX)

    # 画图
    plt.scatter(newX[:, 0], newX[:, 1], c=y)
    plt.title("lda_2")
    plt.show()


def lda_3():
    """LDA降维到3维"""
    global x, y
    clf = LinearDiscriminantAnalysis(n_components=3)
    newX = clf.fit_transform(x, y)
    print(newX)

    # 画图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(newX[:, 0], newX[:, 1], newX[:, 2], c=y)
    plt.title("lda_3")
    plt.show()


def lda_predict():
    """LDA预测"""
    global x, y
    clf = LinearDiscriminantAnalysis()
    clf.fit(x, y)
    x, y = generate_data(20)
    print("-----lda_predict-----")
    print("prediction result:", clf.predict(x))
    print("real:", [i.index(1) for i in y])
    """
    type = "F1"
    sample = get_paper_sample(type, show_plt=False)
    features = list(calc_features(sample).values())
    print(SUPPORTED_SAMPLE_TYPES[clf.predict([features])[0]], type)
    """


if __name__ == "__main__":
    pca_1()
    pca_2()
    pca_3()
    lda_2()
    lda_3()
    lda_predict()
    """由此可见，pca、lda效果都不好"""
