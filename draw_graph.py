import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

from sensor import SUPPORTED_SAMPLE_TYPES, get_sample
from extract_features import calc_features
from gru_score import GRUScore

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
