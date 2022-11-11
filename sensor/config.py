SUPPORTED_SAMPLE_TYPES = ["normal", "H1", "H2", "H3",
                          "H4", "H5", "H6", "F1", "F2", "F3", "F4", "F5"]  # 论文中所有可能的曲线类型
SAMPLE_RATE = 60  # 采样率 per second (算法模型对采样率有依赖，因此针对不同输入需插值到固定采样率)
USE_SIMULATED_DATA = True  # 是否使用模拟数据
