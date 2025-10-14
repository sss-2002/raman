import numpy as np

def knn_classify(train_data, train_labels, test_data, k=5):
    """K近邻分类算法实现"""
    # 转置数据以适应样本数×特征数的格式
    train_data = train_data.T
    test_data = test_data.T

    predictions = []
    for test_sample in test_data:
        # 计算与所有训练样本的欧氏距离
        distances = np.sqrt(np.sum((train_data - test_sample) **2, axis=1))
        # 获取最近的k个样本的索引
        k_indices = np.argsort(distances)[:k]
        # 获取这些样本的标签
        k_nearest_labels = [train_labels[i] for i in k_indices]
        # 多数投票决定预测标签
        most_common = np.bincount(k_nearest_labels).argmax()
        predictions.append(most_common)
    return np.array(predictions)
