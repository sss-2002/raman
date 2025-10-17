# arrangement/utils/classifier.py
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

def knn_classify(train_data, train_labels, test_data, k=5):
    """K近邻分类"""
    train_data = train_data.T
    test_data = test_data.T
    predictions = []
    for test_sample in test_data:
        distances = np.sqrt(np.sum((train_data - test_sample) **2, axis=1))
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [train_labels[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        predictions.append(most_common)
    return np.array(predictions)

def evaluate_classification(test_labels, predictions):
    """评估分类结果"""
    accuracy = accuracy_score(test_labels, predictions)
    kappa = cohen_kappa_score(test_labels, predictions)
    cm = confusion_matrix(test_labels, predictions)
    return {
        'accuracy': accuracy,
        'kappa': kappa,
        'confusion_matrix': cm
    }
