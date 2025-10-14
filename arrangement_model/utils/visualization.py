import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_spectrum(wavenumbers, spectrum, title="光谱图", xlabel="波数", ylabel="强度"):
    """绘制单条光谱"""
    plt.figure(figsize=(10, 4))
    plt.plot(wavenumbers, spectrum)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    return plt

def plot_multiple_spectra(wavenumbers, spectra, labels=None, title="多条光谱对比", 
                         xlabel="波数", ylabel="强度", legend_loc="upper right"):
    """绘制多条光谱"""
    plt.figure(figsize=(10, 4))
    if labels is None:
        labels = [f"光谱{i+1}" for i in range(spectra.shape[1])]
    
    for i in range(spectra.shape[1]):
        plt.plot(wavenumbers, spectra[:, i], label=labels[i])
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=legend_loc)
    plt.grid(True, linestyle='--', alpha=0.7)
    return plt

def plot_confusion_matrix(cm, class_names, title="混淆矩阵"):
    """绘制混淆矩阵热图"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    return plt

def plot_k_curve(wavenumbers, k_values, title="k值曲线", xlabel="波数", ylabel="k值"):
    """绘制k值曲线"""
    plt.figure(figsize=(10, 4))
    plt.plot(wavenumbers, k_values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    return plt
