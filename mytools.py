# mytools.py

import umap
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns

def v_umap(data, labels=None):
    """
    对输入的数据data应用UMAP降维，并可视化结果。
    
    参数:
    - data: 要降维的数据集，形状应为(n_samples, n_features)。
    - labels: 每个样本对应的标签（可选），形状为(n_samples,)，用于以不同颜色显示不同的类别。
    
    返回:
    - umap_plot: UMAP降维后数据的二维散点图。
    """
    # 应用UMAP算法
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42, n_jobs=1)
    data_2d = reducer.fit_transform(data)

    # 创建图形
    plt.figure(figsize=(10, 7))
    if labels is not None:
        sns.scatterplot(x=data_2d[:,0], y=data_2d[:,1], hue=labels, palette='bright')
    else:
        plt.scatter(data_2d[:,0], data_2d[:,1], alpha=0.7, s=10)
    
    plt.title('UMAP Visualization')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    
    # 显示图形
    return plt.show()
##---------------------------------------------------------------------------------------------------------------##

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """
    绘制95%置信椭圆
    """
    ax = ax or plt.gca()
    
    # 计算协方差矩阵的特征值和特征向量
    vals, vecs = np.linalg.eigh(covariance)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    # 计算椭圆的角度和尺寸
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(vals) * np.sqrt(chi2.ppf(0.95, df=2))

    # 创建椭圆对象
    ellipse = Ellipse(xy=position, width=width, height=height, angle=theta, facecolor='none', **kwargs)

    # 添加椭圆到图表
    ax.add_artist(ellipse)
    return ellipse

def v_pca(data):
    """
    使用PCA对输入数据进行降维，并在2D图中可视化结果
    """
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(standardized_data)
    
    df_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])
    cov_matrix = np.cov(df_pca['PC1'], df_pca['PC2'])

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(x=df_pca['PC1'], y=df_pca['PC2'], s=10, alpha=0.6, label='Samples')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')

    ax.axhline(0, color='gray', linestyle='--')
    ax.axvline(0, color='gray', linestyle='--')

    draw_ellipse((0, 0), cov_matrix, ax=ax, edgecolor='red', alpha=0.8, label='95% Confidence Ellipse')

    ax.set_title('PCA 2D Visualization with 95% Confidence Ellipse')

    x_min, x_max = df_pca['PC1'].min(), df_pca['PC1'].max()
    y_min, y_max = df_pca['PC2'].min(), df_pca['PC2'].max()

    scale_x = np.sqrt(cov_matrix[0, 0]) * chi2.ppf(0.95, df=2)
    scale_y = np.sqrt(cov_matrix[1, 1]) * chi2.ppf(0.95, df=2)

    x_margin = max(abs(x_min), abs(x_max), scale_x) * 1.1
    y_margin = max(abs(y_min), abs(y_max), scale_y) * 1.1

    ax.set_xlim([-x_margin, x_margin])
    ax.set_ylim([-y_margin, y_margin])

    plt.legend()
    
    return plt.show()

##---------------------------------------------------------------------------------------------------------------##


