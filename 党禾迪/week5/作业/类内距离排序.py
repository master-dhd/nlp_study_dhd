# -*- coding: utf-8 -*-
# @FileName: 5.类内距离排序.py 
# @Time    : 2025/6/18 10:50
# @Author  : CodiX
# @Function: 此处填写脚本功能描述 


import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


def kmeans_intra_cluster_distances(data, n_clusters=3):
    """
    计算并排序 K-means 聚类结果中各类的类内距离

    参数:
    data: 输入数据，numpy array 格式
    n_clusters: 聚类的数量，默认为3

    返回:
    sorted_distances: 排序后的类内距离
    cluster_indices: 对应类别的索引
    kmeans: 训练好的 KMeans 模型
    """

    # 执行 K-means 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)

    # 获取聚类标签和聚类中心
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # 计算每个类的类内距离
    intra_distances = []

    for i in range(n_clusters):
        # 获取当前类别的所有样本点
        cluster_points = data[labels == i]

        if len(cluster_points) > 1:
            # 计算类内所有点对之间的平均距离
            distances = pairwise_distances(cluster_points)
            # 取上三角矩阵的平均值（排除对角线）
            mean_distance = np.sum(np.triu(distances, k=1)) / (len(distances) * (len(distances) - 1) / 2)
        else:
            mean_distance = 0

        intra_distances.append(mean_distance)

    # 对类内距离进行排序
    cluster_indices = np.argsort(intra_distances)
    sorted_distances = np.array(intra_distances)[cluster_indices]

    return sorted_distances, cluster_indices, kmeans


# 示例使用
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    n_samples = 300

    # 创建三个不同分布的聚类
    cluster1 = np.random.normal(0, 1, (n_samples, 2))
    cluster2 = np.random.normal(4, 2, (n_samples, 2))
    cluster3 = np.random.normal(-4, 0.5, (n_samples, 2))

    # 合并数据
    X = np.vstack([cluster1, cluster2, cluster3])

    # 执行聚类和排序
    distances, indices, kmeans = kmeans_intra_cluster_distances(X, n_clusters=3)

    # 打印结果
    print("类内距离（从小到大排序）:")
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        print(f"类别 {idx}: {dist:.4f}")

    # 可视化结果
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    colors = ['r', 'g', 'b']

    for i in range(3):
        cluster_points = X[kmeans.labels_ == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    c=colors[i], label=f'Cluster {i}')

    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                marker='x', s=200, linewidths=3, color='black', label='Centroids')
    plt.title('K-means Clustering Results')
    plt.legend()
    plt.show()
