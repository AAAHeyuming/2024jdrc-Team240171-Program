import xlrd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

# 读取数据
spot = xlrd.open_workbook('./附件1.xlsx')
sheet1 = spot.sheet_by_index(0)
spotx = np.array(sheet1.col_values(1)[1:])  # 排除表头
spoty = np.array(sheet1.col_values(2)[1:])  # 排除表头
data = np.column_stack((spotx, spoty))

D = 5  # 临界距离
min_k = 1
max_k = 50
best_centers = None
best_labels = None

def is_coverage_sufficient(centers, data, D):
    distances = np.min(cdist(data, centers), axis=1)
    return np.all(distances <= D)

def kmeans_with_points_as_centers(n_clusters, data):
    num_points = data.shape[0]
    centers = np.zeros((n_clusters, data.shape[1]))

    # 初始化簇心
    kmeans_init = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1, random_state=0).fit(data)
    centers = kmeans_init.cluster_centers_
    
    while True:
        # 计算距离并归类到簇
        distances = euclidean_distances(data, centers)
        labels = np.argmin(distances, axis=1)
        
        # 更新簇心（必须是数据点）
        new_centers = np.copy(centers)
        for i in range(n_clusters):
            cluster_points = data[labels == i]
            if cluster_points.shape[0] > 0:
                new_centers[i] = cluster_points.mean(axis=0)
            else:
                # 如果簇没有数据点，则重新随机选择一个簇心
                new_centers[i] = data[np.random.choice(num_points)]

        # 确保簇心是最近的数据点
        new_centers = np.array([data[np.argmin(np.linalg.norm(data - center, axis=1))] for center in new_centers])
        
        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels

def optimize_centers(centers, labels, data, D):
    #进一步优化已找到的簇心
    n_clusters = centers.shape[0]
    for _ in range(10):  # 迭代优化次数
        distances = euclidean_distances(data, centers)
        labels = np.argmin(distances, axis=1)
        new_centers = np.copy(centers)
        for i in range(n_clusters):
            cluster_points = data[labels == i]
            if cluster_points.shape[0] > 0:
                new_centers[i] = cluster_points.mean(axis=0)
        new_centers = np.array([data[np.argmin(np.linalg.norm(data - center, axis=1))] for center in new_centers])
        
        if is_coverage_sufficient(new_centers, data, D):
            return new_centers, labels
    return centers, labels

# 尝试不同的k值，找到最小的满足条件的k
for k in range(min_k, max_k + 1):
    centers, labels = kmeans_with_points_as_centers(k, data)
    if is_coverage_sufficient(centers, data, D):
        best_centers, best_labels = optimize_centers(centers, labels, data, D)
        min_k = k
        break

if best_centers is not None:
    print(f"最少需要建设的充电站数量: {min_k}")
    for i, (x, y) in enumerate(best_centers):
        print(f"充电站 {i+1}: ({x:.2f}, {y:.2f})")

    # 可视化
    plt.figure(figsize=(10, 8))
    plt.scatter(spotx, spoty, c='blue', label='Population Centers')

    for i, center in enumerate(best_centers):
        cluster_points = data[best_labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i+1}')
        
        # 绘制簇心到簇中点的连线
        for point in cluster_points:
            plt.plot([center[0], point[0]], [center[1], point[1]], 'k--', linewidth=0.5)
        
        # 绘制覆盖范围圆圈
        circle = plt.Circle(center, D, color='gray', fill=False, linestyle='--', linewidth=1.5)
        plt.gca().add_patch(circle)

    plt.scatter(best_centers[:, 0], best_centers[:, 1], c='red', marker='x', s=100, label='Charging Stations')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)  # 将图例放在图形外部
    plt.title('K-means Clustering with Connecting Lines')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')  # 设置坐标轴比例为1:1
    plt.tight_layout()  # 自动调整以使图形填满整个图像区域
    plt.show()
