import xlrd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# 读取数据
spot = xlrd.open_workbook('./附件1.xlsx')
sheet1 = spot.sheet_by_index(0)
spotx = np.array(sheet1.col_values(1)[1:])  # 排除表头
spoty = np.array(sheet1.col_values(2)[1:])  # 排除表头
data = np.column_stack((spotx, spoty))

def kmeans_clustering(n_clusters, data):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    return kmeans.cluster_centers_, kmeans.labels_

D = 5  # 临界距离
min_k = 1
max_k = 50
best_centers = None
best_labels = None

def is_coverage_sufficient(centers, data, D):
    distances = np.min([np.linalg.norm(data - center, axis=1) for center in centers], axis=0)
    return np.all(distances <= D)

for k in range(min_k, max_k + 1):
    centers, labels = kmeans_clustering(k, data)
    if is_coverage_sufficient(centers, data, D):
        best_centers = centers
        best_labels = labels
        break

if best_centers is not None:
    print(f"最少需要建设的充电站数量: {len(best_centers)}")
    for i, (x, y) in enumerate(best_centers):
        print(f"充电站 {i+1}: ({x:.2f}, {y:.2f})")

    # 可视化
    plt.figure(figsize=(10, 8))
    plt.scatter(spotx, spoty, c='blue', label='Population Centers')
    
    # 绘制簇心和各个簇中的点
    for i, center in enumerate(best_centers):
        cluster_points = data[best_labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i+1}')
        
        # 绘制簇心到簇中点的连线
        for point in cluster_points:
            plt.plot([center[0], point[0]], [center[1], point[1]], 'k--', linewidth=0.5)
        
        # 绘制覆盖范围圆圈
        circle = Circle(center, D, color='gray', fill=False, linestyle='--', linewidth=1.5)
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
