import xlrd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# 读取数据
spot = xlrd.open_workbook('./附件1.xlsx')
sheet1 = spot.sheet_by_index(0)
spotx = np.array(sheet1.col_values(1)[1:])  # 排除表头
spoty = np.array(sheet1.col_values(2)[1:])  # 排除表头
population = np.array(sheet1.col_values(3)[1:])  # 排除表头
data = np.column_stack((spotx, spoty))

D = 5  # 临界距离
n_initial_clusters = 4
max_clusters = 50
n_attempts = 10  # 尝试次数以确保结果一致

def calculate_coverage(stations, data, D):
    #计算每个点的最小距离，并返回满足条件的点数量
    distances = np.min(cdist(data, stations), axis=1)
    return np.sum(distances <= D)

def kmeans_with_initial_stations(data, n_initial_clusters, max_clusters, random_state):
    #使用K-means确定初期充电站并逐步添加站点
    np.random.seed(random_state)
    best_stations = None
    best_cover_count = 0

    # 初期选择4个充电站
    initial_indices = np.random.choice(data.shape[0], n_initial_clusters, replace=False)
    initial_stations = data[initial_indices]

    for k in range(n_initial_clusters, max_clusters + 1):
        if k == n_initial_clusters:
            # 对于初期站点
            kmeans = KMeans(n_clusters=k, init=initial_stations, n_init=1, random_state=random_state).fit(data)
        else:
            # 对于增加新站点
            new_indices = np.random.choice(data.shape[0], k - n_initial_clusters, replace=False)
            new_stations = data[new_indices]
            new_centers = np.vstack([initial_stations, new_stations])
            kmeans = KMeans(n_clusters=k, init=new_centers, n_init=1, random_state=random_state).fit(data)
        
        stations = kmeans.cluster_centers_
        
        # 计算覆盖情况
        cover_count = calculate_coverage(stations, data, D)
        if cover_count > best_cover_count:
            best_cover_count = cover_count
            best_stations = stations
        
        # 如果所有点都被覆盖，停止添加
        if best_cover_count >= len(data):
            break

    return best_stations

# 尝试不同的随机种子，选择最佳结果
best_stations = None
best_cover_count = 0

for attempt in range(n_attempts):
    random_state = attempt  # 通过尝试不同的随机种子
    stations = kmeans_with_initial_stations(data, n_initial_clusters, max_clusters, random_state)
    if stations is not None:
        cover_count = calculate_coverage(stations, data, D)
        if cover_count > best_cover_count:
            best_cover_count = cover_count
            best_stations = stations

if best_stations is not None:
    print(f"最终建设的充电站数量: {len(best_stations)}")
    for i, (x, y) in enumerate(best_stations):
        print(f"充电站 {i+1}: ({x:.2f}, {y:.2f})")

    # 可视化
    plt.figure(figsize=(10, 10))
    plt.scatter(spotx, spoty, c='blue', label='Population Centers', edgecolors='k')
    plt.scatter(best_stations[:, 0], best_stations[:, 1], c='red', marker='x', s=100, label='Charging Stations')

    # 绘制每个充电站的覆盖范围
    for station in best_stations:
        circle = plt.Circle(station, D, color='gray', fill=False, linestyle='--', linewidth=1.5)
        plt.gca().add_patch(circle)

    plt.title('Charging Stations Coverage')
    plt.xlabel('X Coordinate (km)')
    plt.ylabel('Y Coordinate (km)')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.tight_layout()
    plt.show()
