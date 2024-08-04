import xlrd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# 读取数据
spot = xlrd.open_workbook('./附件1.xlsx')
sheet1 = spot.sheet_by_index(0)
spotx = np.array(sheet1.col_values(1)[1:])  # 排除表头
spoty = np.array(sheet1.col_values(2)[1:])  # 排除表头
data = np.column_stack((spotx, spoty))

D = 5  # 临界距离
initial_stations = 4  # 首期建设的充电站数量

def is_coverage_sufficient(stations, data, D):
    distances = np.min([np.linalg.norm(data - np.array(station), axis=1) for station in stations], axis=0)
    return np.all(distances <= D)

def greedy_initial_stations(data, initial_count):
    num_points = len(data)
    best_stations = []
    best_cover_count = 0

    for combination in combinations(range(num_points), initial_count):
        stations = data[list(combination)]
        distances = np.min([np.linalg.norm(data - np.array(station), axis=1) for station in stations], axis=0)
        cover_count = np.sum(distances <= D)

        if cover_count > best_cover_count:
            best_cover_count = cover_count
            best_stations = stations

    return best_stations

def add_station(stations, data, D):
    num_points = len(data)
    best_station = None
    best_cover_count = 0

    for i in range(num_points):
        if not any(np.all(station == data[i]) for station in stations):
            temp_stations = np.vstack([stations, data[i]])
            distances = np.min([np.linalg.norm(data - np.array(station), axis=1) for station in temp_stations], axis=0)
            cover_count = np.sum(distances <= D)

            if cover_count > best_cover_count:
                best_cover_count = cover_count
                best_station = data[i]

    return best_station

# 首期建设4个充电站
stations = greedy_initial_stations(data, initial_stations)

# 逐步增加充电站，直到覆盖所有用户
while not is_coverage_sufficient(stations, data, D):
    new_station = add_station(stations, data, D)
    if new_station is not None:
        stations = np.vstack([stations, new_station])
    else:
        break

# 打印最终建设的充电站数量和位置
print(f"最终建设的充电站数量: {len(stations)}")
for i, (x, y) in enumerate(stations):
    print(f"充电站 {i+1}: ({x:.2f}, {y:.2f})")

# 可视化
plt.figure(figsize=(10, 10))
plt.scatter(spotx, spoty, c='blue', label='Population Centers', edgecolors='k')
plt.scatter(stations[:, 0], stations[:, 1], c='red', marker='x', s=100, label='Charging Stations')

# 绘制每个充电站的覆盖范围
for station in stations:
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
