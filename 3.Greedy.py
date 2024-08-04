import xlrd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# 读取数据
spot = xlrd.open_workbook('./附件1.xlsx')
sheet1 = spot.sheet_by_index(0)
spotx = np.array(sheet1.col_values(1)[1:])  # 排除表头
spoty = np.array(sheet1.col_values(2)[1:])  # 排除表头
data = np.column_stack((spotx, spoty))

D = 5  # 临界距离

def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def greedy_algorithm_with_constraints(D, data):
    num_points = len(data)
    covered = [False] * num_points
    stations = []

    # 生成所有可能的点
    all_possible_points = {(x, y) for x in range(35) for y in range(35)}
    
    while not all(covered):
        best_station = None
        best_cover_count = 0
        
        for candidate_station in all_possible_points:
            if candidate_station in stations:  # 避免重复选择已选的点
                continue
            
            cover_count = 0
            for j in range(num_points):
                if not covered[j] and calculate_distance(candidate_station[0], candidate_station[1], data[j][0], data[j][1]) <= D:
                    cover_count += 1
            
            if cover_count > best_cover_count:
                best_cover_count = cover_count
                best_station = candidate_station
        
        if best_station is None:  # 无法找到新的有效充电站
            break
        
        # 记录最佳充电站并标记覆盖的点
        stations.append(best_station)
        for j in range(num_points):
            if calculate_distance(best_station[0], best_station[1], data[j, 0], data[j, 1]) <= D:
                covered[j] = True
        
        # 从所有可能点中移除已选点
        all_possible_points.discard(best_station)
    
    return stations

stations = greedy_algorithm_with_constraints(D, data)
min_k = len(stations)

print(f"最少需要建设的充电站数量: {min_k}")
for i, (x, y) in enumerate(stations):
    print(f"充电站 {i+1}: ({x:.2f}, {y:.2f})")

# 可视化
plt.figure(figsize=(10, 8))
plt.scatter(spotx, spoty, c='blue', label='Population Centers')

for i, (x, y) in enumerate(stations):
    plt.scatter(x, y, c='red', marker='x', s=100, label=f'Charging Station {i+1}')
    
    # 绘制充电站与其覆盖的点的连线
    for j in range(len(spotx)):
        if calculate_distance(x, y, spotx[j], spoty[j]) <= D:
            plt.plot([x, spotx[j]], [y, spoty[j]], 'k--', linewidth=0.5)
    
    # 绘制覆盖范围圆圈
    circle = Circle((x, y), D, color='gray', fill=False, linestyle='--', linewidth=1.5)
    plt.gca().add_patch(circle)

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)  # 将图例放在图形外部
plt.title('Greedy Algorithm for Charging Station Placement')
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')  # 设置坐标轴比例为1:1
plt.tight_layout()  # 自动调整以使图形填满整个图像区域
plt.show()
