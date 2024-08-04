import xlrd
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from itertools import combinations

# 读取数据
spot = xlrd.open_workbook('./附件1.xlsx')
sheet1 = spot.sheet_by_index(0)
spotx = sheet1.col_values(1)[1:]  # 排除表头
spoty = sheet1.col_values(2)[1:]  # 排除表头

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def greedy_algorithm_with_constraints(D):
    num_points = len(spotx)
    points = list(range(num_points))
    covered = [False] * num_points
    stations = []
    
    while not all(covered):
        best_station = -1
        best_cover_count = 0
        
        for i in range(num_points):
            if i in [s[0] for s in stations]:  # 避免重复选择已选的点
                continue
            
            cover_count = 0
            for j in range(num_points):
                if not covered[j] and calculate_distance(spotx[i], spoty[i], spotx[j], spoty[j]) <= D:
                    cover_count += 1
            
            if cover_count > best_cover_count:
                best_cover_count = cover_count
                best_station = i
        
        if best_station == -1:  # 无法找到新的有效充电站
            break
        
        stations.append((best_station, spotx[best_station], spoty[best_station]))
        for j in range(num_points):
            if calculate_distance(spotx[best_station], spoty[best_station], spotx[j], spoty[j]) <= D:
                covered[j] = True
    
    return stations

D = 5  # 临界距离

stations = greedy_algorithm_with_constraints(D)
min_k = len(stations)

print(f"最少需要建设的充电站数量: {min_k}")
for i, (idx, x, y) in enumerate(stations):
    print(f"充电站 {i+1}: ({x:.2f}, {y:.2f})")

# 可视化
plt.figure(figsize=(12, 10))
plt.scatter(spotx, spoty, c='blue', label='Population Centers')

for idx, x, y in stations:
    plt.scatter(x, y, c='red', marker='x', s=100, label=f'Charging Station {idx+2}')
    
    # 绘制充电站与其覆盖的点的连线
    for j in range(len(spotx)):
        if calculate_distance(x, y, spotx[j], spoty[j]) <= D:
            plt.plot([x, spotx[j]], [y, spoty[j]], 'k--', linewidth=0.5)
    
    # 绘制覆盖范围圆圈
    circle = Circle((x, y), D, color='gray', fill=False, linestyle='--', linewidth=1.5)
    plt.gca().add_patch(circle)

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Greedy Algorithm for Charging Station Placement')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)  # 将图例放在图形外部
plt.grid(True)

# 设置坐标轴比例为1:1
plt.gca().set_aspect('equal', adjustable='box')

plt.tight_layout()  # 自动调整以使图形填满整个图像区域
plt.show()
