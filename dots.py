import xlrd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
spot = xlrd.open_workbook('./附件1.xlsx')
sheet1 = spot.sheet_by_index(0)
spotx = sheet1.col_values(1)[1:]  # 排除表头
spoty = sheet1.col_values(2)[1:]  # 排除表头
population = sheet1.col_values(3)[1:]  # 排除表头

# 可视化
plt.figure(figsize=(10, 10))  # 设置图形大小，保证长宽相等
plt.scatter(spotx, spoty, c='blue', label='Population Centers', edgecolors='k')  # 绘制散点图
plt.title('The Location of the 50 Dots Needed', fontsize=14, color='darkred', fontweight='bold')
plt.xlabel('km')  # 横坐标标签
plt.ylabel('km')  # 纵坐标标签
plt.grid(True)  # 网格线
plt.gca().set_aspect('equal', adjustable='box')  # 设置坐标轴比例为1:1
plt.legend()  # 显示图例
plt.tight_layout()  # 自动调整以使图形填满整个图像区域
plt.show()
