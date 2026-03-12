#!/usr/bin/env python
# license removed for brevity
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
data_file_cl = '/home/nvidia/P2P_fast_try/P2P_fast/src/planning/src/path1/Record_Path/path/data_watch.txt'
data = np.loadtxt(data_file_cl)

data_file_o = '/home/nvidia/P2P_fast_try/P2P_fast/src/planning/src/path1/Record_Path/path/trajectory.txt' 
data_cl = np.loadtxt(data_file_o)

# 提取数据列
x = data[:, 1]  # 全局路线
y = data[:, 2]

x1 = data_cl[:, 0]  # 实际路线
y1 = data_cl[:, 1]

# 计算大于指定误差值的点
def find_error_points(x, y, x1, y1, threshold=0.2):
    error_points = []
    for i in range(len(x1)):
        # 计算当前点 (x1[i], y1[i]) 到所有点 (x[j], y[j]) 的距离
        distances = np.sqrt((x - x1[i])**2 + (y - y1[i])**2)
        # 找到最小距离
        min_distance = np.min(distances)
        if min_distance > threshold:
            error_points.append((x1[i], y1[i]))
    return error_points

# 计算大于0.2的误差点
error_points = find_error_points(x, y, x1, y1, threshold=0.2)

# 绘制图形
plt.figure()
plt.plot(x, y, 'r-', label='Original Data')
plt.plot(x1, y1, 'g-', label='New Data')
if error_points:
    error_x, error_y = zip(*error_points)
    plt.scatter(error_x, error_y, color='b', label='Error Points', zorder=5)
plt.title('Comparison of Data Sets')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
