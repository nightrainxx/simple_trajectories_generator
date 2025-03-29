"""
检查栅格数据的简单脚本
"""
import numpy as np
import rasterio
from matplotlib import pyplot as plt

# 打开栅格文件
with rasterio.open("data/input/landcover_30m_100km.tif") as src:
    # 读取数据
    data = src.read(1)
    
    # 输出基本信息
    print(f"数据形状: {data.shape}")
    print(f"数据类型: {data.dtype}")
    print(f"唯一值: {np.unique(data)}")
    
    # 采样输出一些数据点的值
    print("\n数据采样:")
    for i in range(0, data.shape[0], 500):
        for j in range(0, data.shape[1], 500):
            print(f"位置 ({i}, {j}): {data[i, j]}")
    
    # 绘制简单的热力图
    plt.figure(figsize=(10, 10))
    plt.imshow(data, cmap='tab20')
    plt.colorbar(label='地表类型')
    plt.title('地表覆盖数据 - 原始值')
    plt.savefig('landcover_heatmap.png', dpi=300)
    print("\n原始热力图已保存为 landcover_heatmap.png") 