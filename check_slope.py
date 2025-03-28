import rasterio
import numpy as np
import os

def check_slope_data():
    """检查坡度数据文件"""
    print("开始检查坡度数据...")
    
    slope_path = os.path.join("data", "input", "slope_30m_100km.tif")
    print(f"读取坡度文件: {slope_path}")
    
    try:
        with rasterio.open(slope_path) as src:
            data = src.read(1)
            
            print("\n基本信息:")
            print(f"数据形状: {data.shape}")
            print(f"数据类型: {data.dtype}")
            print(f"最小值: {np.min(data):.2f}")
            print(f"最大值: {np.max(data):.2f}")
            print(f"平均值: {np.mean(data):.2f}")
            print(f"中位数: {np.median(data):.2f}")
            print(f"标准差: {np.std(data):.2f}")
            
            print("\n数据统计:")
            print(f"0值数量: {np.sum(data == 0)}")
            print(f"负值数量: {np.sum(data < 0)}")
            print(f"大于45度数量: {np.sum(data > 45)}")
            print(f"NaN数量: {np.sum(np.isnan(data))}")
            
            print("\n坡度分布:")
            bins = [0, 5, 15, 30, 45, np.inf]
            labels = ['平地', '缓坡', '中坡', '陡坡', '峭壁']
            hist, _ = np.histogram(data[~np.isnan(data)], bins=bins)
            for i, (count, label) in enumerate(zip(hist, labels)):
                print(f"  {label}: {count} 像素 ({count/data.size*100:.2f}%)")
                
            # 添加一些详细的分布信息
            print("\n详细分布:")
            percentiles = [0, 10, 25, 50, 75, 90, 100]
            for p in percentiles:
                value = np.percentile(data[~np.isnan(data)], p)
                print(f"  {p}百分位: {value:.2f}°")
                
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    check_slope_data() 