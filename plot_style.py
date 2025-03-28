"""
matplotlib作图样式配置文件
将此文件放在项目根目录下，在绘图时导入即可使用统一的样式
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
from pathlib import Path

def setup_style():
    """设置matplotlib的默认样式"""
    # 设置字体路径
    SIMSUN_PATH = '/usr/share/fonts/truetype/custom/simsun.ttc'
    TIMES_PATH = '/usr/share/fonts/truetype/custom/times.ttf'
    
    # 添加字体
    fm.fontManager.addfont(SIMSUN_PATH)
    fm.fontManager.addfont(TIMES_PATH)
    
    # 设置默认字体
    plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设置默认图形大小和分辨率
    plt.rcParams['figure.figsize'] = [8, 6]
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    
    # 设置网格线样式
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.color'] = 'gray'
    
    # 设置刻度朝内
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    # 设置坐标轴线宽
    plt.rcParams['axes.linewidth'] = 1.5
    
    # 设置刻度标签大小
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    
    # 设置图例字体大小
    plt.rcParams['legend.fontsize'] = 16
    
    # 设置标题字体大小
    plt.rcParams['axes.titlesize'] = 16
    
    # 设置坐标轴标签字体大小
    plt.rcParams['axes.labelsize'] = 16

# 在导入模块时自动设置样式
setup_style() 