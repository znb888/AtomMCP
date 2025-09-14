from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent, ImageContent, BlobResourceContents
import logging
import sys 
import matplotlib.pyplot as plt
import io
import os
import numpy as np
import base64
# Set up logging (this just prints messages to your terminal for debugging)
# Create the MCP server object
mcp = FastMCP()

def load_and_filter_data(radius_threshold=1.5):
    """
    加载粒子轨迹数据，并根据指定的半径阈值进行过滤。

    Args:
        radius_threshold (float): 用于筛选数据点的最大半径。

    Returns:
        tuple: 一个包含过滤后的 x 坐标数组和 y 坐标数组的元组 (filtered_x, filtered_y)。
               如果文件未找到或加载失败，则返回 (None, None)。
    """
    # --- 1. 构造数据文件的绝对路径 ---

    full_data_path = r"C:\Users\Buantum\Desktop\lucien_mcp\capture\saved_trajectories.npy"

    # --- 2. 加载数据 ---
    print(f"尝试从绝对路径加载数据: {full_data_path}")
    
    if not os.path.exists(full_data_path):
        print(f"错误: 在指定路径未找到数据文件。")
        print("请确认 'saved_trajectories.npy' 文件与此脚本在同一个文件夹内。")
        return None, None

    try:
        data = np.load(full_data_path)
        print("数据加载成功。")
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return None, None

    # --- 3. 提取绘图所需的坐标 ---
    y_positions = data[:, 1]
    z_positions = data[:, 2]
    
    total_points_before_filter = len(z_positions)
    print(f"共加载了 {total_points_before_filter} 个粒子的位置信息。")

    # --- 3.5. 根据半径过滤数据 ---
    print(f"正在根据半径 r < {radius_threshold} 进行数据过滤...")
    
    radius_squared = z_positions**2 + y_positions**2
    filter_mask = radius_squared < radius_threshold**2
    
    filtered_z = z_positions[filter_mask]
    filtered_y = y_positions[filter_mask]

    total_points_after_filter = len(filtered_z)
    
    # 避免除以零的错误
    if total_points_before_filter > 0:
        percentage_kept = total_points_after_filter / total_points_before_filter * 100
        print(f"过滤后剩余 {total_points_after_filter} 个数据点 (保留了 {percentage_kept:.2f}%)。")
    else:
        print("过滤后剩余 0 个数据点。")
        
    return filtered_y, filtered_z
def plot_density_map(x_data, y_data):
    """
    根据给定的x和y坐标数据，使用面向对象的方法绘制二维柱密度图。

    Args:
        x_data (np.ndarray): 用于绘图的X坐标数据。
        y_data (np.ndarray): 用于绘图的Y坐标数据。
    """
    # --- 1. 创建 Figure 和 Axes 对象 ---
    # plt.subplots() 返回一个 Figure 对象和一个或多个 Axes 对象。
    # 这里我们只需要一个图，所以是 fig, ax。
    fig, ax = plt.subplots(figsize=(8, 8))
    
    print("正在生成柱密度图...")

    # --- 2. 在指定的 Axes 对象上绘制二维直方图 ---
    BINS = 80
    
    # plt.hist2d() 返回 (counts, xedges, yedges, image)
    # image 对象对于创建 colorbar 非常重要
    counts, xedges, yedges, im = ax.hist2d(
        x_data, 
        y_data, 
        bins=BINS, 
        cmap='viridis',
        cmin=1 
    )

    # --- 3. 美化图形（使用 ax 对象的方法） ---
    
    # 为指定的 image 创建 colorbar，并附加到 ax 上
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Counts per Bin (Column Density)') # 设置 colorbar 的标签

    # 设置坐标轴范围
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    # 设置坐标轴标签和标题
    ax.set_xlabel('Y Position')
    ax.set_ylabel('Z Position')
    ax.set_title('Column Density in the Y-Z Plane (Radius < 1.5)')
    
    # 设置坐标轴的纵横比为1:1
    ax.set_aspect('equal', adjustable='box')
    
    # 使用 fig.tight_layout() 来自动调整布局
    fig.tight_layout()

    # --- 4. 返回 Figure 和 Axes 对象 ---
    # 函数返回 fig 和 ax，让调用者决定如何处理（例如显示或保存）
    print("绘图完成。")
    return fig, ax

    #plt.show()
@mcp.tool()
def capture() ->ImageContent:
    """
    用户发起说拍照时，调用该函数，拍照并返回图片。

    Returns:
        dict: 包含图像内容和确认文本的字典。
    """

    filtered_x_coords, filtered_y_coords= load_and_filter_data(radius_threshold=1.5)

    # 步骤2：检查数据是否成功加载，如果成功则进行绘图
    if filtered_x_coords is not None and filtered_y_coords is not None:
        fig, ax=plot_density_map(filtered_x_coords, filtered_y_coords)


    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    plt.close(fig)

    return  ImageContent(type="image", data=img_base64, mimeType="image/png")
        

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Starts the MCP server."""
    logger.info('Starting QuantumSim MCP server...')
    mcp.run('streamable-http')

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    mcp.run()