import numpy as np
import scipy.special
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent, ImageContent, BlobResourceContents
import logging
import sys
import matplotlib.pyplot as plt
import io
import os
import base64

# --------------------------------------------------------------------------
# 1. Professional Logging Setup
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# 2. MCP Server and Tools Definition
# --------------------------------------------------------------------------
mcp = FastMCP()

@mcp.tool()
def trap(positions: list[list[float]], parameters: list[float]) -> list[float]:
    """
    计算一个洛伦兹形状的各向异性势阱。
    数学公式:
    $$ V(\\mathbf{r}) = -\\frac{U}{a(x-x_0)^2 + b(y-y_0)^2 + c(z-z_0)^2 + e} $$
    Args:
        positions (list[list[float]]): (N, 3) 的粒子坐标列表, e.g., [[x1,y1,z1], [x2,y2,z2], ...]。
        parameters (list[float]): [center_position_x, center_position_y, center_position_z, U, a, b, c, e]。
    Returns:
        list[float]: 每个粒子位置上的势能值列表。
    """
    positions = np.array(positions)
    center_position_x, center_position_y, center_position_z, U, a, b, c, e = parameters
    result = -U / (a * (positions[:, 0] - center_position_x)**2 + b * (positions[:, 1] - center_position_y)**2 + c * (positions[:, 2] - center_position_z)**2 + e)
    return result.tolist()

@mcp.tool()
def polytrap(positions: list[list[float]], parameters: list[float]) -> list[float]:
    """
    计算一个高阶多项式势阱（平顶阱）。
    数学公式:
    $$ V(\\mathbf{r}) = -U \\left(1 - \\left[ (a(x-x_0))^{50} + (b(y-y_0))^{50} + (c(z-z_0))^{50} \\right] \\right) $$
    Args:
        positions (list[list[float]]): (N, 3) 的粒子坐标列表。
        parameters (list[float]): [center_position_x, center_position_y, center_position_z, U, a, b, c]。
    Returns:
        list[float]: 每个粒子位置上的势能值列表。
    """
    positions = np.array(positions)
    center_position_x, center_position_y, center_position_z, U, a, b, c = parameters
    result = -U * (1 - ((a * (positions[:, 0] - center_position_x))**50 + (b * (positions[:, 1] - center_position_y))**50 + (c * (positions[:, 2] - center_position_z))**50))
    return result.tolist()

@mcp.tool()
def doubletrap(positions: list[list[float]], parameters: list[float]) -> list[float]:
    """
    计算一个对称的双阱势，由两个洛伦兹势阱叠加而成。
    Args:
        positions (list[list[float]]): (N, 3) 的粒子坐标列表。
        parameters (list[float]): 与 `trap` 函数相同的8个参数。
    Returns:
        list[float]: 每个粒子位置上的总势能值。
    """
    parameters_1 = parameters.copy()
    parameters_1[0] = -parameters[0]
    result = np.array(trap(positions, parameters)) + np.array(trap(positions, parameters_1))
    return result.tolist()

@mcp.tool()
def double_harmonictrap(positions: list[list[float]], parameters: list[float]) -> list[float]:
    """
    计算一个对称的双谐振势阱。
    Args:
        positions (list[list[float]]): (N, 3) 的粒子坐标列表。
        parameters (list[float]): 与 `harmonic_trap` 函数相同的7个参数。
    Returns:
        list[float]: 每个粒子位置上的总势能值。
    """
    parameters_1 = parameters.copy()
    parameters_1[0] = -parameters[0]
    result = np.array(harmonic_trap(positions, parameters)) + np.array(harmonic_trap(positions, parameters_1))
    return result.tolist()

@mcp.tool()
def movetrap(positions: list[list[float]], parameters: list[float]) -> list[float]:
    """
    计算由两个独立的洛伦兹势阱组成的复合势。
    Args:
        positions (list[list[float]]): (N, 3) 的粒子坐标列表。
        parameters (list[float]): 包含16个浮点数的列表，前后8个分别定义两个势阱。
    Returns:
        list[float]: 每个粒子位置上的总势能值。
    """
    result = np.array(trap(positions, parameters[:8])) + np.array(trap(positions, parameters[8:]))
    return result.tolist()

@mcp.tool()
def harmonic_trap(positions: list[list[float]], parameters: list[float]) -> list[float]:
    """
    计算一个标准的各向异性谐振势阱。
    数学公式:
    $$ V(\\mathbf{r}) = -U \\left(1 - a(x-x_0)^2 - b(y-y_0)^2 - c(z-z_0)^2 \\right) $$
    Args:
        positions (list[list[float]]): (N, 3) 的粒子坐标列表。
        parameters (list[float]): [center_x, center_y, center_z, U, a, b, c]。
    Returns:
        list[float]: 每个粒子位置上的势能值。
    """
    positions = np.array(positions)
    center_position_x, center_position_y, center_position_z, U, a, b, c = parameters
    result = -U * (1 - a * (positions[:, 0] - center_position_x)**2 - b * (positions[:, 1] - center_position_y)**2 - c * (positions[:, 2] - center_position_z)**2)
    return result.tolist()

@mcp.tool()
def rota_z_trap(positions: list[list[float]], parameters: list[float]) -> list[float]:
    """
    计算一个绕Z轴旋转了60度（pi/3）的洛伦兹势阱。
    Args:
        positions (list[list[float]]): (N, 3) 的粒子坐标列表。
        parameters (list[float]): 与 `trap` 函数相同的8个参数。
    Returns:
        list[float]: 每个粒子位置上的势能值。
    """
    positions = np.array(positions)
    center_position_x, center_position_y, center_position_z, U, a, b, c, e = parameters
    phi = np.pi / 3
    x = positions[:, 0]
    y = positions[:, 1]
    rota_z = [np.cos(phi), np.sin(phi)]
    new_x = x * rota_z[0] - y * rota_z[1]
    new_y = x * rota_z[1] + y * rota_z[0]
    result = -U / (a * (new_x - center_position_x)**2 + b * (new_y - center_position_y)**2 + c * (positions[:, 2] - center_position_z)**2 + e)
    return result.tolist()

@mcp.tool()
def cross_trap(positions: list[list[float]], parameters: list[float]) -> list[float]:
    """
    计算交叉光偶极阱的势能。
    Args:
        positions (list[list[float]]): (N, 3) 的粒子坐标列表。
        parameters (list[float]): 与 `trap` 函数相同的8个参数。
    Returns:
        list[float]: 每个粒子位置上的总势能值。
    """
    result = np.array(trap(positions, parameters)) + np.array(rota_z_trap(positions, parameters))
    return result.tolist()

@mcp.tool()
def cross_trap_move(positions: list[list[float]], parameters: list[float]) -> list[float]:
    """
    计算一个交叉光偶极阱的势能，其中两个光束的深度可以独立调节。
    Args:
        positions (list[list[float]]): (N, 3) 的粒子坐标列表。
        parameters (list[float]): [center_x, y, z, U1, U2, a, b, c, e]。
    Returns:
        list[float]: 每个粒子位置上的总势能值。
    """
    center_position_x, center_position_y, center_position_z, U1, U2, a, b, c, e = parameters
    parameters1 = [center_position_x, center_position_y, center_position_z, U1, a, b, c, e]
    parameters2 = [center_position_x, center_position_y, center_position_z, U2, a, b, c, e]
    result = np.array(trap(positions, parameters1)) + np.array(rota_z_trap(positions, parameters2))
    return result.tolist()

@mcp.tool()
def evaporative(positions: list[list[float]], U: float) -> list[float]:
    """
    计算一个线性的“蒸发”势或重力势。
    数学公式: $$ V(z) = U \\cdot z $$
    Args:
        positions (list[list[float]]): (N, 3) 的粒子坐标列表。
        U (float): 线性势的斜率。
    Returns:
        list[float]: 每个粒子位置上的势能值。
    """
    positions = np.array(positions)
    result = U * positions[:, 2]
    return result.tolist()

@mcp.tool()
def cross_trap_move_evaporative(positions: list[list[float]], parameters: list[float]) -> list[float]:
    """
    计算一个可变功率交叉阱与一个线性蒸发势的叠加。
    Args:
        positions (list[list[float]]): (N, 3) 的粒子坐标列表。
        parameters (list[float]): [center_x, y, z, U1, U2, a, b, c, e, H]。
    Returns:
        list[float]: 每个粒子位置上的总势能值。
    """
    center_x, center_y, center_z, U1, U2, a, b, c, e, H = parameters
    
    common_prefix = (center_x, center_y, center_z)
    common_suffix = (a, b, c, e)
    parameters1 = [*common_prefix, U1, *common_suffix]
    parameters2 = [*common_prefix, U2, *common_suffix]
    
    result = np.array(trap(positions, parameters1)) + \
             np.array(rota_z_trap(positions, parameters2)) + \
             np.array(evaporative(positions, H))
    return result.tolist()

@mcp.tool()
def cross_trap_evaporative(positions: list[list[float]], parameters: list[float]) -> list[float]:
    """
    计算一个标准交叉阱与一个线性蒸发势的叠加。
    Args:
        positions (list[list[float]]): (N, 3) 的粒子坐标列表。
        parameters (list[float]): 前8个定义交叉阱，最后一个是蒸发势斜率H。
    Returns:
        list[float]: 每个粒子位置上的总势能值。
    """
    result = np.array(cross_trap(positions, parameters[:8])) + np.array(evaporative(positions, parameters[-1]))
    return result.tolist()

@mcp.tool()
def transport_trap(positions: list[list[float]], parameters: list[float]) -> list[float]:
    """
    计算一个带线性项的洛伦兹势阱，用于模拟原子输运。
    数学公式: $$ V(\\mathbf{r}) = -\\frac{U}{(ax)^2 + (by)^2 + (cz)^2 + e} + A \\cdot x $$
    Args:
        positions (list[list[float]]): (N, 3) 的粒子坐标列表。
        parameters (list[float]): [U, a, b, c, e, Acceleration]。
    Returns:
        list[float]: 每个粒子位置上的势能值。
    """
    positions = np.array(positions)
    U, a, b, c, e, Acceleration = parameters
    result = -U / ((a * positions[:, 0])**2 + (b * positions[:, 1])**2 + (c * positions[:, 2])**2 + e) + Acceleration * positions[:, 0]
    return result.tolist()

@mcp.tool()
def center_position(N: int, dt: float) -> list[list[float]]:
    """
    生成一个沿x轴正弦振荡的中心位置轨迹。
    Args:
        N (int): 轨迹中的时间步数。
        dt (float): 每个时间步的持续时间（在此实现中未使用，但为通用性保留）。
    Returns:
        list[list[float]]: 一个 (N, 3) 的数组，每一行是 [x, y, z] 坐标。
    """
    center_positions = []
    # N-1 needs to be non-zero to avoid division by zero
    if N > 1:
        for i in range(N):
            center_positions.append([10 * np.sin(np.pi / (N - 1) * i), 0, 0])
    elif N == 1:
         center_positions.append([0.0, 0.0, 0.0])
    return center_positions

@mcp.tool()
def center_positions_group(N1: int, N2: int, dt: float) -> list[list[list[float]]]:
    """
    生成一组（N1个）正弦振荡的中心位置轨迹。
    Args:
        N1 (int): 要生成的轨迹数量。
        N2 (int): 每个轨迹中的时间步数。
        dt (float): 每个时间步的持续时间。
    Returns:
        list: 包含 N1 个 (N2, 3) numpy 数组的列表，每个数组都是一个独立的轨迹。
    """
    center_positions = []
    for i in range(N1):
        center_positions.append(center_position(N2, dt))
    return center_positions


@mcp.tool()
def write_config_to_txt(trap_function: str, evolution_function: str, filepath: str) -> str:
    """
    将输入的三个字符串按照指定格式写入一个 .txt 文件，并返回该字符串。

    Args:
        trap_function: a string for the trap function name.
        evolution_function: a string for the evolution function name.
        filepath: the full path (including filename) to the output .txt file.

    Returns:
        The formatted string that was written to the file.
    """
    # 按照指定的格式构建字符串
    # 使用 f-string 和三重引号可以方便地构建多行字符串并插入变量
    output_string = f'''"trap_function": {trap_function},
"evolution_function": {evolution_function},
"filepath": "{filepath}"'''

    try:
        # 使用 'w' 模式打开文件进行写入。如果文件不存在，它将被创建。
        # 如果文件已存在，其内容将被覆盖。
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(output_string)
        print(f"成功将配置写入文件： {filepath}")
    except IOError as e:
        print(f"写入文件时出错： {e}")

    # 返回格式化后的字符串
    return output_string

# --------------------------------------------------------------------------
# 3. Main Execution Block
# --------------------------------------------------------------------------
def main():
    """Starts the MCP server."""
    logger.info("Starting QuantumSim MCP server...")
    logger.info("Access the interactive API documentation at http://127.0.0.1:8000/docs")
    #mcp.run('streamable-http')
    mcp.run('stdio')

if __name__ == "__main__":
    main()