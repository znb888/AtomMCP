import sys
import os

# 1. 定义包含 dsmc_GPU 包的根目录的路径
package_root_path = r'E:\movefire\code\DSMC_GPU'

# 2. 检查该路径是否已在Python的搜索路径中，如果不在，则添加到列表的最前面
#    使用 insert(0, ...) 比 append(...) 更好，可以优先搜索您的模块
if package_root_path not in sys.path:
    sys.path.insert(0, package_root_path)

# --- 现在 Python 知道去哪里找 dsmc_GPU 了，下面的导入将正常工作 ---
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent, ImageContent, BlobResourceContents
from typing import List, Dict, Any
import logging
from dsmc_GPU.collision import collision
from dsmc_GPU.trap_function import cross_trap_move, cross_trap_move_evaporative, transport_trap
import time
import torch
import numpy as np
from dsmc_GPU.sample import metropolis_sampling_mass
mcp = FastMCP()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

@mcp.tool()
def run_dsmc_simulation_from_config(scenarios_list: List[Dict[str, Any]]):
    """
    根据传入的场景配置列表，执行完整的DSMC模拟流程。

    Args:
        scenarios_list (list): 一个包含多个字典的列表。
                               每个字典定义了模拟的一个阶段，
                               需要包含 "param_file", "trap_function", 
                               和 "evolution_function_name" 三个键。
                               注意：trap_function 可以是函数对象或函数名字符串。
    返回是一个字符串，告诉用户模拟流程执行完毕
    """
    #
    # --- 全局配置 ---
    # 参数文件所在的目录
    PARAMS_DIR = r"E:\movefire\code\DSMC_GPU\parametersfile"
    # 模拟基本参数
    DT = 0.04
    BETA = 1
    SAMPLING_STEPS = 2000

    # --- 模拟初始化 ---
    print("正在通过Metropolis抽样生成初始轨迹...")
    initial_trap_params = np.array([40, 0.3, 1, 1, 1, 0])
    trajectories = metropolis_sampling_mass(
        transport_trap, initial_trap_params, 2*BETA, SAMPLING_STEPS, np.ones(100000, dtype=int)
    ).numpy()
    trajectories[:, 1] += 2
    trajectories[:, 4] += 1
    print("初始轨迹生成完毕。")

    # 初始化模拟对象 (确保提供了所有必要的参数)
    evolution = collision(
        trajectories, DT, transport_trap, torch.float32, device="cuda",
        grid_width=0.005, grid_height=0.005, grid_length=0.005,
        Nx=1000, Ny=1000, Nz=1000
    )

    # 定义可用的陷阱函数映射
    trap_functions = {
        "transport_trap": transport_trap,
        "cross_trap_move": cross_trap_move,
        "cross_trap_move_evaporative": cross_trap_move_evaporative
    }

    # --- 执行模拟流程 ---
    print("\n开始执行模拟流程...")
    s = time.time()

    # 循环遍历从外部传入的场景配置列表
    for scenario in scenarios_list:
        # 1. 从文件加载当前阶段的参数
        param_filepath = os.path.join(PARAMS_DIR, scenario['param_file'])
        print(f"正在从 {param_filepath} 加载参数...")
        parameters = np.load(param_filepath)

        # 2. 设置陷阱函数
        # trap_function 可以是函数对象或函数名字符串
        # 如果trap_function是字符串，则从映射中获取实际函数
        if isinstance(scenario['trap_function'], str):
            trap_func = trap_functions[scenario['trap_function']]
            print(f"设置陷阱函数为: {scenario['trap_function']}")
        else:
            trap_func = scenario['trap_function']
            print(f"设置陷阱函数为: {trap_func.__name__}")
        evolution.trap = trap_func
        
        # 3. 动态获取演化函数并执行
        # 使用 getattr 从字符串名称获取 evolution 对象上的实际方法
        evolution_func_name = scenario['evolution_function_name']
        evolution_func = getattr(evolution, evolution_func_name)

        N1 = int(parameters.shape[0] / 2)
        print(f"执行演化函数: {evolution_func.__name__}")
        evolution_func(parameters.reshape(N1, 2, -1))

    # --- 结束并可视化 ---
    print(f"\n模拟流程执行完毕。总耗时: {time.time() - s:.2f} 秒")

    print("正在生成动画...")
    #evolution.animate_evolution_vispy(10, 2, 2, 2)
    return TextContent(type="text", text="模拟流程执行完毕")


def main():
    """Starts the MCP server."""
    logger.info("Starting QuantumSim MCP server...")
    logger.info("Access the interactive API documentation at http://127.0.0.1:8000/docs")
    #mcp.run('streamable-http')
    mcp.run('stdio')

if __name__ == "__main__":
    main()