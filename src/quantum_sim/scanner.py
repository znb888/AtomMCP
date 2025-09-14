# -*- coding: utf-8 -*-
"""
scanner.py - 量子仿真参数扫描模块
"""
import numpy as np
from typing import Any, Callable, Iterable
from qutip import Qobj, Result
from tqdm.auto import tqdm
from .simulator import Simulator # 相对导入
import inspect

class Scanner:
    """
    一个通用的参数扫描器。
    ...
    """
    def __init__(self, simulator: Simulator, base_sim_args: dict):
        self.simulator = simulator
        self.base_sim_args = base_sim_args

    def run(self, 
            sweep_values: Iterable,
            setup_function: Callable[[Any], dict],
            post_process_function: Callable,
            description: str = "Scanning Parameter") -> tuple:
        """执行参数扫描。

        该方法会遍历 `sweep_values` 中的每一个值，调用 `setup_function` 来配置
        当次仿真的参数，然后运行仿真，最后通过 `post_process_function` 处理
        并收集结果。扫描过程会使用 tqdm 显示进度条。

        Args:
            sweep_values (Iterable): 
                要扫描的参数值序列, 例如 `np.linspace(0, 1, 51)`。
            setup_function (Callable[[Any], dict]):
                一个回调函数，它接受 `sweep_values` 中的单个值，并返回一个
                包含本次仿真所需的可变参数的字典 (例如 `{'sequences': ...}`)。
            post_process_function (Callable):
                一个回调函数，用于处理单次仿真的结果并返回需要被收集的值。
                为了提供灵活性，该函数可以接受两种签名：
                1. `post_process(result: qutip.Result) -> Any`
                   当您只关心仿真结果时使用。
                2. `post_process(value: Any, result: qutip.Result) -> Any`
                   当您同时需要当前扫描的参数值和仿真结果时使用。
                扫描器会自动检测并使用正确的调用方式。
            description (str, optional): 
                显示在 tqdm 进度条上的描述文字。默认为 "Scanning Parameter"。

        Returns:
            tuple: 一个包含 `(sweep_list, collected_results)` 的元组。
        """
        collected_results = []
        
        # 将 sweep_values 转换为列表，以便安全地获取长度
        sweep_list = list(sweep_values)
        
        progress_bar = tqdm(sweep_list, desc=description)

        # 2. 获取后处理函数的参数签名
        post_process_sig = inspect.signature(post_process_function)
        num_post_params = len(post_process_sig.parameters)

        for value in progress_bar:
            variable_sim_args = setup_function(value)
            current_sim_args = {**self.base_sim_args, **variable_sim_args}
            result = self.simulator.run(**current_sim_args)
            # 3. [核心修正] 根据参数数量，用不同的方式调用后处理函数
            if num_post_params == 2:
                # 新版函数，接收 (value, result)
                processed_result = post_process_function(value, result)
            elif num_post_params == 1:
                # 旧版函数，只接收 result
                processed_result = post_process_function(result)
            else:
                raise TypeError(
                    f"post_process_function 必须接收 1 个 (result) 或 2 个 (value, result) 参数, "
                    f"但您提供的函数接收 {num_post_params} 个。"
                )
            collected_results.append(processed_result)
            
            # <<< 核心修正开始
            # 检查 processed_result 的类型并选择合适的显示方式
            postfix_dict = {'value': f"{value:.3f}"}
            if isinstance(processed_result, (list, tuple)):
                # 如果是列表或元组, 将其中每个元素格式化后拼接成字符串
                display_str = ", ".join([f"{item:.3f}" for item in processed_result])
                postfix_dict['result'] = f"({display_str})"
            elif isinstance(processed_result, (int, float, np.number)):
                # 如果是数字，使用科学记数法
                postfix_dict['result'] = f"{processed_result:.3e}"
            else:
                # 对于其他类型，直接转换为字符串
                postfix_dict['result'] = str(processed_result)
            
            progress_bar.set_postfix(postfix_dict)
            # <<< 核心修正结束

        return sweep_list, collected_results



# class Scanner:
#     """
#     一个通用的参数扫描器。
#     它接收一个仿真器实例和固定的仿真参数，
#     然后根据用户的指令扫描指定参数并收集结果。
#     """
#     def __init__(self, simulator: Simulator, base_sim_args: dict):
#         """
#         初始化扫描器。

#         Args:
#             simulator (Simulator): 用于执行单次仿真的仿真器实例。
#             base_sim_args (dict): 一个包含所有仿真运行中保持不变的参数的字典,
#                                   例如 {'psi0': ..., 't_list': ..., 'e_ops': ...}。
#         """
#         self.simulator = simulator
#         self.base_sim_args = base_sim_args

#     def run(self, 
#             sweep_values: Iterable,
#             setup_function: Callable[[Any], dict],
#             post_process_function: Callable[[Result], Any],
#             description: str = "Scanning Parameter") -> tuple:
#         """
#         执行参数扫描。

#         Args:
#             sweep_values (Iterable): 要扫描的参数值序列, 例如 np.linspace(0, 1, 51)。
#             setup_function (Callable[[Any], dict]):
#                 一个回调函数，它接受 sweep_values 中的单个值，
#                 并返回一个包含本次仿真所需的可变参数的字典 (例如 {'sequences': ...})。
#             post_process_function (Callable[[Result], Any]):
#                 一个回调函数，它接受单次仿真返回的 qutip.Result 对象，
#                 并返回一个需要被记录的标量值。

#         Returns:
#             tuple: 一个包含 (sweep_values, collected_results) 的元组。
#         """
#         collected_results = []

#         progress_bar = tqdm(sweep_values)

#         for value in progress_bar:
#             variable_sim_args = setup_function(value)
#             current_sim_args = {**self.base_sim_args, **variable_sim_args}
#             result = self.simulator.run(**current_sim_args)
#             processed_result = post_process_function(result)
#             collected_results.append(processed_result)
            
#             # 3. 使用 set_postfix 在进度条末尾动态显示当前值和结果
#             progress_bar.set_postfix(
#                 value=f"{value:.3f}", 
#                 result=f"{processed_result:.3e}"
#             )


#         print("...扫描完成。")
#         return list(sweep_values), collected_results