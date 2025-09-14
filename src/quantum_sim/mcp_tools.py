# -*- coding: utf-8 -*-
"""
mcp_tools.py - quantum_sim library MCP tools
"""
import logging
import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

# Import the MCP classes based on the official documentation examples
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

# Import from our library
from .pulses import (
    GaussianEnvelope, CosineEnvelope, MicrowaveSequence, DRAGCorrector, ConstantEnvelope, ArbitraryFunctionEnvelope, Envelope, ControlSequence, ZSequence
)
from .hamiltonian import *

# from .pulses import (
#     GaussianEnvelope, CosineEnvelope, MicrowaveSequence, DRAGCorrector
# )

# --- MCP Server and Tools ---
logger = logging.getLogger(__name__)
# Per the Pydantic error, we must allow arbitrary types like the mcp.Image class.
mcp = FastMCP()

@mcp.tool()
def test_plot_return():
    """
    A simple tool to test the image return mechanism using Pillow and mcp.Image.
    It plots a simple line and returns the image.
    """
    try:
        # 1. Simple Plot
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3, 4])
        # ax.set_title("Simple Test Plot")
        # ax.grid(True)

        # Convert figure to bytes for the final MCP response
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        img_buffer.close()

        return ImageContent(type="image", data=img_base64, mimeType="image/png")

    except Exception as e:
        logger.exception("Error in test_plot_return")
        plt.close()
        return TextContent(type="text", text=f"Caught exception: {e}")
    

def _parse_envelope_from_json(envelope_json: str) -> Envelope:
    """
    一个内部辅助函数，用于将JSON字符串解析为单个Envelope对象。
    """
    env_def = json.loads(envelope_json)
    env_type = env_def.get('type')

    if env_type == 'gaussian':
        return GaussianEnvelope(duration=env_def['duration'], amp=env_def['amp'], sigma=env_def.get('sigma'))
    elif env_type == 'cosine':
        return CosineEnvelope(duration=env_def['duration'], amp=env_def['amp'])
    elif env_type == 'constant':
        return ConstantEnvelope(duration=env_def['duration'], amp=env_def['amp'])
    elif env_type == 'arbitrary':
        return ArbitraryFunctionEnvelope(
            duration=env_def['duration'],
            formula_str=env_def['formula_str'],
            params=env_def['params']
        )
    else:
        raise ValueError(f"不支持的包络类型: {env_type}")
    
@mcp.tool()
def create_and_visualize_envelope(
    envelope_json: str,
    t_steps: int = 1000
) -> list:
    """(第一步) 根据JSON定义，创建并可视化一个纯粹的包络形状 Ω(t)。

    功能与定位:
    此工具是量子控制序列设计的核心第一步，作为一个交互式的、可视化的
    包络形状设计器。它严格遵循“关注点分离”原则，只处理包络的纯粹
    数学形状 Ω(t)，不涉及任何相位或I/Q调制的概念。

    工作流:
    它被设计用于“人机协作”的迭代循环中。智能体根据用户（或论文）
    的需求提出一个 `envelope_json`，此工具将其可视化；用户根据图像
    给出反馈，智能体再修改 `envelope_json` 并重新调用，如此往复，
    直到用户对包络形状完全满意。最终的产物是一个经过验证的、可用于
    后续步骤的 `envelope_json` “工件”。

    Args:
        envelope_json (str):
            一个包含包络完整定义的JSON格式字符串。其结构由 "type" 字段决定。

            **通用字段 (所有类型都必须包含):**
            - `type` (str): 包络的类型。支持的值: "gaussian", "constant", "cosine", "arbitrary"。
            - `duration` (float): 脉冲的总时长，单位为纳秒 (ns)。

            **按类型分的专属字段:**

            1.  **`gaussian`**: 标准高斯函数形状。
                - `amp` (float): 峰值振幅。
                - `sigma` (float): 高斯函数的标准差 (ns)，控制脉冲宽度。
                - 示例:
                  ```json
                  {
                    "type": "gaussian",
                    "duration": 20,
                    "amp": 1.0,
                    "sigma": 5
                  }
                  ```

            2.  **`constant`**: 矩形脉冲。
                - `amp` (float): 脉冲的恒定振幅。
                - 示例:
                  ```json
                  {
                    "type": "constant",
                    "duration": 50,
                    "amp": 0.5
                  }
                  ```

            3.  **`cosine`**: 升余弦脉冲的半周期形状。
                - `amp` (float): 脉冲的峰值振幅。
                - 示例:
                  ```json
                  {
                    "type": "cosine",
                    "duration": 30,
                    "amp": 0.8
                  }
                  ```

            4.  `arbitrary`: 由任意数学公式定义的包络。
                - `formula_str` (str): 
                    一个数学表达式字符串。
                    - **规则1**: 必须包含时间变量 `'t'`。
                    - **规则2**: 可以使用 `numpy` 和 `sympy` 支持的标准数学函数，
                      例如 `cos`, `sin`, `exp`, `tanh`, `cosh`, `sqrt` 等。
                    - **规则3**: 公式中的所有其他变量（参数），都必须在
                      下方的 `"params"` 字典中有对应的定义。

                - `params` (dict): 
                    一个JSON对象（字典），包含了 `"formula_str"` 中用到的所有参数
                    及其对应的数值。

                - **重要使用约定 (Best Practice):**
                  为了避免歧义和错误，请遵循以下规范：
                  - 顶层的 `"duration"` 字段定义了脉冲的**物理时间边界**（即 `t` 的取值范围 `[0, duration]`）。
                  - 如果您的**数学公式**本身需要用到总时长这个参数，
                    **强烈建议**在公式中使用一个不同的符号（例如 `T`, `width`, `total_time`），
                    然后在 `"params"` 字典中为这个符号赋值，使其等于顶层的 `"duration"` 值。
                  - **请勿**在 `formula_str` 中直接使用名为 `duration` 的变量。

                - **建议:**
                    - 使用 `t` 作为时间变量。
                    - 使用 `T` 作为总时长参数。
                    - 涉及到需要用分段函数时，可以使用 `Piecewise`。

                - **示例 (应用了最佳实践的平滑开关脉冲):**
                  下面的例子展示了如何正确地使用一个依赖于总时长的公式。
                  ```json
                  {
                    "type": "arbitrary",
                    "duration": 20,
                    "formula_str": "A * tanh(t / sigma) * tanh((T - t) / sigma)",
                    "params": {
                      "A": 1.0,
                      "sigma": 4,
                      "T": 20
                    }
                  }
                  ```
                  *在这个例子中，`"duration": 20` 告诉系统脉冲持续20ns。公式中的 `T` 是一个数学参数，我们通过 `"T": 20` 将其值也设为20ns，逻辑非常清晰。*
                  

        t_steps (int, optional): 
            用于生成图像的时间点数量，决定了曲线的平滑度。
            Defaults to 1000.

    Returns:
        list: 
            一个包含两项内容的列表:
            1. `ImageContent`: 包含包络形状图。
            2. `TextContent`: 包含一条确认信息和可用于后续步骤的JSON工件。
    """
    try:
        # --- 1. 解析JSON，构建包络对象 ---
        envelope = _parse_envelope_from_json(envelope_json)
        duration = envelope.duration
        
        # --- 2. 计算波形 ---
        t_list = np.linspace(0, duration, t_steps)
        omega_t = envelope.value(t_list)

        # --- 3. 绘图 ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t_list, omega_t, label='Ω(t) (Base Envelope Shape)', color='red', linewidth=2)
        ax.set_title("Pure Envelope Shape Visualization")
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()

        # --- 4. 编码图像并准备返回内容 ---
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        plt.close(fig)


        confirmed_text = (
            f"包络形状已确认。这是可用于下一步的JSON工件:\n"
            f"<envelope_artifact>{envelope_json}</envelope_artifact>"
        )

        return [
            ImageContent(type="image", data=img_base64, mimeType="image/png"),
            TextContent(type="text", text=confirmed_text)
        ]

    except Exception as e:
        logger.exception("Error in create_and_visualize_envelope")
        plt.close()
        return [TextContent(type="text", text=f"创建包络时发生错误: {e}")]


def _parse_single_sequence_from_json(sequence_json: str) -> ControlSequence:
    """
    一个内部辅助函数，用于将描述单个序列的JSON字符串解析为ControlSequence对象。
    """
    seq_def = json.loads(sequence_json)
    seq_type = seq_def.get('type')

    # 步骤 1: 解析内嵌的 envelope JSON。这是设计的关键，实现了代码复用。
    # 我们需要将 envelope 对象转换回 JSON 字符串以传递给已有的解析器。
    envelope_json_str = json.dumps(seq_def['envelope'])
    envelope = _parse_envelope_from_json(envelope_json_str)

    # 步骤 2: 根据序列类型，创建对应的 ControlSequence 对象
    base_sequence: ControlSequence
    if seq_type == 'microwave':
        base_sequence = MicrowaveSequence(
            envelope=envelope,
            carrier_freq=seq_def.get('carrier_freq_ghz', 0) * 2 * np.pi,
            phi=seq_def.get('phi', 0)
        )
        # 步骤 3 (可选): 如果有DRAG定义，则应用修正
        if 'drag' in seq_def:
            drag_def = seq_def['drag']
            anh_rad_ns = drag_def.get('anh_ghz', -0.3 * 2 * np.pi) * 2 * np.pi
            corrector = DRAGCorrector(anh=anh_rad_ns, alpha=drag_def['alpha'])
            return corrector.apply(base_sequence)
        return base_sequence
        
    elif seq_type == 'z':
        base_sequence = ZSequence(envelope=envelope)
        return base_sequence
        
    else:
        raise ValueError(f"不支持的序列类型: {seq_type}")
    


# mcp_tools.py

@mcp.tool()
def visualize_sequence(
    sequence_json: str,
    t_steps: int = 2000
) -> ImageContent | TextContent:
    """
    根据JSON定义，可视化一个单一的控制序列（例如Microwave或Z序列）。这通常是包络确定后的第二步，所以 sequence_json 很可能和上一步 create_and_visualize_envelope 的确认的工件 envelope_json 有关联。

    功能与定位:
    此工具用于详细预览一个完整控制序列的时域波形，包括其慢变包络
    以及（对于微波序列）合成后的实验室坐标系信号。它接收一个描述
    单个序列的JSON，该JSON内部嵌套了由 `create_and_visualize_envelope`
    工具设计和验证的 `envelope` JSON工件，用于画图返回图像，最后确认 sequence_json。

    参数:
        sequence_json (str):
            描述单个控制序列的JSON字符串。其顶层结构由 "type" 字段决定。

            **通用字段:**
            - `type` (str): 序列的类型。支持 "microwave" 或 "z"。
            - `envelope` (dict): 一个JSON对象，其结构与 `create_and_visualize_envelope`
              工具所使用的 `envelope_json` 完全相同。

            **按类型分的专属字段:**

            1.  **`microwave`**: 微波驱动序列。
                - `carrier_freq_ghz` (float): 载波频率，单位为GHz。
                - `phi` (float): 驱动的全局相位，单位为弧度。
                - `drag` (dict, optional): DRAG修正的参数。
                    - `alpha` (float): DRAG系数。
                    - `anh_ghz` (float): 量子比特的非谐性(GHz)，例如-0.3。
                - 示例 (一个带DRAG修正的任意形状(sech)脉冲):
                  ```json
                  {
                    "type": "microwave",
                    "carrier_freq_ghz": 4.8,
                    "phi": 0,
                    "drag": {
                        "alpha": 0.8,
                        "anh_ghz": -0.33
                    },
                    "envelope": {
                        "type": "arbitrary",
                        "duration": 30,
                        "formula_str": "A / cosh((t - t0) / sigma)",
                        "params": { "A": 0.9, "t0": 15, "sigma": 4 }
                    }
                  }
                  ```

            2.  **`z`**: Z向磁通脉冲序列。
                - 无专属字段。
                - 示例 (一个高斯Z脉冲):
                  ```json
                  {
                    "type": "z",
                    "envelope": {
                        "type": "gaussian",
                        "duration": 20,
                        "amp": -0.2,
                        "sigma": 5
                    }
                  }
                  ```

        t_steps (int, optional): 
            绘图的时间点数量，决定了曲线的平滑度。
            默认为 2000.

    返回:
        ImageContent: 包含序列波形图的图片。
        TextContent: 如果发生错误，返回错误信息。
    """
    try:
        # --- 1. 解析JSON，构建序列对象 ---
        sequence = _parse_single_sequence_from_json(sequence_json)
        duration = sequence.duration
        t_list = np.linspace(0, duration, t_steps, endpoint=False)

        # --- 2. 获取慢变包络 ---
        envelopes = sequence.get_envelopes(t_list)

        # --- 3. 动态绘图 ---
        # 检查序列类型以决定绘图布局
        is_microwave = isinstance(sequence, MicrowaveSequence)
        num_plots = 2 if is_microwave else 1
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 6 * num_plots), sharex=True)
        # 确保axes总是一个可迭代的数组
        if num_plots == 1:
            axes = [axes]

        fig.suptitle('Sequence Visualization', fontsize=16)

        # --- 图 1: 慢变包络 (I/Q, Z, etc.) ---
        ax1 = axes[0]
        for name, values in envelopes.items():
            ax1.plot(t_list, values, label=name)
        ax1.set_ylabel('amp (rad/ns)')
        ax1.set_title('Control Envelopes')
        ax1.legend()
        ax1.grid(True)

        # --- 图 2: 实验室坐标系信号 (仅限微波序列) ---
        if is_microwave:
            ax2 = axes[1]
            i_env = envelopes.get('I_drive', np.zeros_like(t_list))
            q_env = envelopes.get('Q_drive', np.zeros_like(t_list))
            carrier_freq_rad_ns = sequence.carrier_freq
            
            lab_signal = i_env * np.cos(carrier_freq_rad_ns * t_list) + q_env * np.sin(carrier_freq_rad_ns * t_list)
            
            ax2.plot(t_list, lab_signal, label='Synthesized Signal', color='r')
            ax2.set_xlabel('time (ns)')
            ax2.set_ylabel('amp (rad/ns)')
            ax2.set_title('Lab Frame Signal')
            ax2.legend()
            ax2.grid(True)
        else:
            # 如果不是微波序列，最后一个图的x轴标签需要显示
            ax1.set_xlabel('time (ns)')

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # --- 4. 编码并返回图像 ---
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        plt.close(fig)

        return ImageContent(type="image", data=img_base64, mimeType="image/png")

    except Exception as e:
        logger.exception("Error in visualize_sequence")
        plt.close()
        return TextContent(type="text", text=f"可视化序列时发生错误: {e}")


def _parse_hamiltonian_from_json(hamil_json:str) -> Qobj:
    """
    从JSON字符串解析哈密顿量。
    """
    try:
        config_data = json.loads(hamil_json)
    except json.JSONDecodeError:
        raise ValueError("Input string is not a valid JSON string.")

    hamil_type = config_data.get('type')
    params = config_data.get('params')
    if hamil_type == 'arbitrary':
        return ArbitraryHamiltonianModel(
            system_dims = params.get('system_dims'),
            control_hamiltonian_terms = params.get('control_hamiltonian_terms'),
            drift_hamiltonian_terms = params.get('drift_hamiltonian_terms'),
        )
    elif hamil_type == 'duffin':
        return DuffingOscillatorModel(
            d = params.get('d'),
            omega_q = params.get('omega_q'),
            anh = params.get('anharmonicity'),
        )
    else:
        raise ValueError(f"Unknown Hamiltonian type: '{hamil_type}'")

def _plot_latex(hamil_model: HamiltonianModel, path: str, mode :str='drift'):
    fig = plt.figure(figsize=(6, 2))
    if mode == 'drift':
        hamil = hamil_model.get_drift_hamiltonian(mode='latex')
        fig.text(0.5, 0.5, r"$" + hamil + r"$", 
            fontsize=20, ha='center', va='center')
    elif mode == 'control':
        hamil = hamil_model.get_control_hamiltonian(mode='latex')
        for drive_name, channels in hamil.items():
            for channel, h in channels.items():
                fig.text(0.5, 0.5, rf"$H_{drive_name}_{channel} = " + h + r"$", 
                    fontsize=20, ha='center', va='center')
    
    # 移除坐标轴
    plt.axis('off')

    # 保存为图片
    plt.savefig(path+'\\'+mode+'_hamiltonian.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(path+'\\'+mode+'_hamiltonian.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()
    
    return fig

@mcp.tool()
def extract_hamiltonian_model(
    hamil_json: str,
    path: str,
) -> Qobj:
    """
        Constructs a QuTiP Hamiltonian (Qobj) from a JSON configuration.

        Args:
            config_input (str):
                A string that is either a path to a JSON file
                or a raw JSON string.

        Returns:
            List[qutip.Qobj]: The constructed Hamiltonian object, [drift_hamiltonian, control_hamiltonian].

        Raises:
            ValueError: If the JSON format is invalid, a key is missing,
                        or an operator name is not supported.
            IndexError: If a subsystem_index is out of bounds.

        -----------------------------------------------------------------------
        JSON Configuration Format Requirements:
        -----------------------------------------------------------------------
        The JSON object must contain two main keys: `type`, `params`. The form of `params` depends on `type`. 
        1. `type` (str):
            - The type of the Hamiltonian model.
            - Must be either "arbitrary" or "duffin".

        2. `params` (dict):
            - Parameters for the Hamiltonian model.
            - The form of `params` depends on `type`.
            - For "arbitrary" type:
                - `system_dims` (list[int]):
                    - A list of positive integers defining the dimension of each subsystem.
                    - The length of the list determines the total number of subsystems.
                    - Example: `[2, 10]` represents a 2-level system (qubit) coupled to a
                    10-level system (e.g., a truncated oscillator).
                - `drift_hamiltonian_terms` (list[dict]):
                    - A list of dictionaries, each defining a term of the drift Hamiltonian.
                    - Each term object has the following structure:
                        - `coefficient` (float or dict):
                            - The numerical coefficient of the term.
                            - For real numbers: `1.5`
                            - For complex numbers: `{"real": 2.0, "imag": 0.5}`
                        - `operators` (dict[int, list[str]]):
                            - A dictionary of operator objects that define each operator of every subsystem, which will be tensor-multiplied to form
                            the full operator.
                            - If the index of subsystem is not in the keys, the operator would be
                            the identity matrix for the subsystem.
                            - Each operator object has following structure:
                                - each key represents the index of subsystem
                                - each value represents the name of operator that would be tensor-multiplied to form the full operator.
                        - `comment` (str, optional):
                            - An optional field for adding comments and improving readability.
                            - Example: `"Term 1: X on qubit 0, Y on oscillator 1"`
                - `control_hamiltonian_terms` (list[dict]):
                    - A list of control Hamiltonian terms, each defining a way of how the microwave pulse interact with the system in the respect of operators.
                    - Each term object has the following structure:
                        - `comment` (str, optional):
                            - An optional field for adding comments and improving readability.
                            - Example: `"Term 1: XY drive on qubit 0"`
                        - `drive_name` (str):
                            - The name of the drive.
                            - Must be unique.
                            - Example: `"XY_drive"`
                        - `channels` (dict[list[dict]]):
                            - A dictionary of lists of dictionaries
                            - Keys are channel names (e.g., "I", "Q").
                            - Values are lists of dictionaries, each defining an operator of the drive's channel like "I" or "Q".
                            - Each dictionary has the following structure:
                                - `coefficient` (float or dict):
                                    - The numerical coefficient of the term.
                                    - For real numbers: `1.5`
                                    - For complex numbers: `{"real": 2.0, "imag": 0.5}`
                                - `operators` (dict[int, list[str]]):
                                    - A dictionary of operator objects that define each operator of every subsystem, which will be tensor-multiplied to form
                                    the full operator.
                                    - If the index of subsystem is not in the keys, the operator would be
                                    the identity matrix for the subsystem.
                                    - Each operator object has following structure:
                                        - each key represents the index of subsystem
                                        - each value represents the name of operator that would be tensor-multiplied to form the full operator.
                                
            - For "duffin" type:
                - `d` (int):
                    - The dimension of the oscillator.
                - `omega_q` (float):
                    - The qubit frequency.
                - `anharmonicity` (float):
                    - The anharmonicity of the oscillator.

        -----------------------------------------------------------------------
        JSON Template:
        -----------------------------------------------------------------------
        ```json
        {
            "type": "arbitrary",
            "params": {
                "system_dims": [2, 5],
                "drift_hamiltonian_terms": [
                    {
                        "comment": "Term 1: A single-body term on the qubit (subsystem 0).",
                        "coefficient": 2.0,
                        "operators": {
                            "0": ["sigmaz"]
                        }
                    },
                    {
                        "comment": "Term 2: A number operator on the oscillator (subsystem 1).",
                        "coefficient": 5.0,
                        "operators": {
                            "1": ["create","destroy"]
                        }
                    },
                    {
                        "comment": "Term 4: A constant energy offset (coefficient * Identity).",
                        "coefficient": -10.0,
                        "operators": {}
                    }
                ],
                "control_hamiltonian_terms": [
                    {
                        "comment": "Term 1: XY drive pulse on the qubit (subsystem 0).",
                        "drive_name": "XY_drive1",
                        "channels": {
                            "I": [
                                {
                                    "coefficient": 0.5,
                                    "operators": {
                                        "0": ["sigmap"]
                                    }
                                },
                                {
                                    "coefficient": 0.5,
                                    "operators": {
                                        "0": ["sigmam"]
                                    }
                                }
                            ],
                            "Q": [
                                {
                                    "coefficient": {"real": 0.0, "imag": 0.5},
                                    "operators": {
                                        "0": ["sigmap"]
                                    }
                                },
                                {
                                    "coefficient": {"real": 0.0, "imag": -0.5},
                                    "operators": {
                                        "0": ["sigmam"]
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        }
        ```
    """
    
    hamil_model = _parse_hamiltonian_from_json(hamil_json)
    print("drift hamiltonian:\n")
    _plot_latex(hamil_model, path, mode='drift')
    print("control hamiltonian:\n")
    _plot_latex(hamil_model, path, mode='control')
    
    return hamil_model

