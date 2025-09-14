# QuantumSim

一个用于高保真量子动力学仿真和控制脉冲设计的Python库，特别关注超导量子比特系统中常见的DRAG（Derivative Removal by Adiabatic Gate）修正技术。

## 核心功能

- **模块化物理模型**: 使用抽象基类 (`HamiltonianModel`) 定义物理系统，轻松切换不同的量子模型（已实现 `DuffingOscillatorModel`）。
- **灵活的脉冲构建**: 采用“组合优于继承”的设计哲学，将脉冲解构为三个独立的层次：
    1.  **包络 (Envelope)**: 定义脉冲的数学形状（如 `GaussianEnvelope`, `CosineEnvelope`），支持拼接 (`+`) 和重复 (`.repeat()`)。
    2.  **控制序列 (ControlSequence)**: 将包络与物理控制通道（如 `MicrowaveSequence`）相结合。
    3.  **修正器 (Corrector)**: 将高级修正技术（如 `DRAGCorrector`）作为独立的“修改器”应用到基础脉冲上。
- **高层仿真引擎**: `Simulator` 类封装了 `qutip` 的 `mesolve`，自动处理实验室坐标系 (Lab Frame) 和旋转波近似 (RWA) 的哈密顿量构建。
- **自动化参数扫描**: 内置 `Scanner` 工具，可方便地对任意参数进行扫描和后处理，是进行脉冲校准和物理优化的利器。
- **可视化工具**: 包含绘图函数，用于可视化物理系统的能级 (`plot_energy_levels`) 或仿真中的控制波形。
****
## 安装

首先，克隆本仓库，然后使用 pip 以可编辑模式进行安装。这可以确保你的代码更改能够被立即应用。

```bash
git clone https://github.com/your-username/QuantumSim.git
cd QuantumSim
pip install -e .
```

## 快速上手

下面的例子展示了如何定义一个Duffing振子，创建一个带有DRAG修正的 $\pi$ 脉冲，并仿真其动力学过程。

```python
import numpy as np
from qutip import basis, num
import matplotlib.pyplot as plt

# 1. 从库中导入核心组件
from quantum_sim.hamiltonian import DuffingOscillatorModel
from quantum_sim.pulses import GaussianEnvelope, MicrowaveSequence, DRAGCorrector
from quantum_sim.simulator import Simulator

# 2. 定义物理系统 (Duffing Oscillator)
d = 3  # 考虑的能级数
omega_q = 2 * np.pi * 4.8  # 跃迁频率 (rad/ns)
anh = -2 * np.pi * 0.25   # 非谐性 (rad/ns)
model = DuffingOscillatorModel(d=d, omega_q=omega_q, anh=anh)

# 3. 构建控制脉冲 (带DRAG修正的 Pi 脉冲)
# 3.1 定义脉冲包络
duration = 10  # ns
sigma = duration / 4
# (此处振幅需要预先校准，这里仅为示例)
pi_amp = np.pi / (np.sqrt(2 * np.pi) * sigma) 

gauss_env = GaussianEnvelope(duration=duration, amp=pi_amp, sigma=sigma)

# 3.2 创建基础微波序列
# 载波频率通常设置为比特的跃迁频率
pi_pulse_base = MicrowaveSequence(envelope=gauss_env, carrier_freq=omega_q)

# 3.3 应用DRAG修正
# alpha 是无量纲的DRAG系数, 通常接近1
drag_corrector = DRAGCorrector(anh=anh, alpha=1.0)
pi_pulse_drag = drag_corrector.apply(pi_pulse_base)

# 4. 设置并运行仿真
simulator = Simulator(model)
psi0 = basis(d, 0)  # 初始状态 |0>
t_list = np.linspace(0, duration, 201)

# 算符，用于计算其期望值
e_ops = [num(d)] 

# 运行仿真，将修正后的脉冲应用到 'XY_drive' 通道
result = simulator.run(
    psi0,
    t_list,
    sequences={'XY_drive': pi_pulse_drag},
    use_rwa=True,
    e_ops=e_ops
)

# 5. 可视化结果
plt.figure(figsize=(8, 5))
plt.plot(t_list, result.expect[0], label='Population of $|n\rangle$ states')
plt.xlabel("Time (ns)")
plt.ylabel("Expectation Value")
plt.title("Population Dynamics under DRAG Pi-Pulse")
plt.legend()
plt.grid(True)
plt.show()
```

## 项目结构

- `src/quantum_sim/`: 包含所有核心库代码。
    - `hamiltonian.py`: 定义物理系统模型。
    - `pulses.py`: 定义控制脉冲的形状、序列和修正。
    - `simulator.py`: 核心的动力学仿真引擎。
    - `scanner.py`: 用于参数扫描的工具。
    - `plotting.py`: 可视化辅助函数。
- `examples/`: 包含一系列 Jupyter Notebook，展示了库在不同场景下的具体应用，是学习和理解本库的最佳实践。
- `requirements.txt`: 项目的Python依赖项。
- `setup.py`: 项目的安装配置文件。

## 依赖

本库构建于以下核心科学计算库之上：
- `qutip`: 用于量子动力学仿真的核心后端。
- `numpy` & `scipy`: 用于数值计算。
- `matplotlib`: 用于数据可视化。
- `tqdm`: 用于在参数扫描中显示进度条。

## License

This project is licensed under the MIT No Commercial (MIT-NC) License. See the [LICENSE](LICENSE) file for details.