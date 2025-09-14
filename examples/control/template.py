import numpy as np
from qutip import basis, num
import matplotlib.pyplot as plt

# 1. 从库中导入核心组件
from quantum_sim.hamiltonian import DuffingOscillatorModel
from quantum_sim.pulses import CosineEnvelope, MicrowaveSequence, DRAGCorrector
from quantum_sim.simulator import Simulator

# 2. 定义物理系统 (Duffing Oscillator)
d = 3  # 考虑的能级数
omega_q = 2 * np.pi * 4.8  # 跃迁频率 (rad/ns)
anh = -2 * np.pi * 0.25   # 非谐性 (rad/ns)
model = DuffingOscillatorModel(d=d, omega_q=omega_q, anh=anh)

# 3. 构建控制脉冲 (带DRAG修正的 Pi 脉冲)
# 3.1 定义脉冲包络
duration = 10  # ns

# 对于 (A/2)*(1-cos(2*pi*t/T)) 形式的余弦包络,
# 其积分面积为 A*T/2。
# 要实现绕X轴旋转pi角 (pi脉冲), 积分面积需要为 pi。
# 因此 A * duration / 2 = pi  =>  A = 2 * pi / duration
pi_amp = 2 * np.pi / duration

cos_env = CosineEnvelope(duration=duration, amp=pi_amp)

# 3.2 创建基础微波序列
# 载波频率通常设置为比特的跃迁频率
pi_pulse_base = MicrowaveSequence(envelope=cos_env, carrier_freq=omega_q)

# 3.3 应用DRAG修正
# alpha 是无量纲的DRAG系数, 通常接近1
drag_corrector = DRAGCorrector(anh=anh, alpha=1.0)
pi_pulse_drag = drag_corrector.apply(pi_pulse_base)

# 4. 设置并运行仿真
simulator = Simulator(model)
psi0 = basis(d, 0)  # 初始状态 |0>
t_list = np.linspace(0, duration, 201)

# 算符，用于计算其期望值 (追踪每个能级的布居数)
e_ops = [basis(d, i) * basis(d, i).dag() for i in range(d)] 

# 运行仿真，将修正后的脉冲应用到 'XY_drive' 通道
print("--- 开始仿真 ---")
result = simulator.run(
    psi0,
    t_list,
    sequences={'XY_drive': pi_pulse_drag},
    use_rwa=True,
    e_ops=e_ops,
    show_waveforms=False # 显示控制波形以供验证
)
print("-- 仿真完成 ---")

# 5. 可视化结果
print("--- 正在生成布居数演化图 ---")
fig, ax = plt.subplots(figsize=(10, 6))

# 获取初始和最终的布居数
initial_populations = [exp[0] for exp in result.expect]
final_populations = [exp[-1] for exp in result.expect]

print(f"初始状态 |0> 布居: {initial_populations[0]:.4f}")
print(f"最终状态 |0> 布居: {final_populations[0]:.4f}")
print(f"最终状态 |1> 布居: {final_populations[1]:.4f}")
print(f"最终状态 |2> 布居 (泄漏): {final_populations[2]:.4f}")


ax.plot(t_list, result.expect[0], label=r'$|0\rangle$ Population')
ax.plot(t_list, result.expect[1], label=r'$|1\rangle$ Population')
ax.plot(t_list, result.expect[2], label=r'$|2\rangle$ Population (Leakage)')

ax.set_xlabel("Time (ns)", fontsize=14)
ax.set_ylabel("Population", fontsize=14)
ax.set_title(r"Population Dynamics: $|0\rangle \rightarrow |1\rangle$ with DRAG $\pi$-Pulse", fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)
plt.show()

