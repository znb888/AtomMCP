# -*- coding: utf-8 -*-
"""
rabi_anime.py - 演示重复脉冲序列下的拉比振荡

本程序通过以下步骤演示了如何使用 quantum_sim 库来仿真一个重复的脉冲序列，
并观察系统在布洛赫球面上的演化。

1.  定义一个 pi/2 脉冲 (使用 CosineEnvelope)。
2.  定义一个短暂的空闲等待时间 (amp=0 的包络)。
3.  使用 '+' 操作符将 pi/2 脉冲和空闲等待拼接成一个基本单元。
4.  使用 .repeat() 方法将这个基本单元重复多次，形成一个长序列。
5.  设置仿真器，初始状态为 |0>。
6.  在仿真过程中计算 sigma_x, sigma_y, sigma_z 的期望值。
7.  使用 matplotlib 绘制 <sigma_x>, <sigma_y>, <sigma_z> 随时间演化的图像。
"""

from multiprocessing import Process
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from qutip import Qobj, basis, sigmax, sigmay, sigmaz, mesolve, Bloch
from qutip import expect, num
from tqdm import tqdm

# 导入我们自己库中的所有组件
# 注意：请确保 quantum_sim 包已正确安装或在 Python 路径中
try:
    from quantum_sim import (
        DuffingOscillatorModel,
        Simulator,
        CosineEnvelope,
        MicrowaveSequence
    )
except ImportError:
    # 如果直接在 examples 目录下运行，可能需要调整路径
    import sys
    sys.path.append('../src')
    from quantum_sim import (
        DuffingOscillatorModel,
        Simulator,
        CosineEnvelope,
        MicrowaveSequence
    )

# --- 1. 物理系统参数 ---
D_LEVELS = 3  # 对于简单的拉比振荡，2能级系统足够
OMEGA_Q_GHZ = 5.3
ANH_GHZ = -0.212 # 在2能级系统中此参数无影响，但为了模型完整性而保留

# --- 2. 脉冲和序列参数 ---
PI_HALF_DURATION = 10  # ns, 单个 pi/2 脉冲的时长
# 根据 CosineEnvelope 的定义 (A/2)*(1-cos), 其面积为 A*T/2。
# 在RWA下，旋转角 theta = 脉冲面积 / 2。
# 对于 pi/2 脉冲, theta = pi/2, 所以 A*T/4 = pi/2, A = 2*pi/T
PI_HALF_AMP = np.pi / PI_HALF_DURATION
IDLE_DURATION = 4     # ns, 每个脉冲后的空闲等待时长
REPEAT_NUM = 4        # 重复次数

# 假设驱动频率与比特频率共振
carrier_freq = OMEGA_Q_GHZ  # GHz

# --- 3. 构建脉冲序列包络 ---
# a) 创建 pi/2 脉冲包络
pi_half_pulse_env = CosineEnvelope(duration=PI_HALF_DURATION, amp=PI_HALF_AMP)

# b) 创建空闲等待时间的包络
idle_env = CosineEnvelope(duration=IDLE_DURATION, amp=0.0)

# --- 3.1 使用先叠加envelope后构造sequence的方法 ---
# # c) 使用 '+' 拼接成基本单元
# base_unit_env = pi_half_pulse_env + idle_env

# # d) 重复基本单元形成完整序列
# full_sequence_env = base_unit_env.repeat(REPEAT_NUM)

# # e) 基于最终的包络创建微波序列
# mw_sequence = MicrowaveSequence(envelope=full_sequence_env, carrier_freq=carrier_freq * 2 * np.pi, phi=0)

# print(f"基本单元时长: {base_unit_env.duration} ns")
# print(f"总序列时长: {full_sequence_env.duration} ns")

# --- 3.2 先构造sequence后叠加的方法 (等价) ---

# c) 构造基本sequence
pi_half_pulse_seq = MicrowaveSequence(envelope=pi_half_pulse_env, carrier_freq=OMEGA_Q_GHZ * 2 * np.pi, phi=0)
idle_seq = MicrowaveSequence(envelope=idle_env, carrier_freq=OMEGA_Q_GHZ * 2 * np.pi, phi=0)
base_unit_seq = pi_half_pulse_seq + idle_seq

mw_sequence = base_unit_seq.repeat(REPEAT_NUM)

# --- 4. 设置仿真 ---
# a) 创建物理模型和仿真器
model = DuffingOscillatorModel(
    d=D_LEVELS,
    omega_q=OMEGA_Q_GHZ * 2 * np.pi,
    anh=ANH_GHZ * 2 * np.pi
)
sim = Simulator(model=model)

# b) 定义初始状态 |0>
psi0 = basis(D_LEVELS, 0)
# psi0 = (basis(D_LEVELS, 0) + basis(D_LEVELS, 1)).unit()  # 初始态为 |+> 态
# psi0 = (basis(D_LEVELS, 0) + 0.1 * basis(D_LEVELS, 1)).unit()  # 初始态为 非对称叠加态

# c) 定义需要计算期望值的算符
sigma_x = basis(D_LEVELS,0)*basis(D_LEVELS,1).dag() + basis(D_LEVELS,1)*basis(D_LEVELS,0).dag()
sigma_y = -1j*basis(D_LEVELS,0)*basis(D_LEVELS,1).dag() + 1j*basis(D_LEVELS,1)*basis(D_LEVELS,0).dag()
sigma_z = basis(D_LEVELS,0)*basis(D_LEVELS,0).dag() - basis(D_LEVELS,1)*basis(D_LEVELS,1).dag()
e_ops = [sigma_x, sigma_y, sigma_z]

# d) 定义仿真时间列表
total_duration = mw_sequence.duration
t_list = np.linspace(0, total_duration, int(total_duration * 5000)) # 增加时间点以获得平滑曲线


# --- 5. 运行仿真 ---
print("开始进行 RWA 动力学仿真...")
result_rwa = sim.run(
    psi0=psi0,
    t_list=t_list,
    sequences={'XY_drive': mw_sequence},
    use_rwa=True,
    show_waveforms=False,
    e_ops=e_ops
)
print("...RWA 仿真完成。")

print("开始进行 Non-RWA (Exact) 动力学仿真...")
result_non_rwa = sim.run(
    psi0=psi0,
    t_list=t_list,
    sequences={'XY_drive': mw_sequence},
    use_rwa=False,
    e_ops=e_ops
)
print("...Non-RWA 仿真完成。")

print("正在将 Non-RWA 结果转换到旋转坐标系...")
omega = carrier_freq * 2 * np.pi
t_points = np.array(result_non_rwa.times)

# 从结果中提取 lab frame 的期望值
sx_lab = result_non_rwa.expect[0]
sy_lab = result_non_rwa.expect[1]
sz_lab = result_non_rwa.expect[2]

# 计算旋转因子
cos_wt = np.cos(omega * t_points)
sin_wt = np.sin(omega * t_points)

# 应用旋转 (矢量化操作)
sx_data_rot = cos_wt * sx_lab - sin_wt * sy_lab
sy_data_rot = sin_wt * sx_lab + cos_wt * sy_lab
sz_data_rot = sz_lab # sz 不变

print("...转换完成。")


# --- 6. 结果可视化 ---
print("正在绘制结果...")
plt.ion() # Turn on interactive mode
fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
fig.suptitle(f"Comparison of RWA and Non-RWA Simulations (n={REPEAT_NUM})", fontsize=16)

# 绘制 RWA 结果
ax_rwa = axes[0]
ax_rwa.plot(t_list, result_rwa.expect[0], label='<σx>')
ax_rwa.plot(t_list, result_rwa.expect[1], label='<σy>')
ax_rwa.plot(t_list, result_rwa.expect[2], label='<σz>')
ax_rwa.set_title("RWA Simulation")
ax_rwa.set_ylabel("Expectation Value")
ax_rwa.grid(True)
ax_rwa.legend()
ax_rwa.set_ylim(-1.1, 1.1)

# 绘制 Non-RWA 结果
ax_non_rwa = axes[1]
ax_non_rwa.plot(t_list, result_non_rwa.expect[0], label='<σx>')
ax_non_rwa.plot(t_list, result_non_rwa.expect[1], label='<σy>')
ax_non_rwa.plot(t_list, result_non_rwa.expect[2], label='<σz>')
ax_non_rwa.set_title("Non-RWA (Exact) Simulation")
ax_non_rwa.set_xlabel("Time (ns)")
ax_non_rwa.set_ylabel("Expectation Value")
ax_non_rwa.grid(True)
ax_non_rwa.legend()
ax_non_rwa.set_ylim(-1.1, 1.1)

# 绘制 Non-RWA 转换回lab frame的结果
ax_rot_non_rwa = axes[2]
ax_rot_non_rwa.plot(t_list, sx_data_rot, label='<σx>')
ax_rot_non_rwa.plot(t_list, sy_data_rot, label='<σy>')
ax_rot_non_rwa.plot(t_list, sz_data_rot, label='<σz>')
ax_rot_non_rwa.set_title("Non-RWA (Exact) Simulation - Rotating Frame")
ax_rot_non_rwa.set_xlabel("Time (ns)")
ax_rot_non_rwa.set_ylabel("Expectation Value")
ax_rot_non_rwa.grid(True)
ax_rot_non_rwa.legend()
ax_rot_non_rwa.set_ylim(-1.1, 1.1)

# 在三个子图上都添加垂直线
for ax in axes:
    for i in range(REPEAT_NUM):
        pulse_end_time = i * (PI_HALF_DURATION + IDLE_DURATION) + PI_HALF_DURATION
        ax.axvline(x=pulse_end_time, color='gray', linestyle='--', alpha=0.7)

plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局以适应主标题
plt.show()
plt.pause(0.5)


# --- 7. 创建Bloch球动画窗口 ---
print("正在创建Bloch球动画...")

# 创建第二个窗口显示Bloch球动画
fig_bloch = plt.figure(figsize=(8, 8))
ax_bloch = fig_bloch.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.1)

# 创建Bloch球
b = Bloch(fig=fig_bloch, axes=ax_bloch)
b.make_sphere()

# 为动画准备数据
ANIMATION_FRAMES = 100
# 转换为numpy数组以确保正确的索引操作
# 用RWA结果进行动画
# sx_data = np.array(result_rwa.expect[0])
# sy_data = np.array(result_rwa.expect[1])
# sz_data = np.array(result_rwa.expect[2])

# 用非RWA结果进行动画
# sx_data = np.array(result_non_rwa.expect[0])
# sy_data = np.array(result_non_rwa.expect[1])
# sz_data = np.array(result_non_rwa.expect[2])

# 用非RWA转换到旋转坐标系的结果进行动画
sx_data = np.array(sx_data_rot)
sy_data = np.array(sy_data_rot)
sz_data = np.array(sz_data_rot)




total_sim_points = len(sx_data)
sample_indices = np.linspace(0, total_sim_points - 1, ANIMATION_FRAMES, dtype=int)
sx_anime = sx_data[sample_indices]
sy_anime = sy_data[sample_indices]
sz_anime = sz_data[sample_indices]

def animate_bloch(frame):
    """动画函数：更新Bloch球上的态矢量"""
    b.clear()
    
    # 添加轨迹（从开始到当前帧）
    if frame > 0:
        b.add_points([sx_anime[:frame+1], sy_anime[:frame+1], sz_anime[:frame+1]], alpha=0.3)
    
    # 添加当前态矢量
    current_vector = [sx_anime[frame], sy_anime[frame], sz_anime[frame]]
    b.add_vectors(current_vector, colors=['red'])
    
    # 设置标题显示当前时间
    current_time = t_list[sample_indices[frame]]
    ax_bloch.set_title(f'Bloch Sphere Evolution\nTime: {current_time:.2f} ns', fontsize=14)
    
    b.render()
    return ax_bloch.get_children()

def init_bloch():
    """初始化函数"""
    b.clear()
    b.make_sphere()
    return ax_bloch.get_children()

# 创建动画
print("开始Bloch球动画...")
ani = FuncAnimation(fig_bloch, animate_bloch, frames=ANIMATION_FRAMES, 
                   init_func=init_bloch, interval=100, blit=False, repeat=True)

plt.show()

# --- 8. 保存动画到文件 ---
def save_animation():
    """保存动画的函数"""
    print("正在保存动画到文件...")
    try:
        ani.save("temp/rabi_animation.mp4", writer='ffmpeg', fps=20, dpi=100)
        print("动画已保存到 temp/rabi_animation.mp4")
    except Exception as e:
        print(f"保存动画时出错: {e}")
        try:
            # 尝试使用pillow保存为gif
            ani.save("temp/rabi_animation.gif", writer='pillow', fps=10)
            print("动画已保存为 temp/rabi_animation.gif")
        except Exception as e2:
            print(f"保存gif时也出错: {e2}")

if __name__ == '__main__':
    # 在主线程中保存动画
    save_animation()
print("脚本执行完毕。所有窗口将保持打开状态，直到手动关闭。")
plt.ioff()
plt.show()
