# -*- coding: utf-8 -*-
"""
simulator.py - 核心动力学仿真器模块
"""
import qutip
import numpy as np
from qutip import Qobj, mesolve, Options
from typing import Dict, List, Any
import matplotlib.pyplot as plt


from .hamiltonian import HamiltonianModel
from .pulses import ControlSequence, MicrowaveSequence, CompositeSequence

class Simulator:
    """
    一个通用的量子动力学仿真引擎。
    它接收一个物理模型和一个信号字典，并动态构建哈密顿量进行演化。
    """
    def __init__(self, model: HamiltonianModel, options: dict | None = None):
        """
        初始化仿真器。

        Args:
            model (HamiltonianModel): 要仿真的物理系统模型。
            options (dict, optional): 传递给 qutip.mesolve 的求解器选项。
        """
        self.model = model
        if options is None:
            # 使用默认的求解器选项，按照严格要求来
            self.options = Options(nsteps=2000000, 
                atol=1e-12,      
                rtol=1e-10,      
                store_states=True,
                max_step=0.1    
            )
        else:
            self.options = Options(**options)

    def run(self, 
            psi0: Qobj, 
            t_list: np.ndarray,
            sequences: Dict[str, ControlSequence],
            use_rwa: bool = True,
            show_waveforms: bool = False,
            e_ops: List[Qobj] | None = None) -> qutip.Result:
        """
        运行动力学仿真。

        Args:
            psi0 (Qobj): 系统的初始状态。
            t_list (np.ndarray): 仿真的时间点列表。
            sequences (Dict[str, ControlSequence]): 一个信号字典，
                键为控制通道名，值为对应的 ControlSequence 实例。
            use_rwa (bool, optional): 是否使用旋转波近似。默认为 True。
            e_ops (List[Qobj], optional): 需要计算期望值的算符列表。默认为 None。
        """
        
        control_ops = self.model.get_control_hamiltonian(use_rwa)
        hamiltonian = [self.model.get_drift_hamiltonian(use_rwa)]

        waveforms_to_plot  = {}

        # --- 动态构建时变哈密顿量 ---
        for channel_name, seq_object in sequences.items():
            if channel_name not in control_ops:
                print(f"警告: 序列字典提供了 '{channel_name}' 通道, 但模型中未定义，将被忽略。")
                continue
            
            op_dict = control_ops[channel_name]
            channel_type = op_dict.get('type')

            if channel_type == 'XY':
                carrier_freq = 0
                # 检查传入的是否为合法的XY驱动序列
                if isinstance(seq_object, MicrowaveSequence):
                    carrier_freq = seq_object.carrier_freq
                elif isinstance(seq_object, CompositeSequence):
                    # 如果是复合序列，检查内部是否全部为微波序列
                    is_all_mw = all(isinstance(s, MicrowaveSequence) for s in seq_object.sequences)
                    if not is_all_mw:
                        raise TypeError(f"通道 '{channel_name}' 的复合序列必须只包含 MicrowaveSequence 对象。但是得到: {[type(s) for s in seq_object.sequences]}")
                    # 假设所有子序列的载波频率相同，使用第一个的频率
                    if seq_object.sequences:
                        carrier_freq = seq_object.sequences[0].carrier_freq
                else:
                    raise TypeError(f"通道 '{channel_name}' 需要一个 MicrowaveSequence 或 CompositeSequence 对象。但是得到: {type(seq_object)}")
                
                envelopes = seq_object.get_envelopes(t_list)
                i_envelope = envelopes['I_drive']
                q_envelope = envelopes['Q_drive']
                
                if use_rwa:
                    hamiltonian.append([op_dict['I'], i_envelope])
                    hamiltonian.append([op_dict['Q'], q_envelope])
                    waveforms_to_plot[f'{channel_name} - I Envelope'] = i_envelope
                    waveforms_to_plot[f'{channel_name} - Q Envelope'] = q_envelope
                else:
                    composite_signal = i_envelope * np.cos(carrier_freq * t_list) + \
                                       q_envelope * np.sin(carrier_freq * t_list)
                    hamiltonian.append([op_dict['S'], composite_signal])
                    waveforms_to_plot[f'{channel_name} - Lab Frame Signal'] = composite_signal

            
            elif channel_type == 'Z':
                envelopes = seq_object.get_envelopes(t_list)
                z_envelope = envelopes['Z_drive']
                hamiltonian.append([op_dict['op'], z_envelope])
            
            else:
                raise TypeError(f"模型中定义了未知的通道类型: '{channel_type}'")
        
        if show_waveforms:
            plt.figure(figsize=(12, 6))
            for label, waveform in waveforms_to_plot.items():
                plt.plot(t_list, waveform, label=label)
            
            plt.title("Input Control Waveforms for Simulation")
            plt.xlabel("Time (ns)")
            plt.ylabel("Amplitude (rad/ns)")
            plt.grid(True)
            plt.legend()
            plt.show()

        solver_options = self.options
        return mesolve(hamiltonian, psi0, t_list, [], e_ops, options=solver_options)