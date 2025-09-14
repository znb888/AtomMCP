# -*- coding: utf-8 -*-
"""
pulses.py - 模块化量子控制脉冲定义

该模块定义了构建时变控制序列所需的核心类。
其设计哲学遵循“组合优于继承”和“分离关注点”的原则。

核心类：
1.  Envelope: 定义脉冲包络的数学形状及其解析导数的抽象基类。
    - GaussianEnvelope: 一个具体的包络实现。
2.  ControlSequence: 定义一个完整的控制序列的抽象基类，它必须能够
   为RWA和非RWA（实验室坐标系）两种仿真模式提供信号。
    - EnvelopeModulatedSequence: 由一个Envelope和一个载波频率构成的控制序列。
3.  DRAGCorrector: 一个修正器类，它接受一个兼容的ControlSequence，
   并应用DRAG修正，返回一个新的、经过修正的ControlSequence。
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict
from typing_extensions import Self
from typing import List
from sympy import sympify, lambdify, diff, Symbol

# ==============================================================================
# 1. 包络定义层 (Envelope Definition Layer)
# ==============================================================================

class Envelope(ABC):
    """
    所有脉冲包络形状的抽象基类（契约）。
    它只关心包络的数学形式，不关心载波频率或任何修正。
    """
    def __init__(self, duration: float, **params):
        self.duration = duration
        self.params = params

    @abstractmethod
    def value(self, t: np.ndarray) -> np.ndarray:
        """
        计算在时间点 t 上的包络值。

        Args:
            t (np.ndarray): 时间数组。

        Returns:
            np.ndarray: 对应时间点的包络值数组。
        """
        pass

    @abstractmethod
    def derivative(self, t: np.ndarray) -> np.ndarray:
        """
        计算在时间点 t 上的包络的 *解析* 导数。

        Args:
            t (np.ndarray): 时间数组。

        Returns:
            np.ndarray: 对应时间点的包络导数值数组。
        """
        pass

    def repeat(self, repeat_num: int) -> Self:
        """
        创建一个新的 Envelope 实例，该实例将当前包络的形状重复指定的次数。

        Args:
            repeat_num (int): 波形重复的次数，必须为正整数。

        Returns:
            Envelope: 一个总时长为 repeat_num * self.duration 的新 Envelope 实例。
        """
        if not isinstance(repeat_num, int) or repeat_num < 1:
            raise ValueError("repeat_num 必须是一个正整数。")
        
        # 如果只重复1次，直接返回自身的副本以提高效率 (或者直接返回self)
        if repeat_num == 1:
            return self

        # 返回一个包含重复逻辑的特殊 Envelope 实例
        return _RepeatedEnvelope(original_envelope=self, repeat_num=repeat_num)
    
    def __add__(self, other: "Envelope") -> "CompositeEnvelope":
        """
        通过 '+' 操作符将当前包络与另一个包络串联起来。

        Args:
            other (Envelope): 要串联在当前包络后面的另一个包络对象。

        Returns:
            CompositeEnvelope: 一个包含了两个包络的新的复合包络。
        """
        if not isinstance(other, Envelope):
            return NotImplemented

        # 智能地处理操作数，无论它们是单个包络还是复合包络
        left_envelopes = self.envelopes if isinstance(self, CompositeEnvelope) else [self]
        right_envelopes = other.envelopes if isinstance(other, CompositeEnvelope) else [other]

        # 返回一个由所有子包络组成的、扁平化的新复合包络
        return CompositeEnvelope(left_envelopes + right_envelopes)

class _RepeatedEnvelope(Envelope):
    """
    一个私有的辅助类，用于封装一个被周期性重复的包络。
    它对用户是透明的，通过 Envelope.repeat() 方法创建。
    """
    def __init__(self, original_envelope: Envelope, repeat_num: int):
        self.original_envelope = original_envelope
        self.repeat_num = repeat_num
        self.segment_duration = original_envelope.duration

        # 调用父类的构造函数，设置新的总时长和继承原始参数
        super().__init__(
            duration=self.segment_duration * repeat_num,
            **original_envelope.params
        )

    def value(self, t: np.ndarray) -> np.ndarray:
        """
        计算重复包络在时间 t 的值。
        通过模运算将全局时间 t 映射到单个分段的局部时间 t_local。
        """
        # 创建一个布尔掩码，只在脉冲总持续时间内计算
        mask = (t >= 0) & (t <= self.duration)
        result = np.zeros_like(t, dtype=float)

        # 仅对掩码内的部分进行计算
        t_masked = t[mask]
        # 使用模运算找到在原始包络内的等效时间点
        t_local = t_masked % self.segment_duration
        
        # 调用原始包络的 value 方法
        result[mask] = self.original_envelope.value(t_local)
        
        return result

    def derivative(self, t: np.ndarray) -> np.ndarray:
        """
        计算重复包络在时间 t 的导数。
        逻辑与 value 方法相同。
        """
        mask = (t >= 0) & (t <= self.duration)
        result = np.zeros_like(t, dtype=float)
        
        t_masked = t[mask]
        t_local = t_masked % self.segment_duration
        
        # 调用原始包络的 derivative 方法
        result[mask] = self.original_envelope.derivative(t_local)
        
        return result

class CompositeEnvelope(Envelope):
    """
    一个复合包络，将多个包络对象按顺序拼接成一个单一的包络。
    """
    def __init__(self, envelopes: List[Envelope]):
        """
        Args:
            envelopes (List[Envelope]): 一个包含多个 Envelope 对象的列表。
            name (str): 复合包络的名称。
        """
        if not envelopes:
            raise ValueError("envelopes 列表不能为空。")

        self.envelopes = envelopes

        # 计算并存储每个包络的起止时间点
        self.durations = [env.duration for env in self.envelopes]
        self.end_times = np.cumsum(self.durations)
        self.start_times = self.end_times - self.durations
        
        total_duration = self.end_times[-1]

        # 调用父类的构造函数
        super().__init__(duration=total_duration)

    def value(self, t: np.ndarray) -> np.ndarray:
        """计算复合包络在时间 t 的值。"""
        result = np.zeros_like(t, dtype=float)

        # 遍历每个子包络
        for i, env in enumerate(self.envelopes):
            start_time = self.start_times[i]
            end_time = self.end_times[i]
            
            # 找到在当前子包络时间区间内的时间点
            mask = (t >= start_time) & (t < end_time)
            
            if np.any(mask):
                # 将全局时间 t 转换为子包络的局部时间 t_local
                t_local = t[mask] - start_time
                result[mask] = env.value(t_local)
        
        return result

    def derivative(self, t: np.ndarray) -> np.ndarray:
        """计算复合包络在时间 t 的导数。"""
        result = np.zeros_like(t, dtype=float)

        # 逻辑与 value 方法完全相同
        for i, env in enumerate(self.envelopes):
            start_time = self.start_times[i]
            end_time = self.end_times[i]
            
            mask = (t >= start_time) & (t < end_time)
            
            if np.any(mask):
                t_local = t[mask] - start_time
                result[mask] = env.derivative(t_local)
        
        return result


class CosineEnvelope(Envelope):
    """
    升余弦包络的具体实现。
    包络形式: (A/2) * (1 - cos(2*pi*t/duration))
    """
    def __init__(self, duration: float, amp: float):
        """
        Args:
            duration (float): 脉冲总时长。脉冲在 [0, duration] 区间外为0。
            amp (float): 脉冲峰值振幅。
        """
        super().__init__(duration=duration, amp=amp)

    def value(self, t: np.ndarray) -> np.ndarray:
        """计算升余弦包络值。"""
        amp = self.params['amp']
        
        mask = (t >= 0) & (t <= self.duration)
        result = np.zeros_like(t, dtype=float)
        
        t_masked = t[mask]
        result[mask] = (amp / 2) * (1 - np.cos(2 * np.pi * t_masked / self.duration))
        
        return result

    def derivative(self, t: np.ndarray) -> np.ndarray:
        """计算升余弦包络的解析导数。"""
        amp = self.params['amp']
        
        mask = (t >= 0) & (t <= self.duration)
        result = np.zeros_like(t, dtype=float)
        
        t_masked = t[mask]
        result[mask] = (amp * np.pi / self.duration) * np.sin(2 * np.pi * t_masked / self.duration)
        
        return result

class GaussianEnvelope(Envelope):
    """
    高斯包络的具体实现。
    包络形式: amp * exp[ - (t - t0)^2 / (2 * sigma^2) ]
    """
    def __init__(self, duration: float, amp: float, sigma: float):
        """
        Args:
            duration (float): 脉冲总时长。脉冲在 [0, duration] 区间外为0。
            amp (float): 脉冲峰值振幅。
            sigma (float): 高斯函数的标准差。
        """
        self.t0 = duration / 2
        super().__init__(duration=duration, amp=amp, sigma=sigma)

    def value(self, t: np.ndarray) -> np.ndarray:
        """计算高斯包络值。"""
        amp = self.params['amp']
        sigma = self.params['sigma']
        
        # 创建一个布尔掩码，只在脉冲持续时间内计算
        mask = (t >= 0) & (t <= self.duration)
        result = np.zeros_like(t, dtype=float)
        
        # 仅对掩码内的部分进行计算
        t_masked = t[mask]
        result[mask] = amp * np.exp(-((t_masked - self.t0)**2) / (2 * sigma**2))
        
        return result

    def derivative(self, t: np.ndarray) -> np.ndarray:
        """计算高斯包络的解析导数。"""
        amp = self.params['amp']
        sigma = self.params['sigma']
        
        mask = (t >= 0) & (t <= self.duration)
        result = np.zeros_like(t, dtype=float)
        
        t_masked = t[mask]
        exp_term = np.exp(-((t_masked - self.t0)**2) / (2 * sigma**2))
        pre_factor = - (t_masked - self.t0) / (sigma**2)
        result[mask] = amp * pre_factor * exp_term
        
        return result

class ConstantEnvelope(Envelope):
    """一个值为常数的矩形包络。"""
    def __init__(self, duration: float, amp: float):
        super().__init__(duration=duration, amp=amp)

    def value(self, t: np.ndarray) -> np.ndarray:
        mask = (t >= 0) & (t <= self.duration)
        result = np.zeros_like(t, dtype=float)
        result[mask] = self.params['amp']
        return result

    def derivative(self, t: np.ndarray) -> np.ndarray:
        # 导数处处为0
        return np.zeros_like(t, dtype=float)


class ArbitraryFunctionEnvelope(Envelope):
    """
    一个由任意数学函数字符串定义的通用包络。
    由 Sympy 提供符号计算和自动求导支持。
    """
    def __init__(self, duration: float, formula_str: str, params: dict):
        """
        Args:
            duration (float): 脉冲总时长。
            formula_str (str): 描述包络的数学公式字符串。
                必须包含变量 't'，可以包含 params 中的任意参数。
                例如: "amp * exp(-(t - t0)**2 / (2 * sigma**2))"
            params (dict): 公式中使用的参数及其值的字典。
                例如: {"amp": 0.5, "t0": 15, "sigma": 5}
        """
        self.formula_str = formula_str
        self._params = params
        
        # --- Sympy 符号运算核心 ---
        # 1. 定义符号变量
        t_sym = Symbol('t')
        param_syms = {k: Symbol(k) for k in params.keys()}
        
        # 2. 将字符串公式转换为符号表达式
        self.expr = sympify(formula_str, locals=param_syms)
        
        # 3. 自动计算符号导数
        self.derivative_expr = diff(self.expr, t_sym)
        
        # 4. 将符号表达式编译为快速的、可数值计算的 aaccc 函数
        #    这避免了在每次调用时都进行缓慢的符号求值
        all_syms = [t_sym] + list(param_syms.values())
        self._value_func = lambdify(all_syms, self.expr, 'numpy')
        self._derivative_func = lambdify(all_syms, self.derivative_expr, 'numpy')
        
        super().__init__(duration=duration, **params)

    def value(self, t: np.ndarray) -> np.ndarray:
        mask = (t >= 0) & (t <= self.duration)
        result = np.zeros_like(t, dtype=float)
        
        if np.any(mask):
            param_values = list(self._params.values())
            result[mask] = self._value_func(t[mask], *param_values)
        return result

    def derivative(self, t: np.ndarray) -> np.ndarray:
        mask = (t >= 0) & (t <= self.duration)
        result = np.zeros_like(t, dtype=float)

        if np.any(mask):
            param_values = list(self._params.values())
            result[mask] = self._derivative_func(t[mask], *param_values)
        return result

# ==============================================================================
# 2. 控制序列定义层 (Control Sequence Definition Layer)
# ==============================================================================

class ControlSequence(ABC):
    """控制序列的抽象基类。其唯一职责是生成基础时变包络。"""
    def __init__(self, duration: float):
        self.duration = duration

    @abstractmethod
    def get_envelopes(self, t: np.ndarray) -> Dict[str, np.ndarray]:
        """
        返回一个包含一个或多个“通道-包络”键值对的字典。
        这些包络是基础的、慢变的系数，不包含载波信息。
        """
        pass

    def __add__(self, other: Self) -> "CompositeSequence":
        """通过 '+' 操作符将当前序列与另一个序列串联起来。"""
        if not isinstance(other, ControlSequence):
            return NotImplemented

        # 展平左右两个操作数，确保最终的序列列表是一维的
        left_sequences = self.sequences if isinstance(self, CompositeSequence) else [self]
        right_sequences = other.sequences if isinstance(other, CompositeSequence) else [other]

        return CompositeSequence(left_sequences + right_sequences)

    def repeat(self, repeat_num: int) -> Self:
        """将当前序列重复指定的次数。"""
        if not isinstance(repeat_num, int) or repeat_num < 1:
            raise ValueError("repeat_num 必须是一个正整数。")
        if repeat_num == 1:
            return self
        
        # 确定要重复的基础序列列表 (展平)
        base_sequences = self.sequences if isinstance(self, CompositeSequence) else [self]
        
        # 将基础序列列表重复N次，结果仍然是扁平的
        final_flat_list = base_sequences * repeat_num
        
        return CompositeSequence(final_flat_list)


class CompositeSequence(ControlSequence):
    """
    一个复合控制序列，将多个控制序列对象按顺序拼接成一个单一的序列。
    """
    def __init__(self, sequences: List[ControlSequence]):
        if not sequences:
            raise ValueError("sequences 列表不能为空。")

        self.sequences = sequences
        self.durations = [seq.duration for seq in self.sequences]
        self.end_times = np.cumsum(self.durations)
        self.start_times = self.end_times - self.durations
        
        total_duration = self.end_times[-1]
        super().__init__(duration=total_duration)

    def get_envelopes(self, t: np.ndarray) -> Dict[str, np.ndarray]:
        """
        计算复合序列在时间 t 的包络值。
        它会遍历所有子序列，并将它们各自的包络在正确的时间段内拼接起来。
        """
        # 初始化一个空的包络字典
        final_envelopes = {}

        # 遍历每个子序列
        for i, seq in enumerate(self.sequences):
            start_time = self.start_times[i]
            end_time = self.end_times[i]
            
            # 找到在当前子序列时间区间内的时间点
            mask = (t >= start_time) & (t < end_time)
            
            if np.any(mask):
                # 将全局时间 t 转换为子序列的局部时间 t_local
                t_local = t[mask] - start_time
                
                # 获取子序列的包络
                sub_envelopes = seq.get_envelopes(t_local)
                
                # 将子包络的值合并到最终结果中
                for channel, envelope_values in sub_envelopes.items():
                    if channel not in final_envelopes:
                        # 如果通道首次出现，用0初始化整个时间数组
                        final_envelopes[channel] = np.zeros_like(t, dtype=float)
                    
                    # 将子包络的计算结果填充到对应的时间段
                    final_envelopes[channel][mask] = envelope_values

        return final_envelopes


class MicrowaveSequence(ControlSequence):
    """
    一个微波驱动序列，提供 I 和 Q 两个通道的包络。

    此序列的实现基于一个核心假设：物理信号最终由 I/Q 包络和正交载波合成。
    一个带相位的目标信号 Ω(t) * cos(ωt + phi) 可以通过三角函数展开：
    Ω(t) * cos(ωt + phi) = Ω(t)cos(phi)cos(ωt) - Ω(t)sin(phi)sin(ωt)

    如果硬件合成信号的形式为 `I(t)cos(ωt) + Q(t)sin(ωt)`，
    那么通过比较，我们可以得到 I/Q 包络的定义：
    I(t) = Ω(t)cos(phi)
    Q(t) = -Ω(t)sin(phi)

    本类正是基于此定义来计算 I 和 Q 包络。
    """
    def __init__(self, envelope: Envelope, carrier_freq: float, phi: float = 0):
        self.envelope = envelope
        self.carrier_freq = carrier_freq
        self.phi = phi
        self.control_type = 'XY'
        super().__init__(duration=envelope.duration)

    def get_envelopes(self, t: np.ndarray) -> Dict[str, np.ndarray]:
        """
        根据给定的相位 phi 计算 I 和 Q 包络。
        """
        envelope = self.envelope.value(t)
        i_envelope = envelope * np.cos(self.phi)
        q_envelope = -envelope * np.sin(self.phi)
        return {'I_drive': i_envelope, 'Q_drive': q_envelope}


class ZSequence(ControlSequence):
    """一个 Z (flux) 驱动序列，提供 Z 一个通道的包络。"""
    def __init__(self, envelope: Envelope):
        self.envelope = envelope
        self.control_type = 'Z'
        super().__init__(duration=envelope.duration)
        
    def get_envelopes(self, t: np.ndarray) -> Dict[str, np.ndarray]:
        z_envelope = self.envelope.value(t)
        return {'Z_drive': z_envelope}



# ==============================================================================
# 3. 修正器定义层 (Corrector Definition Layer)
# ==============================================================================

class DRAGCorrector:
    """
    DRAG 修正器，作用于 MicrowaveSequence，返回一个修正后的实例。

    DRAG (Derivative Removal by Adiabatic Gate) 修正旨在抑制由于邻近能级
    （如 qutrit 的 |2> 能级）引起的相位误差以及 leakage 问题。

    其物理原理是在原始驱动场上，额外施加一个正交的、且幅度与原始包络导数
    成正比的修正场。

    如果原始驱动场为：
    Ω(t) * cos(ωt + phi)

    那么修正后的驱动场为：
    Ω(t) * cos(ωt + phi) - (alpha/anh) * (dΩ/dt) * sin(ωt + phi)

    将上式按 cos(ωt) 和 sin(ωt) 展开，并对比硬件产生的信号形式
    I(t)cos(ωt) + Q(t)sin(ωt)，即可得到修正后 I(t) 和 Q(t) 的表达式，
    这正是本类中 `get_envelopes` 方法的实现。
    """
    def __init__(self, anh: float, alpha: float):
        self.anh = anh
        self.alpha = alpha

    def apply(self, base_sequence: ControlSequence) -> ControlSequence:
        """
        智能地对一个控制序列应用DRAG修正。

        - 如果是 CompositeSequence，则递归地对其每个子序列应用修正。
        - 如果是 MicrowaveSequence，则应用DRAG修正。
        - 如果是其他类型的序列（如 ZSequence），则原样返回。
        """
        # Case 1: 复合序列，递归应用
        if isinstance(base_sequence, CompositeSequence):
            corrected_sub_sequences = [self.apply(sub_seq) for sub_seq in base_sequence.sequences]
            return CompositeSequence(corrected_sub_sequences)

        # Case 2: 微波序列，应用修正
        if isinstance(base_sequence, MicrowaveSequence):
            class CorrectedMicrowaveSequence(MicrowaveSequence):
                def __init__(self, original_sequence, corrector):
                    super().__init__(original_sequence.envelope, original_sequence.carrier_freq, original_sequence.phi)
                    self.corrector = corrector

                def get_envelopes(self, t: np.ndarray) -> Dict[str, np.ndarray]:
                    i_envelope = self.envelope.value(t) * np.cos(self.phi) - self.corrector.alpha * self.envelope.derivative(t) / self.corrector.anh * np.sin(self.phi)
                    q_envelope = - self.envelope.value(t) * np.sin(self.phi) - self.corrector.alpha * self.envelope.derivative(t) / self.corrector.anh * np.cos(self.phi)
                    return {'I_drive': i_envelope, 'Q_drive': q_envelope}
            
            return CorrectedMicrowaveSequence(base_sequence, self)

        # Case 3: 其他序列（如ZSequence），直接返回
        return base_sequence

