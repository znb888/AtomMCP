# -*- coding: utf-8 -*-
"""
hamiltonian.py - 物理系统哈密顿量的定义模块
"""

import os
import json
import numpy as np
import qutip as qt
from qutip import Qobj, destroy, num, qeye
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from collections import defaultdict
import matplotlib.pyplot as plt

# 设置 matplotlib 使用 LaTeX 渲染
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

#--- 1. 定义更通用、更精简的抽象基类 (New, Leaner ABC) ---
class HamiltonianModel(ABC):
    """
    所有物理模型哈密顿量的抽象基类。
    契约只规定了所有模型都必须具备的最小接口。
    """
    @property
    @abstractmethod
    def d(self) -> int:
        """能级数"""
        pass

    @abstractmethod
    def get_control_hamiltonian(self, use_rwa: bool) -> Dict[str, Any]:
        """返回一个定义了所需控制通道的字典（“接线表”）。"""
        pass
    
    @abstractmethod
    def get_drift_hamiltonian(self, use_rwa: bool) -> Qobj:
        """根据仿真模式返回对应的自由哈密顿量。"""
        pass


class ArbitraryHamiltonianModel(HamiltonianModel):
    """
    Hamiltonian Factory Class
    
    A specialized utility class designed for extracting and constructing Hamiltonian objects 
    from various data formats and configurations. Provides multiple static and class methods 
    to parse Hamiltonians from configuration files(json) or JSON strings, dictionaries, and other formats.
    
    In quantum mechanics and quantum computing, the Hamiltonian represents the total energy 
    operator of a system. This class simplifies the process of building Hamiltonian objects 
    from diverse data sources.
    """
    _operator_map = {
        "sigmax": qt.sigmax,
        "sigmay": qt.sigmay,
        "sigmaz": qt.sigmaz,
        "sigmap": qt.sigmap, # raising operator
        "sigmam": qt.sigmam, # lowering operator
        "create": qt.create,
        "destroy": qt.destroy,
        "num": qt.num,
        "identity": qt.identity,
    }
    _op_latexmap = {
        "sigmax": r"\sigma_x",
        "sigmay": r"\sigma_y",
        "sigmaz": r"\sigma_z",
        "sigmap": r"\sigma_+",
        "sigmam": r"\sigma_-",
        "create": r"a^{\dagger}",
        "destroy": r"a",
        "num": r"a^{\dagger} a",
        "identity": r"I",
    }
    
    def __init__(
            self, 
            system_dims: List[int],
            control_hamiltonian_terms: List[Dict[str, Any]],
            drift_hamiltonian_terms: List[Dict[str, Any]],
        ):
        self.system_dims = system_dims
        self.control_hamiltonian_terms = control_hamiltonian_terms
        self.drift_hamiltonian_terms = drift_hamiltonian_terms

    @property
    def d(self) -> int:
        return np.multiply(self.system_dims)

    @classmethod
    def get_operator(cls, name: List[str], op_size: int, mode: str = 'qutip') -> Qobj:
        """
        Retrieves a qutip operator object from its string name, checking for dimension compatibility.
        """
        if mode == 'latex':
            func = [cls._op_latexmap.get(n.lower()) for n in name]
            if None in func:
                idx = func.index(None)
                raise ValueError(f"Unknown operator name: '{name[idx]}'")
            return " ".join(func)
        else:
            func = [cls._operator_map.get(n.lower()) for n in name]
            if None in func:
                idx = func.index(None)
                raise ValueError(f"Unknown operator name: '{name[idx]}'")
        
        # 定义固定为2维的算符（泡利算符）
        qubit_only_ops = {"sigmax", "sigmay", "sigmaz", "sigmap", "sigmam"}
        
        # 检查维度兼容性
        if any(n.lower() in qubit_only_ops for n in name):
            if op_size != 2:
                raise ValueError(f"Operator '{name}' is only defined for 2-dimensional systems (qubits), but subsystem has dimension {op_size}.")
            op1 = func[0]()
            for f in func[1:]:
                op1 = op1 * f()
            return op1
        else:
            # 对于其他算符（如 create, destroy, identity），传入子系统维度
            op1 = func[0](op_size)
            for f in func[1:]:
                op1 = op1 * f(op_size)
            
            return op1
    
    def get_control_hamiltonian(self, mode: str = 'qutip') -> Dict[str, Qobj]:
        """
        构建控制哈密顿量格式

        Returns:
            Dict[str, Qobj]: 控制哈密顿量
        """
        control_hamiltonian = defaultdict(dict)
        
        for term in self.control_hamiltonian_terms:
            drive_name = term.get("drive_name")
            if drive_name is None:
                raise ValueError("Each control Hamiltonian term must have a 'drive_name' field.")
            channels = term.get("channels")
            if channels is None:
                raise ValueError("Each control Hamiltonian term must have a 'channels' field.")
            
            for channel, operas in channels.items():
                if mode == 'latex':
                    hamil = r""
                elif mode == 'qutip':
                    hamil = qt.qzero(self.system_dims)
                for opera in operas:
                    coeff_val = opera.get("coefficient", 1.0)
                    if isinstance(coeff_val, dict):
                        coefficient = complex(coeff_val.get("real", 0), coeff_val.get("imag", 0))
                    else:
                        coefficient = float(coeff_val)
                    op_list = []
                    op_def = opera.get("operators", {})
                    for idx in range(len(self.system_dims)):
                        if str(idx) in op_def.keys():
                            op_list.append(self.get_operator(op_def[str(idx)], self.system_dims[idx], mode))
                        else:
                            if mode == 'qutip':
                                op_list.append(qt.identity(self.system_dims[idx]))
                    if mode == 'qutip':
                        hamil += coefficient * qt.tensor(op_list)
                    elif mode == 'latex':
                        hamil += f" + {coefficient:.2f} " + " ".join(op_list)
                if mode == 'qutip':
                    control_hamiltonian[drive_name][channel] = hamil
                elif mode == 'latex':
                    control_hamiltonian[drive_name][channel] = hamil

        return control_hamiltonian
    
    def get_drift_hamiltonian(self, mode: str = 'qutip') -> Qobj:
        """
        构建自由哈密顿量

        Returns:
            Qobj: 自由哈密顿量
        """
        if mode == 'qutip':
            drift_hamiltonian = qt.qzero(self.system_dims)
        elif mode == 'latex':
            drift_hamiltonian = r"H_0 = "
        
        for term in self.drift_hamiltonian_terms:
            coeff_val = term.get("coefficient", 1.0)
            if isinstance(coeff_val, dict):
                coefficient = complex(coeff_val.get("real", 0), coeff_val.get("imag", 0))
            else:
                coefficient = float(coeff_val)
            op_list = []
            op_def = term.get("operators", {})
            for idx in range(len(self.system_dims)):
                if str(idx) in op_def.keys():
                    op_list.append(self.get_operator(op_def[str(idx)], self.system_dims[idx], mode))
                else:
                    if mode == 'qutip':
                        op_list.append(qt.identity(self.system_dims[idx]))
            
            if mode == 'latex':
                drift_hamiltonian += f" + {coefficient:.2f} " + " ".join(op_list)
            elif mode == 'qutip':
                drift_hamiltonian += coefficient * qt.tensor(op_list)
        
        return drift_hamiltonian

# --- 2. 实现具体的物理模型 ---
class DuffingOscillatorModel(HamiltonianModel):
    """
    Duffing 振子模型。
    它同时实现了必需的 Lab Frame 方法和可选的 RWA 方法。
    """
    def __init__(self, d: int, omega_q: float, anh: float):
        if d < 2:
            raise ValueError("能级数 d 必须至少为 2。")
        self._d = d
        self._omega_q = omega_q
        self.anh = anh
        self._a = destroy(self.d)
        self._adag = self._a.dag()
        self._num = num(self.d)
        self._identity = qeye(self.d)
    
    @property
    def d(self) -> int:
        return self._d
        
    @property
    def omega_q(self) -> float:
        return self._omega_q

    @classmethod
    def from_circuit_params(cls, d: int, e_j: float, e_c: float) -> 'DuffingOscillatorModel':
        omega_q_approx = np.sqrt(8 * e_j * e_c) - e_c
        anh_approx = -e_c
        return cls(d=d, omega_q=omega_q_approx, anh=anh_approx)
    

    def get_drift_hamiltonian(self, mode: str = 'qutip', use_rwa: bool = True) -> Qobj:
        if use_rwa:
            # RWA 自由项只包含非线性部分
            return (self.anh / 2) * (self._num * (self._num - self._identity))  # type: ignore
        else:
            # Lab Frame 自由项包含线性和非线性部分
            return self._omega_q * self._num + (self.anh / 2) * (self._num * (self._num - 1)) # type: ignore

    def get_control_hamiltonian(self, mode: str = 'qutip', use_rwa: bool = True) -> Dict[str, Any]:
        """
        定义此模型的“接线表”字典。
        """
        if use_rwa:
            return {
                'XY_drive': {
                    'I': ((self._a + self._adag) / 2),
                    'Q': (1j * (self._adag - self._a) / 2),
                    # 'I': 0 * qeye(self._d),
                    # 'Q': 0 * qeye(self._d),
                    'type': 'XY'
                }
            }
        else:
            return {
                'XY_drive': {
                    'S': (self._a + self._adag),
                    'type': 'XY',
                }
            }
        
    def visualize_energy_levels(self, unit: str = 'GHz'):
        """
        便捷方法，调用 plotting 模块来绘制当前模型的能级图。
        """
        # 在方法内部导入，避免循环依赖问题
        from . import plotting
        
        # 注意: 能量总是在 Lab Frame 下计算，所以 use_rwa=False
        h_drift_lab = self.get_drift_hamiltonian(use_rwa=False)
        plotting.plot_energy_levels(h_drift_lab, self.d, unit=unit)

