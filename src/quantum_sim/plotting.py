# -*- coding: utf-8 -*-
"""
plotting.py - 可视化功能模块

该模块包含所有用于绘制仿真结果和模型属性的函数。
"""

import numpy as np
import matplotlib.pyplot as plt
from .hamiltonian import DuffingOscillatorModel # 使用相对导入
from qutip import Qobj

def plot_energy_levels(h_drift: Qobj, d: int, model_name: str | None = None, unit: str = 'GHz'):
    """
    绘制给定的自由哈密顿量的能级图。
    """
    eigenvalues = h_drift.eigenenergies()
    
    if unit == 'GHz':
        energies = eigenvalues /  (2 * np.pi)
        ylabel = f"Energy ({unit})" 
    elif unit == 'rad/ns':
        energies = eigenvalues
        ylabel = f"Energy ({unit})"
    else:
        raise ValueError("Unit must be 'GHz' or 'rad/ns'")

    energies -= np.min(energies)

    fig, ax = plt.subplots(figsize=(6, 8))
    
    x_min, x_max = 0.1, 0.4
    cmap = plt.get_cmap('rainbow')
    colors = cmap(np.linspace(0, 1, len(energies)))

    for i, energy in enumerate(energies):
        ax.hlines(energy, x_min, x_max, color=colors[i], linewidth=3)
        ax.text(
            x_max + 0.05, 
            energy, 
            f"$|{i}\\rangle$: {energy:.4f} {unit}", 
            va='center', 
            fontsize=12,
            color=colors[i]
        )

    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_ylabel(ylabel, fontsize=14)
    if model_name:
        title = f"{model_name} Energy Levels"
    else:
        title = "Energy Levels"
    ax.set_title(title, fontsize=16) 
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.show()