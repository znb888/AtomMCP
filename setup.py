# setup.py

from setuptools import setup, find_packages

# 从 README.md 读取长描述
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 从 requirements.txt 读取依赖项
with open("requirements.txt", "r", encoding="utf-8") as f:
    install_requires = f.read().splitlines()

setup(
    name='atommcp',
    version='0.1.0',
    author='hugo, zhnb',
    author_email='thegranddesigner@outlook.com',
    description='A Python library for simulating neutral-atom quantum computing experiments.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    
    # 核心部分：告诉 setuptools 在哪里寻找源代码
    # package_dir={'': 'src'} 表示包的根目录是 'src'
    # find_packages(where='src') 会在 'src' 目录下寻找所有包
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    
    install_requires=install_requires,
    
    # 其他元数据
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    python_requires='>=3.8', # 与您的环境兼容的Python版本要求
)