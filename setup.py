#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :setup.py
# @Time      :2022/12/14 23:00
# @Author    :juzipi
from setuptools import find_packages, setup

setup(
    name="aitoolkit",
    version='1.0.0',
    author="juzipi",
    author_email="1129501586@qq.com",
    license="MIT",
    keywords=['pip', 'ai toolkit'],
    description="ai toolkit",
    include_package_data=True,
    package_dir={"": "aitoolkit"},
    platforms="any",
    packages=find_packages('aitoolkit'),
    install_requires=['loguru', 'selenium', 'torch', 'transformers'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
)
