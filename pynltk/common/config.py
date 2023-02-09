#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :config.py
# @Time      :2023/2/4 17:15
# @Author    :juzipi

from .common import singleton


@singleton
class SingletonObj(object):
    """
    thread safety obj
    """
    pass


SingletonConfig = SingletonObj().__class__  # thread safety obj
