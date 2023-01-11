#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :common.py
# @Time      :2022/12/17 12:16
# @Author    :juzipi
import functools
import threading
import time
from typing import Callable
import loguru


def timer(func: Callable):
    """
    calculate func run time
    Args:
        func: callable function

    Returns: func run result

    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        loguru.logger.info(f"{func.__name__} function cost time {time.time() - start} s")
        return res

    return wrapper


def singleton(cls):
    """
    线程安全的单例模式装饰器
    Args:
        cls: need singleton object

    Returns:

    """

    _instance = {}
    _lock = threading.Lock()

    def wrapper(*args, **kwargs):
        if cls not in _instance:
            with _lock:
                if cls not in _instance:
                    _instance[cls] = cls(*args, **kwargs)
                return _instance[[cls]]
    return wrapper

