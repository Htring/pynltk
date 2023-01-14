#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :log.py
# @Time      :2022/12/4 19:47
# @Author    :juzipi
import time
from pathlib import Path
from loguru import logger


class Logger(object):
    __doc__ = """ custom logger by loguru """

    __instance = None

    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = super(Logger, cls).__new__(cls, *args, **kwargs)
        return cls.__instance

    def __init__(self, need_log: bool = True):
        self.__logger = logger

        if need_log:
            self.__logger.add(f"{Path().cwd()}/log_{time.strftime('%Y_%m_%d')}.log",
                              rotation="500MB",
                              encoding="utf-8",
                              enqueue=True,
                              retention="10 days")
            self.info(f"{Path().cwd()}/log_{time.strftime('%Y_%m_%d')}.log")

    def info(self, content: str):
        """
        log info
        Args:
            content: log content

        Returns:

        """
        self.__logger.info(content)

    def debug(self, content: str):
        """
        log debug
        Args:
            content: log debug

        Returns:

        """
        self.__logger.debug(content)

    def error(self, content: str):
        """
        log error
        Args:
            content: log error

        Returns:

        """
        self.__logger.error(content)

    def critical(self, content: str):
        """
        log critical
        Args:
            content: log critical

        Returns:

        """
        self.__logger.critical(content)

    def success(self, content: str):
        """
        log success
        Args:
            content: log success

        Returns:

        """
        self.__logger.success(content)

    def trace(self, content: str):
        """
        log trace
        Args:
            content: log trace

        Returns:

        """
        self.__logger.trace(content)

    def traceback(self):
        import traceback
        self.__logger.error(f'run error, error info:\n{traceback.format_exc()}')

