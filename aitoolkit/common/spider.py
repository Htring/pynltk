#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :spider.py
# @Time      :2023/1/10 16:48
# @Author    :juzipi
import os
import platform
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options


def create_chrome_browser(headless=True, param=None) -> webdriver.Chrome:
    """
    创建一个chrome 浏览器
    Args:
        param:
        headless:

    Returns:

    """
    if param is None:
        param = {}
    load_dir = param.get("load_dir", None)
    if load_dir:
        os.makedirs(load_dir, exist_ok=True)
        os.chdir(Path(load_dir))
    load_dir = os.getcwd()
    options = Options()
    if headless or platform.system().lower() != "windows":
        options.add_argument("--headless")
    options.add_argument("window-size=1920x3000")
    options.add_experimental_option("excludeSwitches", ['enable-automation'])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_experimental_option("prefs", {
        # 保存密码设置
        "credentials_enable_service": False,
        "profile.password_manager_enabled": False,
        # 文件下载配置
        'profile.default_content_settings.popups': 0,  # 取消下载的确认弹窗
        'download.default_directory': load_dir
    })

    browser = webdriver.Chrome(options=options)
    browser.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument',
                            {"source": 'Object.defineProperty(navigator, "webdriver", {get:()=>undefined })'})
    browser.maximize_window()
    return browser
