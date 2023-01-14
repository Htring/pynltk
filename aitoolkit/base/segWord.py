#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: segWord.py
@time:2021/06/12
@email:1129501586@qq.com
@description: 中文以字符方式切割，英文以单词方式切割，可自定义字典
"""

import os
import string


class SegWord(object):

    def __init__(self, load_inner=True):
        self._dict_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./innerDict.ini")
        self._word_dict = {}
        if load_inner:
            self.load_dict_file()

    def add_word(self, word):
        """
        添加自定义的分词词语到分词词典中
        :param word: 对应的词语
        :return:
        """
        if not word:
            raise Exception("word is null")
        first_char = word[0]
        self._word_dict.setdefault(first_char, [])
        if word not in self._word_dict[first_char]:
            self._word_dict[first_char].append(word)
            self._sort_word_dict()

    def _sort_word_dict(self):
        """
        对对应的字符所包含的字典进行排序
        :return:
        """
        for first_char, words in self._word_dict.items():
            self._word_dict[first_char] = sorted(words, key=lambda x: len(x), reverse=True)

    def load_dict_file(self, dict_file_path: str = ''):
        """
        加载字典
        :param dict_file_path: 字典文件地址
        :return:
        """
        if not dict_file_path:
            load_dict_path = self._dict_file_path
        else:
            if not os.path.exists(dict_file_path):
                raise Exception("can't find this file %s" % dict_file_path)
            else:
                load_dict_path = dict_file_path
        with open(load_dict_path, 'r', encoding='utf8') as reader:
            words = [word for word in reader.read().replace("\n\n", "\n").split('\n')]
        self.batch_add_words(words)

    def batch_add_words(self, words: list):
        """
        批量增加数据
        :param words: word的list
        :return:
        """
        for word in words:
            first_char = word[0]
            self._word_dict.setdefault(first_char, [])
            if word not in self._word_dict[first_char]:
                self._word_dict[first_char].append(word)
        self._sort_word_dict()

    def _match_word(self, first_char, i, sentence):
        """
        匹配
        :param first_char: 最新需要处理的开头字符
        :param i: 开头字符对应的索引
        :param sentence: 原始语句
        :return:
        """
        if first_char not in self._word_dict:
            if first_char in string.ascii_letters:
                return self._match_ascii(i, sentence)
            return first_char
        words = self._word_dict[first_char]
        for word in words:
            if sentence[i:i + len(word)] == word:
                return word
        if first_char in string.ascii_letters:
            return self._match_ascii(i, sentence)
        return first_char

    @staticmethod
    def _match_ascii(i, sentence):
        _result = ''
        for i in range(i, len(sentence)):
            if sentence[i] not in string.ascii_letters:
                break
            _result += sentence[i]
        return _result

    def tokenize(self, sentence):
        """
        分词
        :param sentence: 待分词语句
        :return:
        """
        tokens = []
        if not sentence:
            return tokens
        i = 0
        while i < len(sentence):
            first_char = sentence[i]
            matched_word = self._match_word(first_char, i, sentence)
            tokens.append(matched_word)
            i += len(matched_word)
        return tokens

    def tokenize_no_space(self, sentence):
        """
        返回无空格的分词
        :param sentence:
        :return:
        """
        _seg_word_result = self.tokenize(sentence)
        return [word for word in _seg_word_result if word not in string.whitespace]
