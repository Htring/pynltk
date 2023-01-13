#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :data_util.py
# @Time      :2023/1/12 10:54
# @Author    :juzipi
from pathlib import Path
from typing import List

from loguru import logger
from common import io_utils


def _convert2json(fp: str, target_fp: str, split="\t", tag_type="bio"):
    """
    将bio, bieso格式的数据转成json格式
    :param split: 数据之间的默认切割符为\t
    :param target_fp: json格式数据保存地址
    :param fp: 文件地址
    :return:
    """

    def format_data(char_lst, tag_lst):
        tmp_dict = {"text": "".join(char_lst), "label": {}}
        entities = get_single_result("".join(char_lst), tag_lst, tag_type=tag_type)
        for entity in entities:
            label_dict = tmp_dict['label']
            if entity['type'] not in tmp_dict['label']:
                label_dict[entity['type']] = {entity['value']: [[entity['begin'], entity['end']]]}
            else:
                if entity['value'] not in label_dict[entity['type']]:
                    label_dict[entity['type']][entity['value']] = [[entity['begin'], entity['end']]]
                else:
                    label_dict[entity['type']][entity['value']].append([entity['begin'], entity['end']])
        return tmp_dict

    dict_lst, tmp_char_lst, tmp_tag_lst = [], [], []
    for index, line in enumerate(io_utils.read_lines(Path(fp))):
        line: str = line.strip()
        if line:
            char, tag = line.split(split)[:2]
            tmp_char_lst.append(char)
            tmp_tag_lst.append(tag)
        else:
            if tmp_tag_lst and tmp_char_lst:
                format_dict = format_data(tmp_char_lst, tmp_tag_lst)
                dict_lst.append(format_dict)
                tmp_char_lst, tmp_tag_lst = [], []
    if tmp_tag_lst and tmp_char_lst:
        dict_lst.append(format_data(tmp_char_lst, tmp_tag_lst))
    io_utils.save_jsonl(Path(target_fp), dict_lst)


def _bio_data_handler(sentence: str, predict_labels: list):
    """
    待处理的bio tag 数据
    :param sentence:单条sentence
    :param predict_labels: sentence对应预测结果
    :return:
    """
    entities = []
    pre_label = predict_labels[0]
    word = sentence[0] if pre_label.startswith("B") else ""
    for i in range(1, len(sentence)):
        current_label = predict_labels[i]
        if current_label.startswith('B'):
            if pre_label[2:] is not current_label[:2] and word != '':
                entities.append([word, pre_label[2:]])
                word = ''
            pre_label = current_label
            word += sentence[i]
        elif current_label.startswith('I'):
            word += sentence[i]
            pre_label = current_label
        elif current_label.startswith('O'):
            if pre_label[2:] is not current_label[2:] and word != '':
                entities.append([word, pre_label[2:]])
                word = ''
    if word != '':
        entities.append([word, pre_label[2:]])
    return entities


def _bieso_data_handler(sentence: str, predict_labels: list):
    """
    待处理的bieso tag 数据
    :param sentence:单条sentence
    :param predict_labels: sentence对应预测结果
    :return:
    """
    entities = []
    pre_label = predict_labels[0]
    word = sentence[0] if pre_label.startswith("B") or pre_label.startswith("S") else ""
    for i in range(1, len(sentence)):
        current_label = predict_labels[i]
        if current_label.startswith('B'):
            if pre_label[2:] is not current_label[:2] and word != '':
                entities.append([word, pre_label[2:]])
                word = ''
            pre_label = current_label
            word += sentence[i]
        elif current_label.startswith('I') or current_label.startswith("M"):  # maybe bmeso
            word += sentence[i]
            pre_label = current_label
        elif current_label.startswith('E'):
            word += sentence[i]
            pre_label = current_label
        elif current_label.startswith('O'):
            if pre_label[2:] is not current_label[2:] and word != '':
                entities.append([word, pre_label[2:]])
                word = ''
        elif current_label.startswith('S'):
            if pre_label[2:] is not current_label[2:] and word != '':
                entities.append([word, pre_label[2:]])
                word = ''
            entities.append([sentence[i], pre_label[2:]])

    if word != '':
        entities.append([word, pre_label[2:]])
    return entities


def get_single_result(sentence: str, predict_labels, tag_type="bio"):
    """
    获取单条数据处理结果
    :param tag_type: 数据的标记方式，默认bio方式
    :param sentence: 语句
    :param predict_labels: 预测结果, 索引区间前后都包括
    :return:
    """
    if len(predict_labels) == 0:
        return []
    if tag_type.lower() == "bio":
        raw_entities = _bio_data_handler(sentence, predict_labels)
    elif tag_type.lower() == "bieso" or tag_type.lower() == 'bmeso':
        raw_entities = _bieso_data_handler(sentence, predict_labels)
    else:
        raise Exception(f"you input tag_type is {tag_type}, not in ['bio', 'bieso', 'bmeso']")
    entities = []
    sentence = sentence.lower()
    if len(raw_entities) != 0:
        prefix_len, end = 0, 0
        for word, label in raw_entities:
            sub_sentence = sentence[end:]
            begin = sub_sentence.find(word.lower()) + prefix_len
            end = begin + len(word)
            if begin >= prefix_len:
                entity = dict(value=sentence[begin: end], type=label, begin=begin, end=end - 1)
                entities.append(entity)
            prefix_len = end
    return entities


def get_entities(contents, predict_labels, tag_type="bio"):
    """
    获取实体结果数据
    :param tag_type: tag的标记方式，default bio
    :param contents: 需要处理的内容
    :param predict_labels: 对于预测的标签
    :return:
    """
    results = []
    if not contents:
        return results
    if isinstance(contents, str):
        contents = [contents]
        predict_labels = [predict_labels]
    for sentence, sub_predict_labels in zip(contents, predict_labels):
        results.append(get_single_result(sentence, sub_predict_labels, tag_type=tag_type))
    return results


def convert_json2bio(fp: str, target_fp: str):
    """
    将json格式的数据转换成bio标注类型的数据
    :param fp: json格式数据地址
    :param target_fp: 生成的数据存储地址
    :return:
    """
    tagged_lst = []
    for index, dict_data in enumerate(io_utils.load_jsonl(Path(fp))):
        text = dict_data['text']
        labels = dict_data['label']
        char_lst = list(text)
        tag_lst = ["O"] * len(char_lst)
        for entity_key in labels:
            current_kind_entities = labels[entity_key]
            for sub_entity_key in current_kind_entities:
                districts = current_kind_entities[sub_entity_key]
                for district in districts:
                    start, end = district
                    if end - start == 0:
                        tag_lst[start] = "B-" + entity_key
                    elif end - start > 0:
                        tag_lst[start] = "B-" + entity_key
                        tag_lst[start + 1: end + 1] = ["I-" + entity_key] * (end - start)
                    else:
                        raise Exception(f"line {index + 1}, entity end > start")
        tagged_sentence = ""
        for char, tag in zip(char_lst, tag_lst):
            if char:  # 过滤空格
                tagged_sentence += f"{char}\t{tag}\n"
        tagged_lst.append(tagged_sentence)

    io_utils.save_lines(Path(target_fp), tagged_lst)


def convert_json2bieso(fp: str, target_fp: str):
    """
    将json格式的数据转换成bio标注类型的数据
    :param fp:
    :param target_fp:
    :return:
    """
    tagged_lst = []
    for index, dict_data in enumerate(io_utils.load_jsonl(Path(fp))):
        text = dict_data['text']
        labels = dict_data['label']
        char_lst = list(text)
        tag_lst = ["O"] * len(char_lst)
        for entity_key in labels:
            current_kind_entities = labels[entity_key]
            for sub_entity_key in current_kind_entities:
                districts = current_kind_entities[sub_entity_key]
                for district in districts:
                    start, end = district
                    if end - start == 0:
                        tag_lst[start] = "S-" + entity_key
                    elif end - start > 0:
                        tag_lst[start] = "B-" + entity_key
                        tag_lst[end] = "E-" + entity_key
                        if end - start > 1:
                            tag_lst[start + 1: end] = ["I-" + entity_key] * (end - start - 1)
                    else:
                        raise Exception(f"line {index + 1}, entity end > start")
        tagged_sentence = ""
        for char, tag in zip(char_lst, tag_lst):
            if char:  # 过滤空格
                tagged_sentence += f"{char}\t{tag}\n"
        tagged_lst.append(tagged_sentence)
    io_utils.save_lines(Path(target_fp), tagged_lst)


def read_seq_tag_file(fp: Path, verbose: bool = True, **kwargs) -> List:
    """
    read sequence tag train data
    Args:
        verbose:
        fp:
        **kwargs:

    Returns:

    """
    if verbose:
        logger.info(f'read sequence tag data from {fp}')
    if "encoding" not in kwargs:
        kwargs['encoding'] = 'utf-8'
    data = []
    sentence, label = [], []
    with open(file=fp, **kwargs) as reader:
        for line in reader:
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                if len(sentence) > 0:
                    data.append((sentence, label))
                    sentence = []
                    label = []
                continue
            splits = line.strip().split()
            sentence.append(splits[0])
            label.append(splits[-1])
        if len(sentence) > 0:
            data.append((sentence, label))
            sentence = []
            label = []
    return data
