#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :dataset.py
# @Time      :2023/1/20 14:34
# @Author    :juzipi
import math
import os
from collections import OrderedDict
from itertools import cycle
from pathlib import Path
from typing import List, Dict, Union
import torch
from pynltk.common import io_utils
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
from ..common.vocab import Vocab
from ..common.serializer import Serializer
from ..common.io_utils import load_csv2jsonl
import numpy as np


def _handle_attribute_data(attribute_data: List[Dict]) -> Dict:
    """
    handle attribute dict to predicate as key dict
    Args:
        attribute_data: list attribute data, such as:
        [{'attribute': 'None', 'index': '0'}, {'attribute': '民族', 'index': '1'}, {'attribute': '字', 'index': '2'},...]
    Returns:
        orderdict: such as:
        {'None': {'index': 0}, '民族': {'index': 1}, '字': {'index': 2}, '朝代': {'index': 3}, '身高': {'index': 4},
        '创始人': {'index': 5}, '上映时间': {'index': 6}}
    """
    atts = OrderedDict()
    attribute_data = sorted(attribute_data, key=lambda i: int(i['index']))
    for d in attribute_data:
        atts[d['attribute']] = {'index': int(d['index'])}
    return atts


def _add_attribute_data(atts: Dict, data: List) -> None:
    """
    transform origin example attribute content to idx
    Args:
        atts: attribute dict
        data: examples

    Returns:

    """
    for d in data:
        d['att2idx'] = atts[d['attribute']]['index']


def _serialize_sentence(data: List[Dict], serial):
    """
    deal example text(sentence) to split status ("")
    Args:
        data:
        serial:

    Returns:

    """
    for d in data:
        sent = d['sentence'].strip()
        # for entity and attribute_value add space
        sent = sent.replace(d['entity'], ' entity ', 1).replace(d['attribute_value'], ' attribute_value ', 1)
        # split sentence but not split entity and attribute value to new attribute tokens
        d['tokens'] = serial(sent, never_split=['entity', 'attribute_value'])
        entity_index, attribute_value_index = d['entity_offset'], d['attribute_value_offset']
        # add new attribute for entity index(first entity) and attribute value index
        d['entity_index'], d['attribute_value_index'] = int(entity_index), int(attribute_value_index)


def _convert_tokens_into_index(data: List[Dict], vocab: Vocab):
    """
    according to vocab convert token to index
    Args:
        data: examples
        vocab: convert dict

    Returns:

    """
    unk_idx = vocab.get_unk_idx()
    for d in data:
        d['token2idx'] = [vocab.word2idx.get(i, unk_idx) for i in d['tokens']]
        d['seq_len'] = len(d['token2idx'])


def _add_pos_seq(train_data: List[Dict], pos_limit):
    """

    Args:
        train_data:
        pos_limit: position limit

    Returns:

    """

    def _handle_pos_limit(pos: List[int], limit: int) -> List[int]:
        """
        trim pos location to limit and trans pos to positive
        Args:
            pos: entity location pos
            limit: pos limit

        Returns: positive position

        """
        for i, p in enumerate(pos):
            if p > limit:
                pos[i] = limit
            if p < -limit:
                pos[i] = -limit
        return [p + limit + 1 for p in pos]

    for d in train_data:
        if d['entity_index'] <= d['attribute_value_index']:
            entities_idx = [d['entity_index'], d['attribute_value_index']]
        else:
            entities_idx = [d['attribute_value_index'], d['entity_index']]
        d['entity_pos'] = list(map(lambda i: i - d['entity_index'], list(range(d['seq_len']))))
        d['attribute_value_pos'] = list(map(lambda i: i - d['attribute_value_index'], list(range(d['seq_len']))))
        d['entity_pos'] = _handle_pos_limit(d['entity_pos'], int(pos_limit))
        d['attribute_value_pos'] = _handle_pos_limit(d['attribute_value_pos'], int(pos_limit))
        # three part mask
        d['entities_pos'] = [1] * (entities_idx[0] + 1) + [2] * (entities_idx[1] - entities_idx[0] - 1) + [3] * (
                d['seq_len'] - entities_idx[1])


class DataProcessorBase(object):

    def __init__(self, data_dir: Union[Path, str]):
        self.data_dir = data_dir if isinstance(data_dir, Path) else Path(data_dir)

    def _check_corpus(self, file_list):
        """
        check file names exist, and file name must equal to define
        Args:
            file_list: file names

        Returns:

        """
        if not self.data_dir.exists():
            raise Exception(f"data dir {self.data_dir} not exist")
        files = os.listdir(self.data_dir)
        not_in = []
        for file in file_list:
            if file not in files:
                not_in.append(file)
        info = '【{}】 not in {}, file name must 【{}】'.format(",".join(not_in),
                                                            self.data_dir,
                                                            ",".join(file_list))
        if not_in:
            raise Exception(info)


class DataProcessor(DataProcessorBase):
    _train_file_name = "train.csv"
    _valid_file_name = "valid.csv"
    _test_file_name = "test.csv"
    _attr_file_name = "attribute.csv"

    def __init__(self, data_dir, pos_limit, min_freq: int = 3, chinese_split: bool = True):
        """

        Args:
            data_dir: train data save  dir
            pos_limit:
            min_freq: vocab 构建时的最低词频控制
            chinese_split: 是否需要分词
        """
        super().__init__(data_dir)
        self.chinese_split = chinese_split
        self.min_freq = min_freq
        self.train_examples, self.valid_examples, self.test_examples = None, None, None
        self.vocab = Vocab("triple_extract")
        self.pos_limit: int = pos_limit
        self._check_corpus([self._train_file_name, self._valid_file_name, self._test_file_name, self._attr_file_name])

    def build_examples(self, *args, **kwargs):
        """
        according to model detail use custom build method
        Args:
            *args:
            **kwargs:

        Returns:

        """
        raise NotImplementedError

    def _load_data(self):
        """
        load raw data
        Returns: tuple: train_data(List[Dict]), valid_data(List[Dict]), test_data(List[Dict])

        """
        # read data to jsonl
        train_data = load_csv2jsonl(Path(self.data_dir, self._train_file_name))
        valid_data = load_csv2jsonl(Path(self.data_dir, self._valid_file_name))
        test_data = load_csv2jsonl(Path(self.data_dir, self._test_file_name))
        attribute_data = load_csv2jsonl(Path(self.data_dir, self._attr_file_name))
        atts = _handle_attribute_data(attribute_data)
        _add_attribute_data(atts, train_data)
        _add_attribute_data(atts, test_data)
        _add_attribute_data(atts, valid_data)
        serializer = Serializer(do_chinese_split=self.chinese_split, do_lower_case=True)
        serial = serializer.serialize  # split sentence method
        _serialize_sentence(train_data, serial)
        _serialize_sentence(valid_data, serial)
        _serialize_sentence(test_data, serial)
        self.vocab.index2tag = {attr['index']: attr['attribute'] for attr in attribute_data}
        train_tokens = [d['tokens'] for d in train_data]
        valid_tokens = [d['tokens'] for d in valid_data]
        test_tokens = [d['tokens'] for d in test_data]
        sent_tokens = [*train_tokens, *valid_tokens, *test_tokens]
        for sent in sent_tokens:
            self.vocab.add_words(sent)
        self.vocab.trim(min_freq=self.min_freq)
        return train_data, valid_data, test_data


class PCNNDataProcessor(DataProcessor):
    """
    base pcnn for triple data extract
    """

    def __init__(self,
                 data_dir,
                 batch_size: int = 128,
                 pos_limit: int = 30,
                 max_seq_length: int = 512,
                 chinese_split: bool = True):
        """

        Args:
            data_dir: data dir
            batch_size: data batch size
            pos_limit: entity position limit
            max_seq_length: sentence max length
        """
        super().__init__(data_dir, pos_limit=pos_limit, chinese_split=chinese_split)
        self.batch_size = batch_size
        self.char2idx, self.idx2char = {}, {}
        self.idx2tag, self.tag2idx = {}, {}
        self.max_seq_length = max_seq_length
        self.build_examples()

    class PCNNDataset(Dataset):

        def __init__(self, data_list: list):
            self.data_list = data_list

        def __getitem__(self, index):
            return self.data_list[index]

        def __len__(self):
            return len(self.data_list)

    @staticmethod
    def collate_fn(max_len: int = 512):
        """
        closure to save some attribute
        Args:
        Returns:
        """

        def wrapper(batch):
            """
             Arg :
                batch () : 数据集
            Returns :
                x (dict) : key为词，value为长度
                y (List) : 关系对应值的集合
            """

            batch.sort(key=lambda example: -example['seq_len'])

            def _padding(x_, max_len_):
                return x_ + [0] * (max_len_ - len(x_))

            x, y = dict(), []
            word, word_len = [], []
            head_pos, tail_pos = [], []
            pcnn_mask = []
            for data in batch:
                word.append(_padding(data['token2idx'], max_len))
                head_pos.append(_padding(data['entity_pos'], max_len))
                tail_pos.append(_padding(data['attribute_value_pos'], max_len))
                pcnn_mask.append(_padding(data['entities_pos'], max_len))
                word_len.append(data['seq_len'])
                y.append(int(data['att2idx']))
            x['word'] = torch.tensor(word)
            x['lens'] = torch.tensor(word_len)
            x['entity_pos'] = torch.tensor(head_pos)
            x['attribute_value_pos'] = torch.tensor(tail_pos)
            x['pcnn_mask'] = torch.tensor(pcnn_mask)
            y = torch.tensor(y)
            return x, y

        return wrapper

    def build_examples(self):
        """
        from load data to build examples
        Returns:

        """
        train_data, valid_data, test_data = self._load_data()
        self.char2idx = self.vocab.word2idx
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.idx2tag = self.vocab.index2tag
        self.tag2idx = {tag: idx for idx, tag in self.idx2tag.items()}
        # convert tokens to index
        _convert_tokens_into_index(train_data, self.vocab)
        _convert_tokens_into_index(valid_data, self.vocab)
        _convert_tokens_into_index(test_data, self.vocab)
        # for examples add position data
        _add_pos_seq(train_data, self.pos_limit)
        _add_pos_seq(valid_data, self.pos_limit)
        _add_pos_seq(test_data, self.pos_limit)
        self.train_examples = train_data
        self.valid_examples = valid_data
        self.test_examples = test_data

    def get_data_loaders(self):
        """
        build model train dataloader
        Returns:

        """
        train_datasets = DataLoader(self.PCNNDataset(self.train_examples),
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    collate_fn=self.collate_fn(self.max_seq_length))
        valid_datasets = DataLoader(self.PCNNDataset(self.valid_examples),
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    collate_fn=self.collate_fn(self.max_seq_length))
        test_datasets = DataLoader(self.PCNNDataset(self.test_examples),
                                   batch_size=self.batch_size,
                                   shuffle=True,
                                   collate_fn=self.collate_fn(self.max_seq_length))
        return train_datasets, valid_datasets, test_datasets


class DataProcessorBert(DataProcessorBase):
    _train_file_name = "train_triples.json"
    _valid_file_name = "dev_triples.json"
    _test_file_name = "test_triples.json"
    _rel2id_file_name = "rel2id.json"

    def __init__(self, data_dir):
        super(DataProcessorBert, self).__init__(data_dir)
        self._check_corpus([self._train_file_name, self._valid_file_name, self._test_file_name, self._rel2id_file_name])
        self.rel2id, self.id2rel = io_utils.load_json(Path(self.data_dir, self._rel2id_file_name))


class CasRelDataProcessor(DataProcessorBert):
    __doc__ = """ CasRel triple extract data processor """

    def __init__(self, data_dir, pre_model, max_length, batch_size: int):
        super(CasRelDataProcessor, self).__init__(data_dir)
        self.tokenizer = BertTokenizer.from_pretrained(pre_model)
        self.max_length = max_length
        self.batch_size = batch_size

    class RelIterableDataset(IterableDataset):  # noqa
        """
        big data read
        """

        def __init__(self, file_path, tokenizer, max_seq_len, batch_size, rel2id: dict):
            self.file_path = file_path
            self.info = self._get_file_info(file_path)
            self.start = self.info['start']
            self.end = self.info['end']
            self.tokenizer = tokenizer
            self.max_seq_len = max_seq_len
            self.rel2id = rel2id
            self.num_rel = len(self.rel2id)
            self.batch_size = batch_size

        def __iter__(self):
            worker_info = get_worker_info()
            if worker_info is None:
                iter_start = self.start
                iter_end = self.end
            else:
                per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
                worker_id = worker_info.id
                iter_start = self.start + worker_id * per_worker
                iter_end = min(iter_start + per_worker, self.end)
            data_iter = self._sample_generator(iter_start, iter_end)
            return cycle(data_iter)

        @staticmethod
        def _get_file_info(file_path):
            info = {'start': 1, 'end': 0}
            for _ in io_utils.load_jsonl_iter(file_path):
                info['end'] += 1
            return info

        def __len__(self):
            return self.end - self.start

        @staticmethod
        def _search(sequence: list, pattern: list):
            n = len(pattern)
            for i in range(len(sequence)):
                if sequence[i: i + n] == pattern:
                    return i
            return -1

        def _sample_generator(self, start, end):
            # text bert token
            batch_token_ids = np.zeros((self.batch_size, self.max_seq_len), dtype=np.int32)
            batch_mask_ids = np.zeros((self.batch_size, self.max_seq_len), dtype=np.int_)
            batch_segment_ids = np.zeros((self.batch_size, self.max_seq_len), dtype=np.int_)
            batch_subject_ids = np.zeros((self.batch_size, 2), dtype=np.int_)
            batch_subject_labels = np.zeros((self.batch_size, self.max_seq_len, 2), dtype=np.int_)
            batch_object_labels = np.zeros((self.batch_size, self.max_seq_len, self.num_rel, 2), dtype=np.int_)
            batch_i = 0
            for i, data in enumerate(io_utils.load_jsonl_iter(self.file_path)):
                if i < start:
                    continue
                if i >= end:
                    return StopIteration
                text = data['text']
                spo_list = data['triple_list']
                batch_token_ids[batch_i, :] = self.tokenizer.encode(text,
                                                                    max_length=self.max_seq_len,
                                                                    pad_to_max_length=True,
                                                                    add_special_tokens=True)
                batch_mask_ids[batch_i, :len(text) + 2] = 1  # text mask
                idx = np.random.randint(0, len(spo_list), size=1)[0]
                s_rand = self.tokenizer.encode(spo_list[idx][0])[1:-1]
                s_rand_idx = self._search(list(batch_token_ids[batch_i, :]), s_rand)
                # subject token sentence idx
                batch_subject_ids[batch_i, :] = [s_rand_idx, s_rand_idx + len(s_rand) - 1]
                for j, spo in enumerate(spo_list):
                    s = self.tokenizer.encode(spo[0])[1: -1]
                    p = self.rel2id.get(spo[1])
                    o = self.tokenizer.encode(spo[2])[1: -1]
                    s_idx = self._search(list(batch_token_ids[batch_i]), s)
                    o_idx = self._search(list(batch_token_ids[batch_i]), o)
                    if s_idx != -1 and o_idx != -1:
                        # subject start, end set 1
                        batch_subject_labels[batch_i, s_idx, 0] = 1
                        batch_subject_labels[batch_i, s_idx + len(s) - 1, 1] = 1
                        if s_idx == s_rand_idx:
                            batch_object_labels[batch_i, o_idx, p, 0] = 1
                            batch_object_labels[batch_i, o_idx + len(o) - 1, p, 1] = 1
                batch_i += 1
                if batch_i == self.batch_size or i == end:
                    yield batch_token_ids, batch_mask_ids, batch_segment_ids, batch_subject_labels, batch_subject_ids, \
                          batch_object_labels
                    batch_token_ids[:, :] = 0
                    batch_mask_ids[:, :] = 0
                    batch_subject_ids[:, :] = 0
                    batch_subject_labels[:, :, :] = 0
                    batch_object_labels[:, :, :, :] = 0
                    batch_i = 0

    def get_dataloaders(self):
        train_data = self.RelIterableDataset(Path(self.data_dir, self._train_file_name),
                                             self.tokenizer,
                                             self.max_length,
                                             self.batch_size,
                                             self.rel2id)
        dev_data = self.RelIterableDataset(Path(self.data_dir, self._valid_file_name),
                                           self.tokenizer,
                                           self.max_length,
                                           self.batch_size,
                                           self.rel2id)
        test_data = self.RelIterableDataset(Path(self.data_dir, self._test_file_name),
                                            self.tokenizer,
                                            self.max_length,
                                            self.batch_size,
                                            self.rel2id)
        return DataLoader(train_data), DataLoader(dev_data), DataLoader(test_data)
