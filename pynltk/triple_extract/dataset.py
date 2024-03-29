#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :dataset.py
# @Time      :2023/1/20 14:34
# @Author    :juzipi
import os
from collections import OrderedDict
from pathlib import Path
from typing import List, Dict, Union
import torch
from pynltk.common import io_utils
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
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
    _train_file_name = "train.json"
    _valid_file_name = "dev.json"
    _test_file_name = "test.json"
    _schema_file_name = "schema.json"

    def __init__(self, data_dir):
        super(DataProcessorBert, self).__init__(data_dir)
        self._check_corpus([self._train_file_name, self._valid_file_name, self._test_file_name, self._schema_file_name])
        self.schema = {}
        self.rel2id = {}
        self._load_schema()

    def _load_schema(self):
        """
        load schema from file
        Returns:

        """
        schema_path = Path(self.data_dir, self._schema_file_name)
        for schema in io_utils.load_jsonl(schema_path):
            self.schema[schema['id']] = schema
            self.rel2id[schema['predicate']] = schema['id']


class CasRelDataProcessor(DataProcessorBert):
    __doc__ = """ CasRel triple extract data processor """

    def __init__(self, data_dir, pre_model, max_length, batch_size: int):
        super(CasRelDataProcessor, self).__init__(data_dir)
        self.tokenizer = BertTokenizer.from_pretrained(pre_model)
        self.max_length = max_length
        self.batch_size = batch_size

    class RelDataset(Dataset):
        """
        big data read
        """

        def __init__(self, file_path):
            self.data_list = self.get_data_list(file_path)

        @staticmethod
        def get_data_list(file_path):
            return io_utils.load_jsonl(file_path)

        def __len__(self):
            return len(self.data_list)

        def __getitem__(self, item):
            example = self.data_list[item]
            return example['text'], example['triple_list']

    def collate_fn(self, batch):

        def search(sequence, pattern):
            n = len(pattern)
            for _i in range(len(sequence)):
                if sequence[_i: _i + n] == pattern:
                    return _i
            return -1

        batch_subject_labels = []
        batch_object_labels = []
        batch_subject_ids = []
        batch_token_ids = []
        batch_attention_mask = []
        batch_token_type_ids = []
        for text, triples in batch:
            if len(text) > self.max_length - 2:
                text = text[: self.max_length - 2]
            tokens = list(text)
            spoes = {}
            callback_text_labels = []
            for s, p, o in triples:
                p_id = int(self.rel2id.get(p))
                s_idx = search(text, s)
                o_idx = search(text, o)
                if s_idx != -1 and o_idx != -1:
                    callback_text_labels.append((s, p, o))
                    s = (s_idx, s_idx + len(s) - 1)
                    o = (o_idx, o_idx + len(o) - 1, p_id)
                    spoes.setdefault(s, [])
                    spoes[s].append(o)
            if spoes:
                subject_labels = np.zeros((self.max_length, 2))
                for s in spoes:
                    subject_labels[s[0], 0] = 1  # subject head
                    subject_labels[s[1], 1] = 1  # subject tail
                start, end = np.array(list(spoes.keys())).T
                start = np.random.choice(start)
                end = np.random.choice(end[end >= start])
                # random for a valid subject
                subject_ids = (start, end)
                # object label
                object_labels = np.zeros((self.max_length, len(self.rel2id), 2))
                for o in spoes.get(subject_ids, []):
                    object_labels[o[0], o[2], 0] = 1  # object start
                    object_labels[o[1], o[2], 1] = 1  # object tail
                # build batch
                encode = self.tokenizer.encode_plus(tokens, padding="max_length", max_length=self.max_length)
                batch_token_ids.append(encode['input_ids'])
                batch_attention_mask.append(encode['attention_mask'])
                batch_token_type_ids.append(encode['token_type_ids'])
                batch_subject_labels.append(subject_labels)
                batch_object_labels.append(object_labels)
                batch_subject_ids.append(subject_ids)
                # callback_text_labels.append((text, callback_text_labels))
        token_ids = torch.tensor(batch_token_ids, dtype=torch.long)
        attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long)
        token_type_ids = torch.tensor(batch_token_type_ids, dtype=torch.long)
        subject_labels = torch.tensor(batch_subject_labels, dtype=torch.float)
        object_labels = torch.tensor(batch_object_labels, dtype=torch.float)
        subject_ids = torch.tensor(batch_subject_ids, dtype=torch.long)
        return token_ids, attention_mask, token_type_ids, subject_labels, object_labels, subject_ids

    def get_dataloaders(self):
        train_data = self.RelDataset(Path(self.data_dir, self._train_file_name))
        dev_data = self.RelDataset(Path(self.data_dir, self._valid_file_name))
        test_data = self.RelDataset(Path(self.data_dir, self._test_file_name))
        train_dataloader = DataLoader(train_data,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      collate_fn=self.collate_fn)
        dev_dataloader = DataLoader(dev_data,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    collate_fn=self.collate_fn)
        test_dataloader = DataLoader(test_data,
                                     batch_size=self.batch_size,
                                     shuffle=True,
                                     collate_fn=self.collate_fn)
        return train_dataloader, dev_dataloader, test_dataloader
