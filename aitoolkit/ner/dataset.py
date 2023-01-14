#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :dataset.py
# @Time      :2023/1/13 23:52
# @Author    :juzipi

from pathlib import Path
from typing import List

import torch
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader

from .data_util import read_seq_tag_file


class InputExample(object):
    """A single training/dev/test example for sequence token."""

    def __init__(self, guid: str, text: list, label=None):
        """
        Constructs a InputExample.
            Args:
                guid(string): Unique id for the example.
                text(List): The untokenized text of the first sequence
                label(List, optional): The label of the example.
        """
        self.guid: str = guid
        self.text: list = text
        self.label: list = label


class DataProcessor(object):
    """Base class for data converters for token tag data sets."""

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class NERDataProcessorCommon(DataProcessor):
    __doc__ = """ bilstm+crf model train data """

    unk_tag = "<unk>"
    pad_tag = "<pad>"

    @staticmethod
    def collate_fn(batch: List[InputExample], word2id: dict, label2id: dict):
        """

        Args:
            batch:
            word2id:
            label2id:

        Returns:
            batch_inputs

        """
        batch.sort(key=lambda x: len(x.text), reverse=True)
        max_len = len(batch[0].text)
        batch_inputs = []
        batch_targets = []
        batch_masks = []
        batch_real_length = []
        UNK = word2id.get('<unk>')
        PAD = word2id.get('<pad>')
        for item in batch:
            inputs = [word2id.get(w, UNK) for w in item.text.copy()]
            targets = [label2id.get(label) for label in item.label.copy()]
            pad_len = max_len - len(inputs)
            assert len(inputs) == len(targets)
            batch_inputs.append(inputs + [PAD] * pad_len)
            batch_targets.append(targets + [0] * pad_len)
            batch_masks.append([1] * len(inputs) + [0] * pad_len)
            batch_real_length.append(len(inputs))
        return torch.tensor(batch_inputs), torch.tensor(batch_targets), torch.tensor(batch_masks).bool(), torch.tensor(batch_real_length)   # noqa

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.labels = set()
        self.chars = set()
        self.train_examples, self.dev_examples, self.test_examples = [], [], []
        self.train_data_loader, self.dev_data_loader, self.test_data_loader = None, None, None
        self._load_data()

    def get_special_tags(self):
        return [self.unk_tag, self.pad_tag]

    def _load_data(self):
        self.train_examples = self._create_examples(read_seq_tag_file(Path(self.data_dir, "train.txt")), "train")
        self.test_examples = self._create_examples(read_seq_tag_file(Path(self.data_dir, "test.txt")), "test")
        self.dev_examples = self._create_examples(read_seq_tag_file(Path(self.data_dir, "dev.txt")), "dev")

    def get_train_examples(self) -> List[InputExample]:
        return self.train_examples

    def get_dev_examples(self):
        return self.dev_examples

    def get_test_examples(self):
        return self.test_examples

    def get_labels(self, add_bert_tag=False):
        if "O" not in self.labels:
            self.labels.add("O")
        if add_bert_tag:
            self.labels.update(['[CLS]', '[SEP]'])
        labels = list(self.labels)
        labels.sort()
        return labels

    def get_chars(self):
        chars = list(self.chars)
        chars.sort()
        return chars

    def _create_examples(self, lines: List, set_type: str):
        """
        create example from read lines
        Args:
            lines:
            set_type:

        Returns:

        """
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            self.labels.update(label)
            self.chars.update(sentence)
            examples.append(InputExample(guid=guid, text=sentence, label=label))
        examples.sort(key=lambda x: -len(x.text))
        return examples

    def get_data_loaders(self, char2idx: dict, tag2idx: dict, batch_size: int = 32, ):
        """

        Args:
            batch_size:
            char2idx:
            tag2idx:

        Returns:

        """
        train_examples = self.get_train_examples()
        dev_examples = self.get_dev_examples()
        test_examples = self.get_test_examples()
        train_sampler = RandomSampler(train_examples)
        eval_sampler = SequentialSampler(dev_examples)
        test_sampler = SequentialSampler(test_examples)
        self.train_data_loader = DataLoader(train_examples,  # noqa
                                            sampler=train_sampler,
                                            batch_size=batch_size,
                                            collate_fn=lambda x: self.collate_fn(x, char2idx, tag2idx))
        self.dev_data_loader = DataLoader(dev_examples,   # noqa
                                          sampler=eval_sampler,
                                          batch_size=batch_size,
                                          collate_fn=lambda x: self.collate_fn(x, char2idx, tag2idx))
        self.test_data_loader = DataLoader(test_examples,   # noqa
                                           sampler=test_sampler,
                                           batch_size=batch_size,
                                           collate_fn=lambda x: self.collate_fn(x, char2idx, tag2idx))
        return self.train_data_loader, self.dev_data_loader, self.test_data_loader
