#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :dataset.py
# @Time      :2023/1/13 23:52
# @Author    :juzipi

from pathlib import Path
from typing import List
from loguru import logger
import torch
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader, TensorDataset
from transformers import BertTokenizer

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

    def __str__(self):
        return "{}, tag data:\n{}".format(self.guid, '\n'.join([f'{t}\t{l}' for t, l in zip(self.text, self.label)]))


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids,label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.label_mask = label_mask


class DataProcessor(object):
    """Base class for data converters for token tag data sets."""

    unk_tag = "<unk>"
    pad_tag = "<pad>"
    labels = {"O"}
    chars = set()

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.train_examples = []
        self.test_examples = []
        self.dev_examples = []

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        return self.train_examples

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        return self.dev_examples

    def get_test_examples(self):
        """Gets a collection of `InputExample`s for the test set."""
        return self.test_examples

    def get_data_loaders(self, *args, **kwargs):
        raise NotImplementedError()

    def _create_examples(self, lines: List, set_type: str):
        """
        create example from read lines
        Args:
            lines:
            set_type:

        Returns:

        """
        examples = []
        for i, (sentence, label) in enumerate(lines, 1):
            guid = "%s-%s" % (set_type, i)
            self.labels.update(label)
            self.chars.update(sentence)
            examples.append(InputExample(guid=guid, text=sentence, label=label))
        # examples.sort(key=lambda x: -len(x.text))
        return examples

    def _load_data(self):
        self.train_examples = self._create_examples(read_seq_tag_file(Path(self.data_dir, "train.txt")), "train")
        self.test_examples = self._create_examples(read_seq_tag_file(Path(self.data_dir, "test.txt")), "test")
        self.dev_examples = self._create_examples(read_seq_tag_file(Path(self.data_dir, "dev.txt")), "dev")

    def get_labels(self, add_bert_tag=False):
        if add_bert_tag:
            labels = list(self.labels) + ['[CLS]', '[SEP]']
        else:
            labels = list(self.labels)
        return labels


class NERDataProcessorCommon(DataProcessor):
    __doc__ = """ bilstm+crf、idcnn+crf model train data """

    @staticmethod
    def collate_fn(batch: List[InputExample], word2id: dict, label2id: dict, tag_pad_index: int = 0):
        """

        Args:
            tag_pad_index:
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
        batch_real_length = []
        UNK = word2id.get('<unk>')
        PAD = word2id.get('<pad>')
        for item in batch:
            inputs = [word2id.get(w, UNK) for w in item.text.copy()]
            targets = [label2id.get(label) for label in item.label.copy()]
            pad_len = max_len - len(inputs)
            assert len(inputs) == len(targets)
            batch_inputs.append(inputs + [PAD] * pad_len)
            batch_targets.append(targets + [tag_pad_index] * pad_len)
            batch_real_length.append(len(inputs))
        return torch.tensor(batch_inputs), torch.tensor(batch_targets), torch.tensor(batch_real_length)  # noqa

    def __init__(self, data_dir: str, batch_size: int = 128):
        super().__init__(data_dir)
        self.labels = set()
        self.chars = set()
        self.batch_size = batch_size
        self.train_examples, self.dev_examples, self.test_examples = [], [], []
        self.char2idx, self.tag2idx = {}, {}
        self.idx2tag, self.idx2char = {}, {}
        self.train_data_loader, self.dev_data_loader, self.test_data_loader = None, None, None
        self._load_data()
        self._build_dict()
        logger.info(f"{__name__} has been init")

    def _build_dict(self):
        """
        build corpus dict
        Returns:

        """
        self.char2idx = {char: i for i, char in enumerate([self.pad_tag, self.unk_tag] + self._get_chars())}
        self.tag2idx = {tag: i for i, tag in enumerate([self.pad_tag] + self.get_labels())}
        self.idx2char = {i: char for char, i in self.char2idx.items()}
        self.idx2tag = {index: value for index, value in enumerate(self.tag2idx.keys())}

    def _get_chars(self):
        chars = list(self.chars)
        chars.sort()
        return chars

    def get_data_loaders(self, batch_size: int = 32):
        """
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
                                            collate_fn=lambda x: self.collate_fn(x, self.char2idx, self.tag2idx))
        self.dev_data_loader = DataLoader(dev_examples,  # noqa
                                          sampler=eval_sampler,
                                          batch_size=batch_size,
                                          collate_fn=lambda x: self.collate_fn(x, self.char2idx, self.tag2idx))
        self.test_data_loader = DataLoader(test_examples,  # noqa
                                           sampler=test_sampler,
                                           batch_size=batch_size,
                                           collate_fn=lambda x: self.collate_fn(x, self.char2idx, self.tag2idx))
        return self.train_data_loader, self.dev_data_loader, self.test_data_loader


class NERDataProcessorBertSoftmax(DataProcessor):

    @staticmethod
    def convert_examples_to_features(examples: List[InputExample],
                                     label_map: dict,
                                     max_seq_length: int,
                                     tokenizer,
                                     sep_token: str = "[SEP]",
                                     cls_token: str = "[CLS]",
                                     ):
        """Loads a data file into a list of `InputBatch`s."""
        features = []
        for ex_index, example in enumerate(examples):
            if isinstance(example.text, str):
                example.text = example.text.split(" ")
            example_text_labels = example.label
            tokens = []  # bert deal tokens
            labels = []  # example token labels
            label_mask = []  # label mask flag
            for i, word in enumerate(example.text):
                word_tokens = tokenizer.tokenize(word)
                tokens.extend(word_tokens)  # current deal method word_tokens with one
                word_label: str = example_text_labels[i]
                labels.append(word_label)
                label_mask.append(1)
            if len(tokens) >= max_seq_length - 1:
                tokens = tokens[: (max_seq_length - 2)]
                labels = labels[: (max_seq_length - 2)]
                label_mask = label_mask[: (max_seq_length - 2)]
            s_tokens = []  # bert input standard tokens, add cls、sep
            segment_ids = []  # bert first sentence use 0 to represent, next sentence use 1 represent
            label_ids = []
            # add cls
            s_tokens.append(cls_token)
            segment_ids.append(0)
            label_mask.insert(0, 1)  # cls also need to mask
            label_ids.append(label_map[cls_token])
            for i, token in enumerate(tokens):
                s_tokens.append(token)
                segment_ids.append(0)
                label_ids.append(label_map[labels[i]])
            # add sep
            s_tokens.append(sep_token)
            segment_ids.append(0)
            label_mask.append(1)
            label_ids.append(label_map[sep_token])
            input_ids: list = tokenizer.convert_tokens_to_ids(s_tokens)
            input_mask = [1] * len(input_ids)
            label_mask = [1] * len(label_ids)
            # pad to max sequence length
            while len(input_ids) < max_seq_length:
                input_ids.append(0)  # bert dict pad token index is 0
                input_mask.append(0)  # pad token mask need set 0
                segment_ids.append(0)  # even pad, also first sentence
                label_ids.append(0)  # build label_map, index 0 for pad
                label_mask.append(0)  # even has label id, but, will not mask
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert len(label_mask) == max_seq_length
            if ex_index < 3:
                logger.info("*** Example ***")
                logger.info("guid: %s" % example.guid)
                logger.info("origin text: %s" % example.text)
                logger.info("origin labels: %s" % example.label)
                logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                logger.info("standard tokens: %s" % " ".join([str(x) for x in s_tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
                logger.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))
            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_ids=label_ids,
                              label_mask=label_mask))

        return features

    __doc__ = """ bert ner 数据处理 """

    def __init__(self, data_dir: str, pre_model: str, batch_size: int, max_length: int):
        """
        build bert ner data, sub word except first token label id is zero
        Args:
            data_dir: deal data dir
            pre_model: pre_train model path or pre_train bert model name
            batch_size:
            max_length: sequence max length
        """
        super().__init__(data_dir)
        self.tokenizer = BertTokenizer.from_pretrained(pre_model, do_lower_case=True)
        self.batch_size = batch_size
        self.max_length = max_length
        self._load_data()
        self.tag2idx = {tag: index for index, tag in enumerate(self.get_labels(add_bert_tag=True), 1)}

    def get_data_loader(self, examples: List[InputExample]):
        """
        single dataloader build
        Args:
            examples: kind examples

        Returns:

        """
        train_features = self.convert_examples_to_features(examples, self.tag2idx, self.max_length, self.tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in train_features], dtype=torch.long)
        all_label_mask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
        tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_label_mask_ids)
        sampler = RandomSampler(tensor_data)
        return DataLoader(tensor_data, sampler=sampler, batch_size=self.batch_size)

    def get_data_loaders(self):
        """
        build train dataloader
        Returns: train, dev, test dataloader
        """
        train_examples = self.get_train_examples()
        dev_examples = self.get_dev_examples()
        test_examples = self.get_test_examples()
        train_dataloader = self.get_data_loader(train_examples)
        dev_dataloader = self.get_data_loader(dev_examples)
        test_dataloader = self.get_data_loader(test_examples)
        return train_dataloader, dev_dataloader, test_dataloader
