#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :file.py
# @Time      :2022/12/3 23:28
# @Author    :juzipi
import json
import pickle
import csv
from pathlib import Path
from typing import List, Dict
from loguru import logger
from functools import wraps


def mkdir_decorator(arg_index: int = 0, kind: int = 0):
    """
    for func create dir
    Args:
        kind: path type, 0 is default and is file path, 1 is dir
        arg_index: func path arg index
    Returns:

    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            path: Path = args[arg_index]
            assert isinstance(path, Path), f'path must type is pathlib.Path, input is {type(path).__name__}'
            if kind == 1:
                file_dir = path
            elif kind == 0:
                file_dir = path.parent
            else:
                raise Exception(f"kind value must int and in [0, 1], you input is {kind}, type is {type(kind)}")
            file_dir.mkdir(exist_ok=True)
            return func(*args, **kwargs)

        return wrapper

    return decorator


@mkdir_decorator(arg_index=1, kind=0)
def save_pickle(obj: object, file_path: Path, verbose=True) -> None:
    """
    save object to file path with pickle style
    Args:
        obj(object): need to save object
        file_path(Path): save path
        verbose: whether to print logs, default True

    Returns: None
    """
    with open(str(file_path), 'wb') as writer:
        pickle.dump(obj, writer)
    if verbose:
        logger.info(f'{obj.__name__} has been save to {file_path} (pickle)')


def load_pickle(file_path: Path, verbose=True):
    """
    load object from file path with pickle style
    Args:
        file_path: object save path
        verbose: whether to print logs, default True
    Returns: object
    """
    try:
        with open(str(file_path), 'rb') as reader:
            if verbose:
                logger.info(f'load data from {file_path}')
            return pickle.load(reader)
    except Exception as e:
        raise Exception(str(e))


@mkdir_decorator(arg_index=1, kind=0)
def save_json(obj: object, file_path: Path):
    """
    save object to file path with json style
    Args:
        obj: need to save object
        file_path:  save path

    Returns: None

    """
    with open(str(file_path), 'w', encoding='utf8') as writer:
        json.dump(obj, writer, ensure_ascii=False)


def load_json(file_path: Path):
    """
    load json from file path
    Args:
        file_path: object save path

    Returns: json data

    """
    try:
        with open(str(file_path), 'r', encoding='utf8') as reader:
            return json.load(reader)
    except Exception as e:
        raise Exception(str(e))


def load_csv(fp: Path, is_include_head: bool = True, verbose: bool = True) -> Dict:
    """
    read csv, tsv style data
    Args:
        is_include_head: whether include head, default True
        fp: file path
        verbose:  whether to print logs, default True

    Returns:

    """
    return __load_t_csv(fp, is_include_head, "csv", verbose)


def load_csv2jsonl(fp: Path, is_include_head: bool = True, verbose: bool = True):
    """
    load csv data to jsonl
    Args:
        is_include_head: whether include head, default True
        fp: file path
        verbose:  whether to print logs, default True

    Returns:

    """
    data = load_csv(fp, is_include_head, verbose)
    return head_content2jsonl(data['head'], data['content'])


def load_tsv(fp: Path, is_include_head: bool = True, verbose: bool = True) -> Dict:
    """
    read csv, tsv style data
    Args:
        is_include_head: whether include head, default True
        fp: file path
        verbose:  whether to print logs, default True

    Returns:

    """
    return __load_t_csv(fp, is_include_head, 'tsv', verbose)


def load_tsv2jsonl(fp: Path, is_include_head: bool = True, verbose: bool = True):
    """
    load csv data to jsonl
    Args:
        is_include_head: whether include head, default True
        fp: file path
        verbose:  whether to print logs, default True

    Returns:

    """
    data = load_tsv(fp, is_include_head, verbose)
    return head_content2jsonl(data['head'], data['content'])


def __load_t_csv(fp: Path, is_include_head: bool = True, file_type='csv', verbose: bool = True) -> Dict:
    """
    load tsv,csv base
    Args:
        fp: file path
        is_include_head: whether include head, default True
        file_type: file type, default csv
        verbose: whether to print logs,  default True
    Returns: read data

    """
    if verbose:
        logger.info(f'load csv/tsv from {fp}')
    assert file_type in ['csv', 'tsv'], f"delimiter must in ['csv','tsv'], you input {file_type}"
    if not fp.exists():
        raise Exception(f"file path {fp} not exist")
    data = {}
    with open(fp, 'r', encoding='utf8') as f:
        reader = csv.reader(f) if file_type == 'csv' else csv.DictReader(f, delimiter="\t")
        data["head"] = next(reader) if is_include_head else []
        data['content'] = list(reader)
    return data


@mkdir_decorator(arg_index=1, kind=0)
def save_csv(data: List[Dict], fp: Path, write_head: bool = True, verbose: bool = True) -> None:
    """
        save data to csv
        Args:
            data: need to save
            fp: save path
            write_head: whether to write head, default true
            verbose: whether to print log, default true

        Returns:

        """
    __save_t_csv(data, fp, write_head, verbose, "csv")


@mkdir_decorator(arg_index=1, kind=0)
def save_tsv(data: List[Dict], fp: Path, write_head: bool = True, verbose: bool = True) -> None:
    """
        save data to tsv
        Args:
            data: need to save
            fp: save path
            write_head: whether to write head, default true
            verbose: whether to print log, default true

        Returns:

        """
    __save_t_csv(data, fp, write_head, verbose, "tsv")


def __save_t_csv(data: List[Dict], fp: Path, write_head: bool = True, verbose: bool = True, file_type: str = 'csv'):
    """
    save data to tsv/csv
    Args:
        data: need to save
        fp: save path
        write_head: whether to write head, default true
        verbose: whether to print log, default true
        file_type: save file type, default csv

    Returns:

    """
    assert file_type in ['csv', 'tsv'], f'you input file_type {file_type}, not in ["csv", "tsv"]'
    if verbose:
        logger.info(f'save {"csv" if file_type else "tsv"} type to {fp}')
    if not data:
        return
    with open(fp, 'w', encoding='utf8') as f:
        head = data[0].keys()
        delimiter = ',' if file_type == 'csv' else "\t"
        writer = csv.DictWriter(f, fieldnames=head, delimiter=delimiter)
        if write_head:
            writer.writeheader()
        writer.writerows(data)


@mkdir_decorator(arg_index=0, kind=0)
def save_jsonl(fp: Path, data: List[Dict], verbose: bool = True) -> None:  # noqa
    """
    save data to jsonl file
    Args:
        data: need to save data
        fp: file path
        verbose: whether to print log, default True

    Returns:
        object: None

    """
    if verbose:
        logger.info(f"save data to {fp}")
    with open(fp, 'w', encoding='utf8') as writer:
        writer.write("\n".join([json.dumps(i, ensure_ascii=False) for i in data]))


def load_jsonl(fp: Path, verbose: bool = True) -> List[Dict]:  # noqa
    """
    load jsonl data from fp
    Args:
        fp: file path
        verbose: whether print log, default true

    Returns:
        object: List[Dict]

    """
    if verbose:
        logger.info(f"load jsonl from {fp}")
    if not fp.exists():
        return []
    with open(fp, 'r', encoding='utf8') as reader:
        return [json.loads(i) for i in reader]


def load_jsonl_iter(fp: Path, verbose: bool = True):
    """
    read jsonl with iterate
    Args:
        fp: file path
        verbose: whether print log, default true

    Returns:

    """
    if not fp.exists():
        return {}
    with open(fp, 'r', encoding='utf8') as reader:
        for i in reader:
            yield json.loads(i)


def jsonl2csv(fp: Path, verbose: bool = True, is_include_head: bool = True) -> Path:
    """
    jsonl data to csv
    Args:
        fp: json data path
        verbose: whether print log, default true
        is_include_head: csv whether write head, default true

    Returns:
        object: Path

    """
    assert isinstance(fp, Path), f'input fp must Path type'
    fp_new = Path(fp.parent, fp.name.split(".")[0] + '.jsonl')
    if verbose:
        logger.info(f"read jsonl data from {fp}")
    data_list = load_jsonl(fp)
    save_csv(data_list, fp_new, write_head=is_include_head, verbose=verbose)
    if verbose:
        logger.info(f"save csv data to {fp_new}")
    return fp_new


def jsonl2tsv(fp: Path, verbose: bool = True, is_include_head: bool = True) -> Path:
    """
    jsonl data to csv
    Args:
        fp: json data path
        verbose: whether print log, default true
        is_include_head: csv whether write head, default true

    Returns:
        object: Path

    """
    assert isinstance(fp, Path), f'input fp must Path type'
    fp_new = Path(fp.parent, fp.name.split(".")[0] + '.jsonl')
    if verbose:
        logger.info(f"read jsonl data from {fp}")
    data_list = load_jsonl(fp)
    save_tsv(data_list, fp_new, write_head=is_include_head, verbose=verbose)
    if verbose:
        logger.info(f"save tsv data to {fp_new}")
    return fp_new


def csv2jsonl(fp: Path, verbose: bool = True):
    """
    transform csv data to jsonl in csv dir path
    Args:
        fp: csv file path, csv file must have head
        verbose: whether print log, default true
    Returns:

    """
    assert isinstance(fp, Path), f'input fp must Path type'
    fp_new = Path(fp.parent, fp.name.split(".")[0] + '.jsonl')
    csv_data = load_csv(fp, verbose=verbose)
    if not csv_data['head']:
        raise Exception(f'csv data has\'t head')
    jsonl_data = head_content2jsonl(csv_data['head'], csv_data['content'])
    save_jsonl(fp_new, jsonl_data)
    if verbose:
        logger.info(f"save jsonl file to {fp_new}")
    return fp_new


def tsv2jsonl(fp: Path, verbose: bool = True):
    """
    transform csv data to jsonl in csv dir path
    Args:
        fp: csv file path, csv file must have head
        verbose: whether print log, default true
    Returns:

    """
    assert isinstance(fp, Path), f'input fp must Path type'
    fp_new = Path(fp.parent, fp.name.split(".")[0] + '.jsonl')
    tsv_data = load_tsv(fp, verbose=verbose)
    if not tsv_data['head']:
        raise Exception(f'tsv data has\'t head')
    jsonl_data = head_content2jsonl(tsv_data['head'], tsv_data['content'])
    save_jsonl(fp_new, jsonl_data)
    if verbose:
        logger.info(f"save jsonl file to {fp_new}")
    return fp_new


def head_content2jsonl(head: List[str], content: List[List]) -> List[Dict]:
    """
    transform head content data to jsonl data
    Args:
        head: key
        content: data
    Returns:
        object: List[Dict]
    """
    data = []
    for i in content:
        line = {h: d for h, d in zip(head, i)}
        data.append(line)
    return data


def read_lines(fp: Path, verbose: bool = True, **kwargs) -> List[str]:
    """
    read lines from fp
    Args:
        fp:
        verbose:
        **kwargs:

    Returns:

    """
    if verbose:
        logger.info(f"read data from {fp}")
    with open(file=fp, **kwargs) as reader:
        return reader.readlines()


@mkdir_decorator(arg_index=0, kind=0)
def save_lines(fp: Path, data_list: List[str], verbose: bool = True, **kwargs) -> None:
    """
    save data to path
    Args:
        data_list:
        fp:
        verbose:
        **kwargs:

    Returns:

    """
    if verbose:
        logger.info(f"save data to {fp}")
    kwargs.update({"mode": "w"})
    with open(file=fp, **kwargs) as writer:
        writer.write("\n".join(data_list))
