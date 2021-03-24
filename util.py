""" Some utility functions."""
import json
import logging
import pathlib
import typing
from itertools import islice
from collections import deque
from collections.abc import Iterable
from datetime import datetime, timezone


def init_logger(
    log_dir: pathlib.Path = pathlib.Path('logs'),
    log_name: str = 'mongo',
    debug: bool = True
) -> logging.Logger:
    """ Initiate logger for fetching."""
    log_fmt = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_dir / f'{log_name}_{datetime.now().timestamp()}')
    file_handler.setFormatter(log_fmt)

    stream_handle = logging.StreamHandler()
    stream_handle.setFormatter(log_fmt)

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO if not debug else logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handle)

    return logger


def year_start(year: int) -> datetime:
    """ Return start of the year."""
    return datetime(year, 1, 1, 0, 0, 0, 0, timezone.utc)


def year_end(year: int) -> datetime:
    """ Return end of the year."""
    return datetime(year, 12, 31, 23, 59, 59, 999999, timezone.utc)


def replace_word_by_keyword(keyword: str, replacement: str, target_lst: list[str]) -> list[str]:
    """ Replace a string element in the `target_lst` with `replacement` str if the
        element contains `keyword` string.
    """
    return [replacement if keyword.lower() in el.lower() else el for el in target_lst]


def json_list_append(obj: typing.Any, path: pathlib.Path):
    """ Append an JSON-able object to the end of a json file.
        **USE WITH CAUSION**: It's only okay to use this function of very small amount of data.
    """
    json_lst: list = []
    if path.is_file():
        with open(path) as fjson:
            json_lst = json.load(fjson)

    json_lst.append(obj)
    with open(path, 'w') as fjson:
        json.dump(json_lst, fjson, indent=4)


def slide_window(iterable: Iterable, size: int = 2) -> typing.Generator[list, None, None]:
    """ Sliding window for arbitrary iterable."""
    iterator = iter(iterable)
    window = deque(islice(iterator, size), maxlen=size)  # `islice` iterate the iterator directly, not making copy
    for element in iterator:
        yield list(window)
        window.append(element)  # Automatically pop left most item
    if window:
        yield list(window)
