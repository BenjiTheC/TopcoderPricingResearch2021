""" Some utility functions."""
import logging
import pathlib
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
