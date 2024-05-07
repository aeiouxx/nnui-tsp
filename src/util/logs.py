import logging

import datetime as dt
import os


def ensure_logs_dir():
    import os
    if not os.path.exists('../logs'):
        os.makedirs('../logs')


def get_logger(name: str, filename: str = None, level: int = logging.DEBUG) -> logging.Logger:
    ensure_logs_dir()
    if filename is None:
        filename = f"{name}-{str(dt.datetime.now().strftime('%Y%m%d-%H%M%S'))}"
    logger = logging.getLogger(name)
    logging.basicConfig(
        filename="../logs/" + filename + ".log",
        level=level)
    return logger
