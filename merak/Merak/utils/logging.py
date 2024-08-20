# https://github.com/microsoft/DeepSpeed/blob/85ce85dd5f4b18c0019a5121b06900e3a2c3933b/deepspeed/utils/logging.py

import logging
import sys
import torch
from .. import print_rank_0
import torch.distributed as dist

log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


class LoggerFactory:
    @staticmethod
    def create_logger(name=None, level=logging.INFO):
        """create a logger

        Args:
            name (str): name of the logger
            level: level of logger

        Raises:
            ValueError is name is None
        """

        if name is None:
            raise ValueError("name for logger cannot be None")

        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] "
            "[%(filename)s:%(lineno)d:%(funcName)s] %(message)s")

        logger_ = logging.getLogger(name)
        logger_.setLevel(level)
        logger_.propagate = False
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger_.addHandler(ch)
        return logger_


logger = LoggerFactory.create_logger(name="Merak", level=logging.INFO)


def log_dist(message, ranks=None, level=logging.INFO):
    """Log message when one of following condition meets

    + not dist.is_initialized()
    + dist.get_rank() in ranks if ranks is not None or ranks = [-1]

    Args:
        message (str)
        ranks (list)
        level (int)

    """
    should_log = not dist.is_initialized()
    ranks = ranks or []
    my_rank = dist.get_rank() if dist.is_initialized() else -1
    if ranks and not should_log:
        should_log = ranks[0] == -1
        should_log = should_log or (my_rank in set(ranks))
    if should_log:
        final_message = "[Rank {}] {}".format(my_rank, message)
        logger.log(level, final_message)


def get_current_level():
    """
    Return logger's current log level
    """
    return logger.getEffectiveLevel()


def should_log_le(max_log_level_str):
    """
    Args:
        max_log_level_str: maximum log level as a string

    Returns ``True`` if the current log_level is less or equal to the specified log level. Otherwise ``False``.

    Example:

        ``should_log_le("info")`` will return ``True`` if the current log level is either ``logging.INFO`` or ``logging.DEBUG``
    """

    if not isinstance(max_log_level_str, str):
        raise ValueError(f"{max_log_level_str} is not a string")

    max_log_level_str = max_log_level_str.lower()
    if max_log_level_str not in log_levels:
        raise ValueError(f"{max_log_level_str} is not one of the `logging` levels")

    return get_current_level() <= log_levels[max_log_level_str]

class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += val
        self.n += 1

    def reset(self):
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    @property
    def avg(self):
        return self.sum / self.n

class AccMetric(object):
    def __init__(self):
        self.metrics = {}

    def update(self, key, val):
        if key in self.metrics:
            self.metrics[key].update(val)
        else:
            self.metrics[key] = Metric(key)
            self.metrics[key].update(val)  

    def reset(self):
        for key in self.metrics:
            self.metrics[key].reset()

    @property
    def avg(self):
        avg_dict = {}
        for key in self.metrics:
            avg_dict[key] = self.metrics[key].avg.item()
        self.reset()

        return avg_dict

