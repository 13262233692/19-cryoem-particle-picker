import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict

_loggers: Dict[str, logging.Logger] = {}

def setup_logger(name: str = "cryoem_picker", 
                 log_dir: str = "logs", 
                 log_level: int = logging.INFO,
                 console_output: bool = True) -> logging.Logger:
    if name in _loggers:
        return _loggers[name]
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(module)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, f"{name}.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    _loggers[name] = logger
    return logger

def get_logger(name: str = "cryoem_picker") -> logging.Logger:
    if name not in _loggers:
        return setup_logger(name)
    return _loggers[name]
