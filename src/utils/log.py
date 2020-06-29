"""Implementation of customized logging configs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import logging.config

__all__ = ['configure_logging']


def configure_logging(level=logging.INFO, file=None, mode='w', format=None, datefmt=None,
                      root_handler_type=1):
    """Configures logging.

    `console` can write colored messages to console will be added.
    If specified `file`, the following loggers will be added:
            `file` that write messages to specified files.

    # simplified code:
    ```
    logging.basicConfig(level=level,
                        format='%(asctime)s %(pathname)s:%(lineno)s %(message)s',
                        handlers=[logging.FileHandler(file, mode='w'),
                                logging.StreamHandler()])
    ```

    Args:
        level (int or string, optional): Logging level, include 'CRITICAL', 'ERROR',
            'WARNING', 'INFO', 'DEBUG', 'NOTSET'.
        file (string, optional): Path to log file. If specified, will add loggers that
            can write message to file.
        mode (string, optional): Specify the mode in which the logging file is opened.
            Default: `w`
        format (string, optional): Format of message.
        datefmt (string, optional): Format of date.
        root_handler_type (int, optional): 0: both console and file logging; 1: console
            logging only; 2: file logging only. Default: 1.
    """
    if root_handler_type in (0, 2) and file is None:
        raise ValueError('file should be specified when root_handler_type is 0 or 2')

    if format is None:
        format = '%(asctime)s %(filename)s:%(lineno)d[%(process)d] ' \
                 '%(levelname)s %(message)s'

    if datefmt is None:
        datefmt = '%Y-%m-%d %H:%M:%S.%f'

    basic_formatters = {
        'console_formatter': {
            '()': 'coloredlogs.ColoredFormatter',
            'format': format,
            'datefmt': datefmt
        }
    }
    basic_handlers = {
        'console_handler': {
            'class': 'logging.StreamHandler',
            'level': level,
            'formatter': 'console_formatter'
        }
    }

    extra_formatters = {}
    extra_handlers = {}

    if file is not None:
        extra_formatters = {
            'file_formatter': {
                'format': format,
                'datefmt': datefmt
            }
        }
        extra_handlers = {
            'file_handler': {
                'class': 'logging.FileHandler',
                'filename': file,
                'mode': mode,
                'level': level,
                'formatter': 'file_formatter'
            }
        }

    if root_handler_type == 0:
        root_handlers = ['console_handler', 'file_handler']
    elif root_handler_type == 1:
        root_handlers = ['console_handler']
    elif root_handler_type == 2:
        root_handlers = ['file_handler']
    else:
        raise ValueError('root_handler_type can only be 0, 1, 2, but got {}'
                         .format(root_handler_type))

    basic_formatters.update(extra_formatters)
    basic_handlers.update(extra_handlers)

    logging.config.dictConfig(dict(
        version=1,
        disable_existing_loggers=False,
        formatters=basic_formatters,
        handlers=basic_handlers,
        root={
            'level': level,
            'handlers': root_handlers,
        }
    ))
