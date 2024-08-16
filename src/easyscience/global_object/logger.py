#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

__author__ = "github.com/wardsimon"
__version__ = "0.1.0"

import logging


class Logger:
    def __init__(self, log_level: int = logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.level = log_level
        self.logger.setLevel(self.level)

    def getLogger(
        self, logger_name, color: str = "32", defaults: bool = True
    ) -> logging:
        """
        Create a logger
        :param color:
        :param logger_name: logger name. Usually __name__ on creation
        :param defaults: Do you want to associate any current file loggers with this logger
        :return: A logger
        """
        logger = logging.getLogger(logger_name)
        logger.setLevel(self.level)
        # self.applyLevel(logger)
        # for handler_type in self._handlers:
        #     for handler in self._handlers[handler_type]:
        #         if handler_type == 'sys' or defaults:
        #             handler.formatter._fmt = self._makeColorText(color)
        #             logger.addHandler(handler)
        # logger.propagate = False
        # self._loggers.append(logger)
        return logger
