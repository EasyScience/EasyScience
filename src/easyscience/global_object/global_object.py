#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

__author__ = "github.com/wardsimon"
__version__ = "0.1.0"

from easyscience.Utils.classUtils import singleton

from .hugger.hugger import ScriptManager
from .logger import Logger
from .map import Map


@singleton
class GlobalObject:
    """
    GlobalObject is the assimilated knowledge of `EasyScience`. Every class based on `EasyScience` gets brought
    into the collective.
    """

    __log = Logger()
    __map = Map()
    __stack = None
    __debug = False

    def __init__(self):
        # Logger. This is so there's a unified logging interface
        self.log: Logger = self.__log
        # Debug. Global debugging level
        self.debug: bool = self.__debug
        # Stack. This is where the undo/redo operations are stored.
        self.stack = self.__stack
        #
        self.script: ScriptManager = ScriptManager()
        # Map. This is the conduit database between all global object species
        self.map: Map = self.__map

    def instantiate_stack(self):
        """
        The undo/redo stack references the collective. Hence it has to be imported
        after initialization.

        :return: None
        :rtype: noneType
        """
        from easyscience.global_object.undo_redo import UndoStack

        self.stack = UndoStack()
