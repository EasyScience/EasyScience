# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta
from abc import abstractmethod
from typing import Any
from typing import Optional

import numpy as np

from ..base_classes.model_base import ModelBase
from ..fitting.minimizers import MinimizerBase


class AnalysisBase(ModelBase, metaclass=ABCMeta):
    """
    This virtual class allows for the creation of technique-specific
    Analysis objects.
    """

    def __init__(
        self,
        *,
        display_name: Optional[str] = None,
        unique_name: Optional[str] = None,
    ):
        super(AnalysisBase, self).__init__(unique_name=unique_name, display_name=display_name)
        self._minimizer = None

    @abstractmethod
    def calculate_theory(self, x: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError('calculate_theory not implemented')

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, e: np.ndarray, **kwargs) -> None:
        raise NotImplementedError('fit not implemented')

    @property
    def minimizer(self) -> MinimizerBase:
        return self._minimizer

    @minimizer.setter
    def minimizer(self, minimizer: MinimizerBase) -> None:
        self._minimizer = minimizer

    # required dunder methods
    def __str__(self):
        return f'Analysis: {self.display_name}'
