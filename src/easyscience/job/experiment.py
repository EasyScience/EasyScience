# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

from ..base_classes.model_base import ModelBase


class ExperimentBase(ModelBase):
    """
    This virtual class allows for the creation of technique-specific
    Experiment objects.
    """

    def __init__(self, display_name: Optional[str] = None, unique_name: Optional[str] = None):
        super(ExperimentBase, self).__init__(unique_name=unique_name, display_name=display_name)

    # required dunder methods
    def __str__(self):
        return f'Experiment: {self.display_name}'
