#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience


from easyscience.Objects.ObjectClasses import BaseObj


class TheoreticalModelBase(BaseObj):
    """
    This virtual class allows for the creation of technique-specific Theory objects.
    """
    def __init__(self, name: str, *args, **kwargs):
        self._name = name
        super().__init__(name, *args, **kwargs)

    # required dunder methods
    def __str__(self):
        raise NotImplementedError("Copy not implemented")
    
    def as_dict(self, skip: list = []) -> dict:
        this_dict = super().as_dict(skip=skip)
        return this_dict

