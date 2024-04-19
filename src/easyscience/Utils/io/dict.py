from __future__ import annotations

__author__ = "https://github.com/materialsvirtuallab/monty/blob/master/monty/json.py"
__version__ = "3.0.0"
#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience


from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from easyscience.Utils.io.template import BaseEncoderDecoder

if TYPE_CHECKING:
    from easyscience.Objects.ObjectClasses import BV

_KNOWN_CORE_TYPES = ("Descriptor", "Parameter")


class DictSerializer(BaseEncoderDecoder):
    """
    This is a serializer that can encode and decode EasyScience objects to a JSON encoded dictionary.
    """

    def encode(
        self,
        obj: BV,
        skip: Optional[List[str]] = None,
        full_encode: bool = False,
        **kwargs,
    ):
        """
        Convert an EasyScience object to a JSON encoded dictionary

        :param obj: Object to be encoded.
        :param skip: List of field names as strings to skip when forming the encoded object
        :param full_encode: Should the data also be JSON encoded (default False)
        :param kwargs: Any additional key word arguments to be passed to the encoder
        :return: object encoded to dictionary containing all information to reform an EasyScience object.
        """

        return self._convert_to_dict(obj, skip=skip, full_encode=full_encode, **kwargs)

    @classmethod
    def decode(cls, d: Dict) -> BV:
        """
        :param d: Dict representation.
        :return: ComponentSerializer class.
        """

        return BaseEncoderDecoder._convert_from_dict(d)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> BV:
        """
        :param d: Dict representation.
        :return: ComponentSerializer class.
        """
        return BaseEncoderDecoder._convert_from_dict(d)


class DataDictSerializer(DictSerializer):
    """
    This is a serializer that can encode the data in an EasyScience object to a JSON encoded dictionary.
    """

    def encode(
        self,
        obj: BV,
        skip: Optional[List[str]] = None,
        full_encode: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Convert an EasyScience object to a JSON encoded data dictionary

        :param obj: Object to be encoded.
        :param skip: List of field names as strings to skip when forming the encoded object
        :param full_encode: Should the data also be JSON encoded (default False)
        :param kwargs: Any additional key word arguments to be passed to the encoder
        :return: object encoded to data dictionary.
        """

        if skip is None:
            skip = []
        elif isinstance(skip, str):
            skip = [skip]
        if not isinstance(skip, list):
            raise ValueError("Skip must be a list of strings.")
        encoded = super().encode(obj, skip=skip, full_encode=full_encode, **kwargs)
        return self._parse_dict(encoded)

    @classmethod
    def decode(cls, d: Dict[str, Any]) -> BV:
        """
        This function is not implemented as a data dictionary does not contain the necessary information to re-form an
        EasyScience object.
        """

        raise NotImplementedError(
            "It is not possible to reconstitute objects from data only dictionary."
        )

    @staticmethod
    def _parse_dict(in_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Strip out any non-data from a dictionary
        """

        out_dict = dict()
        for key in in_dict.keys():
            if key[0] == "@":
                if key == "@class" and in_dict[key] not in _KNOWN_CORE_TYPES:
                    out_dict["name"] = in_dict[key]
                continue
            out_dict[key] = in_dict[key]
            if isinstance(in_dict[key], dict):
                out_dict[key] = DataDictSerializer._parse_dict(in_dict[key])
            elif isinstance(in_dict[key], list):
                out_dict[key] = [
                    DataDictSerializer._parse_dict(x) if isinstance(x, dict) else x
                    for x in in_dict[key]
                ]
        return out_dict
