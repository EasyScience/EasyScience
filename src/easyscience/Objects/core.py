#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

from __future__ import annotations

__author__ = 'github.com/wardsimon'
__version__ = '0.0.1'

import json
from collections import OrderedDict
from hashlib import sha1
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from easyscience.Utils.io.dict import DataDictSerializer
from easyscience.Utils.io.dict import DictSerializer
from easyscience.Utils.io.json import jsanitize

if TYPE_CHECKING:
    from easyscience.Utils.io.template import EC


class ComponentSerializer:
    """
    This is the base class for all EasyScience objects and deals with the data conversion to other formats via the `encode`
    and `decode` functions. Shortcuts for dictionary and data dictionary encoding is also present.
    """

    _CORE = True

    def __deepcopy__(self, memo):
        return self.from_dict(self.as_dict())

    def encode(self, skip: Optional[List[str]] = None, encoder: Optional[EC] = None, **kwargs) -> Any:
        """
        Use an encoder to covert an EasyScience object into another format. Default is to a dictionary using `DictSerializer`.

        :param skip: List of field names as strings to skip when forming the encoded object
        :param encoder: The encoder to be used for encoding the data. Default is `DictSerializer`
        :param kwargs: Any additional key word arguments to be passed to the encoder
        :return: encoded object containing all information to reform an EasyScience object.
        """
        if encoder is None:
            encoder = DictSerializer
        encoder_obj = encoder()
        return encoder_obj.encode(self, skip=skip, **kwargs)

    @classmethod
    def decode(cls, obj: Any, decoder: Optional[EC] = None) -> Any:
        """
        Re-create an EasyScience object from the output of an encoder. The default decoder is `DictSerializer`.

        :param obj: encoded EasyScience object
        :param decoder: decoder to be used to reform the EasyScience object
        :return: Reformed EasyScience object
        """

        if decoder is None:
            decoder = DictSerializer
        return decoder.decode(obj)

    def as_dict(self, skip: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Convert an EasyScience object into a full dictionary using `DictSerializer`.
        This is a shortcut for ```obj.encode(encoder=DictSerializer)```

        :param skip: List of field names as strings to skip when forming the dictionary
        :return: encoded object containing all information to reform an EasyScience object.
        """

        return self.encode(skip=skip, encoder=DictSerializer)

    @classmethod
    def from_dict(cls, obj_dict: Dict[str, Any]) -> None:
        """
        Re-create an EasyScience object from a full encoded dictionary.

        :param obj_dict: dictionary containing the serialized contents (from `DictSerializer`) of an EasyScience object
        :return: Reformed EasyScience object
        """

        return cls.decode(obj_dict, decoder=DictSerializer)

    def encode_data(self, skip: Optional[List[str]] = None, encoder: Optional[EC] = None, **kwargs) -> Any:
        """
        Returns just the data in an EasyScience object win the format specified by an encoder.

        :param skip: List of field names as strings to skip when forming the dictionary
        :param encoder: The encoder to be used for encoding the data. Default is `DataDictSerializer`
        :param kwargs: Any additional keywords to pass to the encoder when encoding
        :return: encoded object containing just the data of an EasyScience object.
        """

        if encoder is None:
            encoder = DataDictSerializer
        return self.encode(skip=skip, encoder=encoder, **kwargs)

    def as_data_dict(self, skip: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Returns a dictionary containing just the data of an EasyScience object.

        :param skip: List of field names as strings to skip when forming the dictionary
        :return: dictionary containing just the data of an EasyScience object.
        """

        return self.encode(skip=skip, encoder=DataDictSerializer)

    def unsafe_hash(self) -> sha1:
        """
        Returns an hash of the current object. This uses a generic but low
        performance method of converting the object to a dictionary, flattening
        any nested keys, and then performing a hash on the resulting object
        """

        def flatten(obj, seperator='.'):
            # Flattens a dictionary

            flat_dict = {}
            for key, value in obj.items():
                if isinstance(value, dict):
                    flat_dict.update({seperator.join([key, _key]): _value for _key, _value in flatten(value).items()})
                elif isinstance(value, list):
                    list_dict = {'{}{}{}'.format(key, seperator, num): item for num, item in enumerate(value)}
                    flat_dict.update(flatten(list_dict))
                else:
                    flat_dict[key] = value

            return flat_dict

        ordered_keys = sorted(flatten(jsanitize(self.as_dict())).items(), key=lambda x: x[0])
        ordered_keys = [item for item in ordered_keys if '@' not in item[0]]
        return sha1(json.dumps(OrderedDict(ordered_keys)).encode('utf-8'))  # noqa: S324
