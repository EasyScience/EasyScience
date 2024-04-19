from __future__ import annotations

__author__ = 'https://github.com/materialsvirtuallab/monty/blob/master/monty/json.py'
__version__ = '3.0.0'

#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience


import json
from typing import TYPE_CHECKING
from typing import List

import numpy as np

from .template import BaseEncoderDecoder

if TYPE_CHECKING:
    from easyscience.Objects.ObjectClasses import BV

_KNOWN_CORE_TYPES = ('Descriptor', 'Parameter')


class JsonSerializer(BaseEncoderDecoder):
    def encode(self, obj: BV, skip: List[str] = []) -> str:
        """
        Returns a json string representation of the ComponentSerializer object.
        """
        ENCODER = type(
            JsonEncoderTemplate.__name__,
            (JsonEncoderTemplate, BaseEncoderDecoder),
            {'skip': skip},
        )
        return json.dumps(obj, cls=ENCODER)

    @classmethod
    def decode(cls, data: str) -> BV:
        return json.loads(data, cls=JsonDecoderTemplate)


class JsonDataSerializer(BaseEncoderDecoder):
    def encode(self, obj: BV, skip: List[str] = []) -> str:
        """
        Returns a json string representation of the ComponentSerializer object.
        """
        from easyscience.Utils.io.dict import DataDictSerializer

        ENCODER = type(
            JsonEncoderTemplate.__name__,
            (JsonEncoderTemplate, BaseEncoderDecoder),
            {
                'skip': skip,
                '_converter': lambda *args, **kwargs: DataDictSerializer._parse_dict(
                    DataDictSerializer._convert_to_dict(*args, **kwargs)
                ),
            },
        )

        return json.dumps(obj, cls=ENCODER)

    @classmethod
    def decode(cls, data: str) -> BV:
        raise NotImplementedError('It is not possible to reconstitute objects from data only objects.')


class JsonEncoderTemplate(json.JSONEncoder):
    """
    A Json Encoder which supports the ComponentSerializer API, plus adds support for
    numpy arrays, datetime objects, bson ObjectIds (requires bson).

    Usage::

        # Add it as a *cls* keyword when using json.dump
        json.dumps(object, cls=MontyEncoder)
    """

    skip = []
    _converter = BaseEncoderDecoder._convert_to_dict

    def default(self, o) -> dict:  # pylint: disable=E0202
        """
        Overriding default method for JSON encoding. This method does two
        things: (a) If an object has a to_dict property, return the to_dict
        output. (b) If the @module and @class keys are not in the to_dict,
        add them to the output automatically. If the object has no to_dict
        property, the default Python json encoder default method is called.

        Args:
            o: Python object.

        Return:
            Python dict representation.
        """
        return self._converter(o, self.skip, full_encode=True)


class JsonDecoderTemplate(json.JSONDecoder):
    """
    A Json Decoder which supports the ComponentSerializer API. By default, the
    decoder attempts to find a module and name associated with a dict. If
    found, the decoder will generate a Pymatgen as a priority.  If that fails,
    the original decoded dictionary from the string is returned. Note that
    nested lists and dicts containing pymatgen object will be decoded correctly
    as well.

    Usage:

        # Add it as a *cls* keyword when using json.load
        json.loads(json_string, cls=MontyDecoder)
    """

    _converter = BaseEncoderDecoder._convert_from_dict

    def decode(self, s):
        """
        Overrides decode from JSONDecoder.

        :param s: string
        :return: Object.
        """
        d = json.JSONDecoder.decode(self, s)
        return self.__class__._converter(d)


def jsanitize(obj, strict=False, allow_bson=False):
    """
    This method cleans an input json-like object, either a list or a dict or
    some sequence, nested or otherwise, by converting all non-string
    dictionary keys (such as int and float) to strings, and also recursively
    encodes all objects using Monty's as_dict() protocol.

    Args:
        obj: input json-like object.
        strict (bool): This parameters sets the behavior when jsanitize
            encounters an object it does not understand. If strict is True,
            jsanitize will try to get the as_dict() attribute of the object. If
            no such attribute is found, an attribute error will be thrown. If
            strict is False, jsanitize will simply call str(object) to convert
            the object to a string representation.
        allow_bson (bool): This parameters sets the behavior when jsanitize
            encounters an bson supported type such as objectid and datetime. If
            True, such bson types will be ignored, allowing for proper
            insertion into MongoDb databases.

    Returns:
        Sanitized dict that can be json serialized.
    """
    # if allow_bson and (
    #     isinstance(obj, (datetime.datetime, bytes))
    #     or (bson is not None and isinstance(obj, bson.objectid.ObjectId))
    # ):
    #     return obj
    if isinstance(obj, (list, tuple)):
        return [jsanitize(i, strict=strict, allow_bson=allow_bson) for i in obj]
    if np is not None and isinstance(obj, np.ndarray):
        return [jsanitize(i, strict=strict, allow_bson=allow_bson) for i in obj.tolist()]
    if isinstance(obj, dict):
        return {k.__str__(): jsanitize(v, strict=strict, allow_bson=allow_bson) for k, v in obj.items()}
    if isinstance(obj, (int, float)):
        return obj
    if obj is None:
        return None

    if not strict:
        return obj.__str__()

    if isinstance(obj, str):
        return obj.__str__()

    return jsanitize(obj.as_dict(), strict=strict, allow_bson=allow_bson)
