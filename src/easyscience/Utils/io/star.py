#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

__author__ = 'github.com/wardsimon'
__version__ = '0.1.0'

import re
import textwrap
import warnings
from collections import OrderedDict
from collections import deque
from math import floor
from math import log10
from numbers import Number
from typing import List

# STANDARD CIF
# _MAX_LEN = 70
# _MAX_LABEL_LEN = 65

_MAX_LEN = 140
_MAX_LABEL_LEN = 130


class FakeItem:
    def __init__(self, value: float, error=None, fixed: bool = None):
        self.raw_value = value
        if fixed is not None:
            self.fixed = fixed
            self.error = 0
        if error is not None:
            self.error = error


class FakeCore:
    def __init__(self):
        self._kwargs = {}


class ItemHolder:
    def __init__(self, item, decimal_places: int = 8):
        self.maxlen = _MAX_LEN

        self.value = item.raw_value
        self.fixed = None
        self.error = None
        self.decimal_places = decimal_places
        if hasattr(item, 'fixed'):
            self.fixed = item.fixed
            if item.error != 0:
                self.error = item.error

    def _get_error_digits(self) -> int:
        return len(f'{round(self.error, self.decimal_places)}'.split('.')[-1])

    def __str__(self) -> str:
        if isinstance(self.value, Number):
            initial_str = '{:.' + str(self.decimal_places) + 'f}'
            s = initial_str.format(round(self.value, self.decimal_places))
            if self.error is not None:
                # x_exp = int(floor(log10(self.value)))
                xe_exp = int(floor(log10(self.error)))

                # uncertainty
                un_exp = xe_exp - self.decimal_places + 1
                un_int = round(self.error * 10 ** (-un_exp))

                # nominal value
                no_exp = un_exp
                no_int = round(self.value * 10 ** (-no_exp))

                # format - nom(unc)
                fmt = '%%.%df' % max(0, -no_exp)
                s = (fmt + '(%.0f)') % (
                    no_int * 10**no_exp,
                    un_int * 10 ** max(0, un_exp),
                )
        elif isinstance(self.value, str):
            s = '{:s}'.format(self.value)
        else:
            s = '{:s}'.format(str(self.value))
        # THIS IS THE OLD CODE, KEPT FOR REFERENCE
        # s = "{}"
        # if isinstance(self.value, str):
        #     s = "{:s}".format(self.value)
        # else:
        #     v_in = [round(self.value, self.decimal_places)]
        #     if self.error is not None:
        #         digits = self._get_error_digits()
        #         if digits > self.decimal_places:
        #             v_in = [round(self.value, digits)]
        #         this_err = int(self.error * 10 ** digits)
        #         v_in.append(this_err)
        #         digits = digits - (len(str(this_err)) - 1)
        #         s = "{" + f":0.0{digits}f" + "}({})"
        #     s = s.format(*v_in)
        if self.fixed is not None and not self.fixed and self.error is None:
            s += '()'
        return self._format_field(s)

    def _format_field(self, v):
        v = v.__str__().strip()
        if len(v) > self.maxlen:
            return ';\n' + textwrap.fill(v, self.maxlen) + '\n;'
        # add quotes if necessary
        if v == '':
            return '""'
        if (' ' in v or v[0] == '_') and not (v[0] == "'" and v[-1] == "'") and not (v[0] == '"' and v[-1] == '"'):
            if "'" in v:
                q = '"'
            else:
                q = "'"
            v = q + v + q
        return v

    @staticmethod
    def _makeFakeItem(in_string: str) -> FakeItem:
        in_string = in_string.strip()
        if "'" in in_string:
            in_string = in_string.replace("'", '')
        fixed = None
        error = None
        tokens = in_string.split('(')
        try:
            value = float(tokens[0])
        except ValueError:
            value = tokens[0]
            return FakeItem(value, error, fixed)
        if len(tokens) > 1:
            fixed = False
            if tokens[1][0] != ')':
                error = (10 ** -(len(f'{tokens[0]}'.split('.')[1]) + len(tokens[1][:-1]) - 1)) * int(tokens[1][:-1])
        return FakeItem(value, error, fixed)

    @classmethod
    def from_string(cls, in_string: str):
        return cls(cls._makeFakeItem(in_string))

    def to_fake_item(self):
        return FakeItem(self.value, self.error, self.fixed)


class StarEntry(ItemHolder):
    def __init__(self, item, entry_name: str = None, prefix='_'):
        if entry_name is None:
            entry_name = item.name

        if len(entry_name) > _MAX_LABEL_LEN:
            raise AttributeError(f'Max label length is {int(_MAX_LABEL_LEN)}')

        self.name = entry_name
        self.prefix = prefix
        super(StarEntry, self).__init__(item)

    def __str__(self) -> str:
        s = '{}{}   '.format(self.prefix, self.name) + super(StarEntry, self).__str__()
        return s

    @classmethod
    def from_string(cls, input_str: str, name_conversion: str = None, prefix='_'):
        name, value = input_str.split('   ')
        name = name[len(prefix) :]
        if name_conversion:
            name = name_conversion
        return cls(cls._makeFakeItem(value), name)

    def to_class(self, cls, name_conversion: str = None):
        if name_conversion is None:
            name_conversion = self.name
        if hasattr(cls, 'from_pars'):
            new_obj = cls.from_pars(**{name_conversion: self.value})
        if hasattr(new_obj, 'fixed'):
            if self.fixed is not None:
                new_obj.fixed = self.fixed
        if hasattr(new_obj, 'error'):
            if self.error is not None:
                new_obj.error = self.error
        return new_obj


class StarProcess:
    @classmethod
    def _process_string(cls, string):
        # remove comments
        string = re.sub(r'(\s|^)#.*$', '', string, flags=re.MULTILINE)
        # remove empty lines
        string = re.sub(r'^\s*\n', '', string, flags=re.MULTILINE)
        # remove non_ascii
        # string = remove_non_ascii(string)
        # since line breaks in .cif files are mostly meaningless,
        # break up into a stream of tokens to parse, rejoining multiline
        # strings (between semicolons)
        q = deque()
        multiline = False
        ml = []
        # this regex splits on spaces, except when in quotes.
        # starting quotes must not be preceded by non-whitespace
        # (these get eaten by the first expression)
        # ending quotes must not be followed by non-whitespace
        p = re.compile(r"""([^'"\s][\S]*)|'(.*?)'(?!\S)|"(.*?)"(?!\S)""")
        for line in string.splitlines():
            if multiline:
                if line.startswith(';'):
                    multiline = False
                    q.append(('', '', '', ' '.join(ml)))
                    ml = []
                    line = line[1:].strip()
                else:
                    ml.append(line)
                    continue
            if line.startswith(';'):
                multiline = True
                ml.append(line[1:].strip())
            else:
                for s in p.findall(line):
                    # s is tuple. location of the data in the tuple
                    # depends on whether it was quoted in the input
                    q.append(s)
        return q

    @classmethod
    def _loadBlock(cls, in_string: str, prefix='_'):
        q = cls._process_string(in_string)
        data = OrderedDict()
        loops = []
        while q:
            s = q.popleft()
            # cif keys aren't in quotes, so show up in s[0]
            if s[0] == '_eof':
                break
            if s[0].startswith(prefix):
                try:
                    data[s[0]] = ''.join(q.popleft())
                except IndexError:
                    data[s[0]] = ''
            elif s[0].startswith('loop_'):
                columns = []
                items = []
                this_data = OrderedDict()
                while q:
                    s = q[0]
                    if s[0].startswith('loop_') or not s[0].startswith(prefix):
                        break
                    columns.append(''.join(q.popleft()))
                    this_data[columns[-1]] = []
                while q:
                    s = q[0]
                    if s[0].startswith('loop_') or s[0].startswith(prefix):
                        break
                    items.append(''.join(q.popleft()))
                n = len(items) // len(columns)
                if len(items) % n != 0:
                    raise ValueError('Wrong dimensions in loop_ block')
                # loops.append(columns)
                for k, v in zip(columns * n, items):
                    this_data[k].append(v.strip())
                loops.append(this_data)
            elif ''.join(s).strip() != '':
                warnings.warn('Possible issue in cif file' ' at line: {}'.format(''.join(s).strip()))
        return data, loops


class StarBase(StarProcess):
    def __init__(
        self,
        core,
        entry_names: List[str] = None,
        exclude: list = None,
        prefix: str = '_',
    ):
        if not hasattr(core, '__len__'):
            core = [core]
        self.data = core
        if exclude is None:
            exclude = []
        self.exclude = exclude

        if self.data:
            if entry_names is None:
                entry_names = [self.data[0]._kwargs[key].name for key in self.data[0]._kwargs.keys() if key not in exclude]
        else:
            entry_names = []

        self.prefix = prefix
        self.labels = entry_names
        self.maxlen = _MAX_LEN


class StarCollection(StarProcess):
    def __init__(self, *star_objects):
        self.data = star_objects

    def __str__(self) -> str:
        return '\n\n'.join([str(data) for data in self.data])

    @classmethod
    def from_string(cls, in_string, prefix='_'):
        in_string = '\n'.join([item for item in in_string.split('\n') if item and item[0] != '#'])

        blocks = in_string.split('data_')
        data_blocks = []
        for block in blocks:
            if not block:
                continue
            data_block = {'header': None, 'loops': [], 'data': {}}
            items = block.split('\n')
            if len(items) == 0:
                continue
            data_block['header'] = StarHeader.from_string('data_' + items[0])
            data, loops = cls._loadBlock('\n'.join(items[1:]))
            for loop in loops:
                data_block['loops'].append(StarLoop.from_data(loop, prefix=prefix))
            for key in data.keys():
                entry = StarEntry.from_string('{}   {}'.format(key, data[key]))
                data_block['data'][entry.name] = entry
            data_blocks.append(data_block)
        if len(data_blocks) == 1:
            data_blocks = data_blocks[0]
        return data_blocks

    @classmethod
    def from_file(cls, filename: str):
        with open(filename, 'r') as reader:
            in_string = reader.read()
        if not in_string:
            in_string = filename
        return cls.from_string(in_string)


class StarSection(StarBase):
    def __str__(self) -> str:
        return self._section_to_string()

    def _section_to_string(self):
        s = ''
        keys = [key for key in self.data[0]._kwargs.keys() if key not in self.exclude]
        for idx, key in enumerate(keys):
            s += f'{StarEntry(self.data[0]._kwargs[key], self.labels[idx], prefix=self.prefix)}\n'
        return s

    def to_class(self, cls, name_conversions=None, skip=[]):
        if not hasattr(cls, 'from_pars'):
            raise AttributeError
        if name_conversions is None:
            name_conversions = [[k1, k2] for k1, k2 in zip(self.labels, self.data[0]._kwargs.keys())]
        new_object = cls.from_pars(**{k[0]: self.data[0]._kwargs[k[1]].raw_value for k in name_conversions})
        for key in name_conversions:
            attr = getattr(new_object, key[0])
            if hasattr(self.data[0]._kwargs[key[1]], 'fixed'):
                if self.data[0]._kwargs[key[1]].fixed is not None:
                    attr.fixed = self.data[0]._kwargs[key[1]].fixed
                if self.data[0]._kwargs[key[1]].error is not None:
                    attr.error = self.data[0]._kwargs[key[1]].error
        return new_object

    @classmethod
    def from_string(cls, in_string: str, name_conversion: List[str] = None, prefix='_'):
        items = in_string.split('\n')
        data = [FakeCore()]
        names = []
        for idx, item in enumerate(items):
            if not item:
                continue
            this_name = None
            if name_conversion is not None:
                this_name = name_conversion[idx]
            conv_item = StarEntry.from_string(item, name_conversion=this_name, prefix=prefix)
            names.append(conv_item.name)
            data[0]._kwargs[conv_item.name] = conv_item.to_fake_item()
        return cls(data, entry_names=names, prefix=prefix)

    @classmethod
    def from_StarEntries(
        cls,
        star_entries: List[StarEntry],
        name_conversions: List[str] = None,
        prefix='_',
    ):
        data = [FakeCore()]
        names = []
        for idx, entry in enumerate(star_entries):
            names.append(entry.name)
            data[0]._kwargs[entry.name] = entry.to_fake_item()
        if name_conversions is None:
            name_conversions = names
        return cls(data, entry_names=name_conversions, prefix=prefix)

    def to_StarEntries(self) -> List[StarEntry]:
        keys = [key for key in self.data[0]._kwargs.keys() if key not in self.exclude]
        return [StarEntry(self.data[0]._kwargs[key], self.labels[idx], prefix=self.prefix) for idx, key in enumerate(keys)]


class StarLoop(StarBase):
    def __str__(self) -> str:
        return self._loop_to_string()

    def _loop_to_string(self):
        s = 'loop_'
        if len(self.data) == 0:
            return ''
        keys = [key for key in self.data[0]._kwargs.keys() if key not in self.exclude]
        for idx, kw in enumerate(keys):
            label = kw
            if not isinstance(self.data[0]._kwargs[kw], FakeItem):
                label = self.data[0]._kwargs[kw].name
            if self.labels[idx] is not None:
                label = self.labels[idx]
            s += '\n {}'.format(self.prefix) + label
        for item in self.data:
            line = '\n'
            for kw in keys:
                val = str(ItemHolder(item._kwargs[kw]))
                if val.count(' ') == 0 and val.count("'") > 0:
                    val = val.strip("'")
                if val[0] == ';':
                    s += line + '\n' + val
                    line = '\n'
                elif len(line) + len(val) + 2 < self.maxlen:
                    line += '  ' + val
                else:
                    s += line
                    line = '\n  ' + val
            s += line
        return s

    @classmethod
    def from_string(cls, in_string: str, name_conversion: List[str] = None, prefix='_'):
        data, loops = cls._loadBlock(in_string, prefix=prefix)
        if len(loops) > 1:
            raise ValueError(f'String has more than one loop: {len(loops)}')
        return cls.from_data(loops[0], name_conversion, prefix)

    @classmethod
    def from_StarSections(
        cls,
        star_sections: List[StarSection],
        name_conversion: List[str] = None,
        prefix='_',
    ):
        this_data = [section.data[0] for section in star_sections]
        all_names = star_sections[0].labels
        if name_conversion is not None:
            all_names = name_conversion
        return cls(this_data, all_names, prefix=prefix)

    def to_StarSections(self) -> List[StarSection]:
        return [StarSection(section, self.labels, prefix=self.prefix) for section in self.data]

    @classmethod
    def from_data(cls, loop: dict, name_conversion: List[str] = None, prefix='_'):
        all_names = []
        all_data = []
        keys = list(loop.keys())
        for idx2 in range(len(loop[keys[0]])):
            fk = FakeCore()
            for idx, key in enumerate(keys):
                this_name = key
                if this_name[0] == '_':
                    this_name = this_name[1:]
                if name_conversion is not None:
                    this_name = name_conversion[idx]
                if idx2 == 0:
                    all_names.append(this_name)
                conv_item = StarEntry.from_string(
                    '{}{}   {}'.format(prefix, this_name, loop[key][idx2]),
                    this_name,
                    prefix=prefix,
                )
                fk._kwargs[conv_item.name] = conv_item.to_fake_item()
            all_data.append(fk)
        return cls(all_data, all_names, prefix=prefix)

    def to_class(self, cls_outer, cls_inner, name_conversions=None):
        if not hasattr(cls_inner, 'from_pars'):
            raise AttributeError
        new_objects = []
        for idx in range(len(self.data)):
            if name_conversions is None:
                keys = [key for key in self.data[idx]._kwargs.keys() if key not in self.exclude]
                name_conversions = [[k, k] for k in keys]
            new_object = cls_inner.from_pars(**{k[0]: self.data[idx]._kwargs[k[1]].raw_value for k in name_conversions})
            for key in name_conversions:
                attr = getattr(new_object, key[0])
                if hasattr(self.data[idx]._kwargs[key[1]], 'fixed'):
                    if self.data[idx]._kwargs[key[1]].fixed is not None:
                        attr.fixed = self.data[idx]._kwargs[key[1]].fixed
                    if self.data[idx]._kwargs[key[1]].error is not None:
                        attr.error = self.data[idx]._kwargs[key[1]].error
            new_objects.append(new_object)
        return cls_outer(cls_outer.__name__, *new_objects)

    def join(self, otherLoop: 'StarLoop', key: str) -> 'StarLoop':
        if key not in self.labels or key not in otherLoop.labels:
            raise AttributeError('Key must be common in both StarLoops')
        if len(self.data) != len(otherLoop.data):
            raise AttributeError('There must be the same number of entries in both StarLoops')
        joint = StarLoop.from_string(str(self))
        for dataset in otherLoop.data:
            lookup_value = dataset._kwargs['label'].raw_value
            try:
                lookup_idx = [d._kwargs['label'].raw_value for d in self.data].index(lookup_value)
            except ValueError:
                raise AttributeError('Both StarLoops must contain the joining same keys')
            joint.data[lookup_idx]._kwargs.update(dataset._kwargs)
        joint.labels.extend([k for k in otherLoop.labels if k != key])
        return joint


class StarHeader:
    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return 'data_' + self.name

    @classmethod
    def from_string(cls, in_string: str):
        name = in_string.split('data_')[1]
        return cls(name)
