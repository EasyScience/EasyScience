import scipp as sc

# This document shows how to define aliases for units in scipp,
# and how to overwrite existing aliases.

sc.units.aliases.clear() # Clear existing aliases
sc.units.aliases['funny_unit'] = sc.scalar(42.0, unit='m')
# ^ 'funny_unit' is now an alias for 42 m and can be used as a unit in scipp and be converted to units of dimension 'm'

# The representation of a unit can be changed by defining an alias for it:
sc.units.aliases['m/ms'] = 'm/ms'
# Setting this alias ensures that 'm/ms' is displayed as 'm/ms' instead of the default 'km/s'