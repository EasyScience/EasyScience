"""
Constraints example
===================
This  example shows the usages of the different constraints.
"""

from easyscience import Constraints
from easyscience.Objects.ObjectClasses import Parameter

p1 = Parameter('p1', 1)
constraint = Constraints.NumericConstraint(p1, '<', 5)
p1.user_constraints['c1'] = constraint

for value in range(4, 7):
    p1.value = value
    print(f'Set Value: {value}, Parameter Value: {p1}')

# %%
# To include embedded rST, use a line of >= 20 ``#``'s or ``#%%`` between your
# rST and your code. This separates your example
# into distinct text and code blocks. You can continue writing code below the
# embedded rST text block:
