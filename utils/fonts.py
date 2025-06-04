
from itertools import cycle
import matplotlib.colors as mcolors
import os
import matplotlib.font_manager as fm
import numpy as np
import sys 

blaa, blaa1, blaa2, blaa3, blaa4, sand, sand1, roed, roed1, roed2 = '#14143c','#434363','#72728a','#a1a1b1','#d0d0d8','#d1c5c3','#e8e2e1','#df0515','#ec6973','#f5b4b9'

def interpolate_colors(color1, color2, num_colors=5):
    cmap = mcolors.LinearSegmentedColormap.from_list("", [color1, color2])
    return [cmap(x) for x in np.linspace(0, 1, num_colors)]
original_colors = ['#14143c','#434363','#72728a','#a1a1b1','#d0d0d8','#d1c5c3']
extended_colors = []
for i in range(len(original_colors) - 1):
    extended_colors.extend(interpolate_colors(original_colors[i], original_colors[i + 1]))
extended_colors.append(original_colors[-1])

# Use the default matplotlib font. The previous implementation loaded a
# custom font from the ``resources`` directory. As custom fonts are no
# longer used, ``prop`` simply holds an empty ``FontProperties`` instance
# so that the rest of the code can keep passing ``fontproperties=prop``
# without modification.
prop = fm.FontProperties()
