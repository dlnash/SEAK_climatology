"""
Filename:    customcmaps.py
Author:      Deanna Nash, dnash@ucsd.edu
Description: Functions for custom cmaps adapted from from https://github.com/samwisehawkins/nclcmaps
"""

import numpy as np
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

__all__ = ['cw3ecmaps']

colors = {"arscale": [[10, 193, 255], # blue
                            [4, 255, 3], # green
                            [255, 255, 3], # yellow
                            [255, 166, 2], # orange
                            [255, 1, 0]], # red
         
          "ivt": [[255, 255, 3], # 250-300
                  [255, 229, 3], # 300-400
                  [255, 201, 2], # 400-500
                  [255, 175, 2], # 500-600
                  [255, 131, 1], # 600-700
                  [255, 79, 1],  # 700-800
                  [255, 24, 1],  # 800-1000
                  [235, 1, 7],   # 1000-1200
                  # [185, 0, 55],  # 1200-1400
                  # [234, 234, 234], # 1400-1600
                  # [86, 0, 137]   # 1600+
          ]
         } 

bounds = {"arscale": [1, 2, 3, 4, 5],
          "ivt": [250, 300, 400, 500, 600, 700, 800, 1000, 1200]}


def cmap(name):
    data = np.array(colors[name])
    data = data / np.max(data)
    cmap = ListedColormap(data, name=name)
    bnds = bounds[name]
    norm = mcolors.BoundaryNorm(bnds, cmap.N)
    return cmap, norm, bnds