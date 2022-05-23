
___author__ = "(c) Antoine Grimaldi & Laurent Perrinet INT - CNRS (2020-)"
__version__ = '1.0'
__licence__ = 'GPLv3'
__all__ = ['network.py', 'layer.py', 'timesurface.py', 'utils.py']
"""
========================================================
A Hierarchy of event-based Time-Surfaces for Pattern Recognition
========================================================
* This code aims to replicate the paper : 'HOTS : A Hierachy of Event Based Time-Surfaces for
Pattern Recognition' Xavier Lagorce, Garrick Orchard, Fransesco Gallupi, And Ryad Benosman'
"""

# TODO : include new names for scripts and code
from .layer import *
from .network import *
from .timesurface import *
from .utils import *