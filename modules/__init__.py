"""
p-Brain: Advanced Neuroimaging Analysis Tool

This package provides functionalities for in-depth analysis of .PAR/.REC MRI data, including...
"""

__version__ = "0.1.1"
__author__ = "Edis Devin Tireli"
__affiliation__ = "Copenhagen University"

# modules/__init__.py
from utils.fonts import *
from config import *
from utils.loading import *
from .start import * # welcome screen
from .images import * # option 0
from .T1_fit import * # option 1
from .input_functions import * # option 2
from utils.mapping import *
from utils.plotting import *
from .time_shifting import * # option 3
from .tissue_function import * # option 4
