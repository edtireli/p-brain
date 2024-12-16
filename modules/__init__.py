"""
p-Brain: Advanced Neuroimaging Analysis Tool

This package provides functionalities for in-depth analysis of .PAR/.REC MRI data, including...
"""

__version__ = "1.2.1"
__author__ = "Edis Devin Tireli"
__affiliation__ = "Copenhagen University"

# modules/__init__.py
from utils.fonts import *
from utils.parameters import * #Change global parameters here!! Important!
from utils.settings import *
from utils.loading import *
from .start import * # welcome screen
from .opt00_images import * # option 0
from .opt01_T1_fit import * # option 1
from .opt02_input_functions import * # option 2
from utils.mapping import *
from utils.plotting import *
from .opt03_time_shifting import * # option 3
from .opt04_tissue_function import * # option 4
from .opt05_BBB_parameters import * # option 5
from .opt06_analysis_notes import * # option 6
from .opt07_axials import * # option 8
<<<<<<< HEAD

from .AI_input_functions import * # AI module 1 (input functions)
from .AI_tissue_functions import * # AI module 2 (tissue functions)


=======
>>>>>>> a0de673fc033368a127dc6bae55e4b3363958e21

