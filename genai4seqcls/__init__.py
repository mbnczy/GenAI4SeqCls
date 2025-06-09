__version__ = "1.0.1"
__author__ = 'Martin Balázs Bánóczy'

import os
os.environ["UNSLOTH_IS_PRESENT"] = "1"
os.environ["WANDB_WATCH"]="all"
os.environ["WANDB_SILENT"]="true"
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
os.environ["CUDA_LAUNCH_BLOCKING"]="1"

from .models import *
from .metrics import *
from .callbacks import *

#import .models
#import .metrics
#import .callbacks
