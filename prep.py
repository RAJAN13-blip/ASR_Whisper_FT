import librosa
import librosa.display
import os
import soundfile as sf

from collections import Counter
import IPython
import os
from glob import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import gc
import warnings
warnings.filterwarnings('ignore')

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, level):
       self.logger = logger
       self.level = level
       self.linebuf = ''

    def write(self, buf):
       for line in buf.rstrip().splitlines():
          self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass