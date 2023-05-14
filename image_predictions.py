from __future__ import print_function, division
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

class ImagePredictions:

    # MAKE PREDICTIONS BY THE TRAINED MODEL
    def makePredictions(self):
        with torch.no_grad():
            # MAKE PREDICTIONS BY FORWARD PASSING IMAGES
            predictions = self.trainedModel(self.testData)