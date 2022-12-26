# IMPORTING NECESSARY LIBRARIES
import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as util
import torchvision


class TrainValTestData():
    def __init__(self):
        super(TrainValTestData, self).__init__()
        self.trainRatio = 0.7
        self.valRatio = 0.15
        self.testRatio = 0.15

    def trainValTestSplit(self):
        trainData, valData, testData = util.random_split()


def main():
    b = 5


if __name__ == "__main__":
    main()