# IMPORTING NECESSARY LIBRARIES
import os
import glob
import torch
from torch.utils import data


class TrainValTestData():
    def __init__(self, datasetPath: list):
        super(TrainValTestData, self).__init__()
        self.dataset = glob.glob(os.path.join(datasetPath, '*.jpg'))
        self.trainRatio = 0.7
        self.valRatio = 0.15
        self.testRatio = 0.15

    def trainValTestSplit(self):
        trainData, valData, testData = data.random_split(self.dataset, [self.trainRatio, self.valRatio, self.testRatio])
        return trainData, valData, testData


def main():
    entireDatasetPath =os.path.join(os.getcwd(), 'data', 'partitioned_images')
    trainData, valData, testData = TrainValTestData(entireDatasetPath).trainValTestSplit()
    print(len(trainData))
    print(len(valData))
    print(len(testData))


if __name__ == "__main__":
    main()