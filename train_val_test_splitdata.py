# IMPORTING NECESSARY LIBRARIES
import os
import glob
import torch
import shutil
from torch.utils import data


class TrainValTestDataCreation:
    def __init__(self):
        super(TrainValTestDataCreation, self).__init__()
        self.datasetPath = os.path.join(os.getcwd(), "data", "dataset")
        self.finalDatasetPath = os.path.join(os.getcwd(), "data", "dataset_for_model")
        self.subDirs = os.listdir(self.datasetPath)
        self.trainRatio = 0.85
        self.testRatio = 0.15

    def trainAndTestSplit(self):
        if os.path.exists(self.finalDatasetPath):
            shutil.rmtree(self.finalDatasetPath)
        os.mkdir(self.finalDatasetPath)

        trainDataPath = os.path.join(self.finalDatasetPath, "train_data")
        testDataPath = os.path.join(self.finalDatasetPath, "test_data")

        for dataDir in [trainDataPath, testDataPath]:
            os.mkdir(dataDir)

        for subDir in self.subDirs:
            subDirImages = glob.glob(os.path.join(self.datasetPath, subDir, "*.jpg"))
            trainSplitData, testSplitData = data.random_split(subDirImages, [self.trainRatio, self.testRatio])

            trainDataFolder = os.path.join(trainDataPath, subDir)
            os.mkdir(trainDataFolder)
            for image in trainSplitData:
                shutil.copy(image, trainDataFolder)

            testDataFolder = os.path.join(testDataPath, subDir)
            os.mkdir(testDataFolder)
            for image in testSplitData:
                shutil.copy(image, testDataFolder)

def main():
    TrainValTestDataCreation().trainAndTestSplit()


if __name__ == "__main__":
    main()