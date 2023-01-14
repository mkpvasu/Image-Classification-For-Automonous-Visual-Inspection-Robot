from __future__ import print_function, division
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from training_model import SandingCanopyDataset, TrainingModel


class TestData:
    def __init__(self):
        super().__init__()

        # CHANGE TEST DATA PATH ACCORDING TO SET UP
        self.testDataPath = os.path.join(os.getcwd(), "data", "dataset_for_model", "test_data")

    def testDataPrep(self):
        testImages = glob.glob(os.path.join(self.testDataPath, "**", "*.jpg"))
        testData = pd.DataFrame(testImages, columns=["ImageName", "Label"])
        return testImages


class TestingModel:
    def __init__(self, batch_size=8):

        # SELECT GPU IF AVAILABLE ELSE RUN IN CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # DEFINE BATCH SIZE FOR DATALOADER
        self.batchSize = batch_size

        # CONVERT IMAGES AND LABELS TO LOADABLE DATASET FOR MODEL
        self.testDataToDataset = TestData.testDataPrep()
        self.testData = SandingCanopyDataset(self.testDataToDataset)
        self.testLoader = DataLoader(dataset=self.testData, batch_size=self.batchSize, shuffle=True, num_workers=2)

        # TRAINING DEEP LEARNING MODEL
        self.trainedModel = TrainingModel(batch_size=16).outputTrainedModel()

    # TEST TRAINED MODEL WITH TESTING DATA AND TAKE THE TEST ACCURACY FOR FINAL MODEL PERFORMANCE
    def testAccuracy(self):
        testCorrects = 0.0
        testDataSize = len(self.testDataToDataset)

        # CHANGE FROM TRAINING MODE TO EVALUATION MODE
        self.trainedModel.eval()

        with torch.no_grad():
            for images, labels in self.testLoader:
                # TRANSFER IMAGES TO GPU IF AVAILABLE FOR PREDICTIONS
                images = images.to(self.device)

                # OUTPUT PROBABILITIES FOR ALL CLASSES
                outputs = self.trainedModel(images)

                # CLASS WITH THE HIGHEST PROBABILITY WILL BE TAKEN AS FINAL PREDICTION BY MODEL
                _, predictions = torch.max(outputs, dim=1)

                # SUM OF ALL CORRECT OUTPUTS
                testCorrects += torch.sum(predictions == labels).item()

        testAccuracy = (100 * (testCorrects / testDataSize))
        return testAccuracy


def main():
    modelPerformance = TestingModel().testAccuracy()


if __name__ == "__main__":
    main()



