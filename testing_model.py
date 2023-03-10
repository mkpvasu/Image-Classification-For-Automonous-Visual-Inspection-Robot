from __future__ import print_function, division
import os
import json
import pandas as pd

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from training_model import ImagesAndLabels, SandingCanopyDataset, ModelTrain


class TestData:
    def __init__(self):
        super().__init__()

        # CHANGE TEST DATA PATH ACCORDING TO SET UP
        self.testDataPath = os.path.join(os.getcwd(), "data", "Dataset_Preparation", "DatasetForModel", "Test_Set", "30_micron")

    def testDataPrep(self):
        testImages = ImagesAndLabels(self.testDataPath).append_images_and_labels()
        testData = pd.DataFrame(testImages, columns=["ImageName", "Label"])
        return testData


class ModelTest:
    def __init__(self, batch_size=8):
        # SELECT GPU IF AVAILABLE ELSE RUN IN CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # DEFINE BATCH SIZE FOR DATALOADER
        self.batchSize = batch_size

        # DEFINE MICRON FOR MODEL TO BE TRAINED
        self.micron = "30_micron"

        # TRAINING DEEP LEARNING MODEL
        self.modelAttributes, self.trainedModel = ModelTrain(batch_size=self.batchSize, save_weights_path=os.path.join(os.getcwd(), "model_attributes", self.micron, "next")).outputTrainedModel()

    # TEST TRAINED MODEL WITH TESTING DATA AND TAKE THE TEST ACCURACY FOR FINAL MODEL PERFORMANCE
    def modelTestSetAccuracy(self):
        # CONVERT IMAGES AND LABELS TO LOADABLE DATASET FOR MODEL
        testDataToDataset = TestData().testDataPrep()
        testData = SandingCanopyDataset(testDataToDataset)
        testLoader = DataLoader(dataset=testData, batch_size=self.batchSize, shuffle=True, num_workers=2)

        testCorrects = 0.0
        testDataSize = len(testDataToDataset)

        # CHANGE FROM TRAINING MODE TO EVALUATION MODE
        self.trainedModel.eval()

        with torch.no_grad():
            for images, labels in testLoader:
                # TRANSFER IMAGES TO GPU IF AVAILABLE FOR PREDICTIONS
                images = images.to(self.device)

                # OUTPUT PROBABILITIES FOR ALL CLASSES
                outputs = self.trainedModel(images)

                # CLASS WITH THE HIGHEST PROBABILITY WILL BE TAKEN AS FINAL PREDICTION BY MODEL
                _, predictions = torch.max(outputs, dim=1)

                # SUM OF ALL CORRECT OUTPUTS
                testCorrects += torch.sum(predictions == labels).item()

                # F1 SCORE OF MODEL
                testF1Score = f1_score(labels.data, predictions, average=None)

        testAccuracy = (100 * (testCorrects / testDataSize))
        return testAccuracy, testF1Score

    def saveModelFeatures(self):

        modelAccuracy, modelF1Score = self.modelTestSetAccuracy()

        performanceAttributes = self.modelAttributes
        # IMPORTANT FEATURES OF MODEL TO BE SAVED
        performanceAttributes["model_accuracy"] = modelAccuracy
        performanceAttributes["model_f1_score"] = modelF1Score

        with open(os.path.join(os.getcwd(), "model_attributes", "30_micron", "next_model", "bestweights",
                               "performance.json")) as saveFile:
            json.dump(performanceAttributes, saveFile, indent=2)


def main():
    ModelTest().saveModelFeatures()


if __name__ == "__main__":
    main()