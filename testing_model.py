from __future__ import print_function, division
import os
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
        testImages = ImagesAndLabels(self.testDataPath).appendImagesAndLabels()
        testData = pd.DataFrame(testImages, columns=["ImageName", "Label"])
        return testData


class ModelTest:
    def __init__(self, batch_size=8):

        # SELECT GPU IF AVAILABLE ELSE RUN IN CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # DEFINE BATCH SIZE FOR DATALOADER
        self.batchSize = batch_size

        # CONVERT IMAGES AND LABELS TO LOADABLE DATASET FOR MODEL
        self.testDataToDataset = TestData().testDataPrep()
        self.testData = SandingCanopyDataset(self.testDataToDataset)
        self.testLoader = DataLoader(dataset=self.testData, batch_size=self.batchSize, shuffle=True, num_workers=2)

        # TRAINING DEEP LEARNING MODEL
        self.trainedModel = ModelTrain(batch_size=self.batchSize, save_weights_path=os.path.join(os.getcwd(), "model_attributes", "30_micron")).outputTrainedModel(n_epochs=3)

    # TEST TRAINED MODEL WITH TESTING DATA AND TAKE THE TEST ACCURACY FOR FINAL MODEL PERFORMANCE
    def modelTestSetAccuracy(self):
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

                # F1 SCORE OF MODEL
                testF1Score = f1_score(labels.data, predictions, average=None)

        testAccuracy = (100 * (testCorrects / testDataSize))
        return testAccuracy, testF1Score


def main():
    modelAccuracy, modelF1Score = ModelTest().modelTestSetAccuracy()
    print(modelAccuracy, modelF1Score)


if __name__ == "__main__":
    main()



