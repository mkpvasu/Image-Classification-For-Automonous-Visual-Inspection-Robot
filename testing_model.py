from __future__ import print_function, division
import os
import json
import pandas as pd
import datetime as dt

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from training_model import ImagesAndLabels, SandingCanopyDataset, ModelTrain


class ModelTest:
    def __init__(self, micron="30_micron", batch_size=8):
        # SELECT GPU IF AVAILABLE ELSE RUN IN CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # DEFINE BATCH SIZE FOR DATALOADER
        self.batchSize = batch_size

        # DEFINE MICRON FOR MODEL TO BE TRAINED
        self.micron = micron

        # CHANGE TEST DATA PATH ACCORDING TO SET UP
        self.testDataPath = os.path.join(os.getcwd(), "data", "dataset_preparation", "dataset_for_model", "test_set",
                                         self.micron)

        # CREATE A NEW TRAINING FOLDER EVERYTIME FOR TRAINING
        self.trainingFolder = "model_training_" + dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        os.mkdir(os.path.join(os.getcwd(), "model_attributes", self.micron, "next", self.trainingFolder))

        # DIRECTORY TO SAVE MODEL ATTRIBUTES
        self.save_model_attributes_path = os.path.join(os.getcwd(), "model_attributes", self.micron, "next",
                                                       self.trainingFolder)

        # TRAIN DEEP LEARNING MODEL
        self.modelAttributes, self.trainedModel = \
            ModelTrain(batch_size=self.batchSize, save_model_attributes_path=self.save_model_attributes_path)\
                .output_trained_model()

    # TEST TRAINED MODEL WITH TESTING DATA AND TAKE THE TEST ACCURACY FOR FINAL MODEL PERFORMANCE
    def model_test_set_accuracy(self):
        # CONVERT IMAGES AND LABELS TO LOADABLE DATASET FOR MODEL
        testDataToDataset = self.test_data_prep()
        testData = SandingCanopyDataset(testDataToDataset)
        testLoader = DataLoader(dataset=testData, batch_size=self.batchSize, shuffle=True, num_workers=2)

        testCorrects = 0.0
        testDataSize = len(testDataToDataset)

        # CHANGE FROM TRAINING MODE TO EVALUATION MODE
        self.trainedModel.eval()

        with torch.no_grad():
            for images, labels in testLoader:
                # TRANSFER IMAGES TO GPU IF AVAILABLE FOR PREDICTIONS
                images = (images - 127.5) / 127.5
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

    def save_model_features(self):
        modelAccuracy, modelF1Score = self.model_test_set_accuracy()

        performanceAttributes = self.modelAttributes
        # IMPORTANT FEATURES OF MODEL TO BE SAVED
        performanceAttributes["model_accuracy"] = modelAccuracy
        performanceAttributes["model_f1_score"] = modelF1Score

        with open(os.path.join(self.save_model_attributes_path, "performance.json"), "w") as saveFile:
            json.dump(performanceAttributes, saveFile, indent=2)

    def test_data_prep(self):
        testImages = ImagesAndLabels(self.testDataPath).append_images_and_labels()
        testData = pd.DataFrame(testImages, columns=["ImageName", "Label"])
        return testData


def main():
    ModelTest().save_model_features()


if __name__ == "__main__":
    main()