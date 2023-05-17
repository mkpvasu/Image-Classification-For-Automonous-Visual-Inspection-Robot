from __future__ import print_function, division
import os
import json

import numpy as np
import pandas as pd
import datetime as dt

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from training_model import ImagesAndLabels, SandingCanopyDataset, ModelTrain


class ModelTest:
    """Class for testing the trained model"""
    def __init__(self, micron, batch_size=2, n_epochs=5, num_workers=2, freeze_weights=False):
        # SELECT GPU IF AVAILABLE ELSE RUN IN CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # DEFINE BATCH SIZE FOR DATALOADER
        self.batch_size = batch_size
        # DEFINE NUMBER OF WORKERS FOR DATALOADER
        self.num_workers = num_workers
        # DEFINE MICRON FOR MODEL TO BE TRAINED
        self.micron = micron
        # EPOCHS
        self.n_epochs = n_epochs
        # FREEZE WEIGHTS
        self.freeze_weights = freeze_weights

        # CHANGE TEST DATA PATH ACCORDING TO MICRON SIZE
        self.train_data_path = os.path.join(os.getcwd(), "data", "dataset_preparation", "dataset_for_model", 
                                            self.micron, "train_set")
        self.test_data_path = os.path.join(os.getcwd(), "data", "dataset_preparation", "dataset_for_model", 
                                           self.micron, "test_set")

        # CREATE A NEW TRAINING FOLDER EVERYTIME FOR TRAINING
        self.training_folder = "model_training_" + dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.create_training_dir()

        # DIRECTORY TO SAVE MODEL ATTRIBUTES
        self.save_model_attributes_path = os.path.join(os.getcwd(), "model_attributes", self.micron, "next",
                                                       self.training_folder)

        # TRAIN DEEP LEARNING MODEL
        self.model_attributes, self.trained_model = \
            ModelTrain(train_data_path=self.train_data_path, save_model_attributes_path=self.save_model_attributes_path,
                       micron=self.micron, n_epochs=self.n_epochs, batch_size=self.batch_size,
                       freeze_weights=self.freeze_weights).output_trained_model()

    def create_training_dir(self):
        """
        Creates a new directory in the training_dir folder path at each time of training to save the model features
        """
        training_dir = os.path.join(os.getcwd(), "model_attributes", self.micron, "next", self.training_folder)
        if not os.path.exists(training_dir):
            os.makedirs(training_dir)

    # TEST TRAINED MODEL WITH TESTING DATA AND TAKE THE TEST ACCURACY FOR FINAL MODEL PERFORMANCE
    def model_test_set_accuracy(self):
        """Calculate the test accuracy and f1 score for the trained model"""
        # CONVERT IMAGES AND LABELS TO LOADABLE DATASET FOR MODEL
        test_data_to_dataset = self.test_data_prep()
        test_data = SandingCanopyDataset(test_data_to_dataset)
        test_loader = DataLoader(dataset=test_data, batch_size=self.batch_size, shuffle=True,
                                 num_workers=self.num_workers)

        image_names = []
        prediction_vals = []
        ground_truths = []
        test_corrects = 0.0
        test_data_size = len(test_data_to_dataset)

        # CHANGE FROM TRAINING MODE TO EVALUATION MODE
        self.trained_model.eval()

        with torch.no_grad():
            for images, labels in test_loader:
                image_names += images
                # TRANSFER NORMALIZED IMAGES TO GPU IF AVAILABLE FOR PREDICTIONS
                images = (images - 127.5) / 127.5
                images = images.to(self.device)

                # OUTPUT PROBABILITIES FOR ALL CLASSES
                outputs = self.trained_model(images)

                # CLASS WITH THE HIGHEST PROBABILITY WILL BE TAKEN AS FINAL PREDICTION BY MODEL
                _, predictions = torch.max(outputs, dim=1)

                # SUM OF ALL CORRECT OUTPUTS
                test_corrects += torch.sum(predictions == labels).item()

                # APPEND PREDICTIONS TO LIST
                prediction_vals += predictions.tolist()

                # APPEND GROUND TRUTHS TO LIST
                ground_truths += labels.tolist()

        # OVERALL TEST ACCURACY FOR THE TEST SET
        test_accuracy = (100 * (test_corrects / test_data_size))
        # F1 SCORE OF ENTIRE TEST SET
        test_f1_score = f1_score(ground_truths, prediction_vals, average="macro")
        if isinstance(test_f1_score, np.ndarray):
            test_f1_score = np.ndarray.tolist(test_f1_score)

        # APPEND IMAGES, GROUND TRUTHS AND PREDICTIONS FOR EACH IMAGE FOR INSPECTION
        images_and_predictions = list(zip(images, ground_truths, prediction_vals))

        torch.cuda.empty_cache()
        return test_accuracy, test_f1_score, images_and_predictions

    def save_model_features(self):
        """Saving all model related attributes for current training in case of reproduction"""
        model_accuracy, model_f1_score, images_and_predictions = self.model_test_set_accuracy()
        # print(model_f1_score)
        performance_attributes = self.model_attributes

        # IMPORTANT FEATURES OF MODEL TO BE SAVED
        # performance_attributes["predictions"] = images_and_predictions
        performance_attributes["test_batch_size"] = int(self.batch_size)
        performance_attributes["test_num_workers"] = int(self.num_workers)
        performance_attributes["model_accuracy"] = float(model_accuracy)
        performance_attributes["model_f1_score"] = model_f1_score

        with open(os.path.join(self.save_model_attributes_path, "performance.json"), "w") as saveFile:
            json.dump(performance_attributes, saveFile, indent=2)

    def test_data_prep(self):
        """Converting all test images to pandas dataframe for PyTorch Dataset class"""
        test_images = ImagesAndLabels(self.test_data_path).append_images_and_labels()
        test_data = pd.DataFrame(test_images, columns=["ImageName", "Label"])
        return test_data


def main():
    ModelTest(micron="20_micron", n_epochs=1).save_model_features()


if __name__ == "__main__":
    main()
