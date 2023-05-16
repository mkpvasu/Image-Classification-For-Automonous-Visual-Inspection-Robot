import os
import glob
import json
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from training_model import Model
from torchvision.models import resnet50


class LoadImages:
    """Class to load all the images for prediction into dataframe"""
    def __init__(self, prediction_images_path):
        self.prediction_images_path = prediction_images_path

    def load_images_as_dataframe(self):
        """Load all the images for prediction into dataframe for PyTorch Dataset class"""
        prediction_images = glob.glob(os.path.join(self.prediction_images_path, "*.jpg"))
        prediction_images_df = pd.DataFrame(prediction_images, columns=["ImageName"])
        return prediction_images_df


class PredictionDataset(Dataset):
    """ Convert images and labels to pytorch dataset """
    def __init__(self, data_to_dataset):
        self.data = data_to_dataset
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        self.simple_transformation = transforms.Compose([transforms.Resize((1056, 1056), interpolation=transforms.
                                                                           InterpolationMode.BICUBIC),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(mean=self.mean, std=self.std)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data.loc[idx, "ImageName"])
        image = self.simple_transformation(img)
        return image


class ModelPredict:
    """Class to perform predictions on new data"""
    def __init__(self, prediction_images_path, model, best_weights_path, batch_size=8):
        super().__init__()
        self.prediction_images_path = prediction_images_path
        self.batch_size = batch_size
        self.model = model
        self.best_weights_path = best_weights_path
        self.num_classes = 3

    def check_paths(self):
        """Check if prediction images and best weights path is valid"""
        if not os.path.exists(self.prediction_images_path):
            raise FileNotFoundError(f"{self.prediction_images_path} doesn't exist")

        if not os.path.exists(self.best_weights_path):
            raise FileNotFoundError(f"{self.best_weights_path} doesn't exist")

    def update_model_fc(self):
        """ Update final fc to the number of output classes and freeze pretrained weights if necessary """
        # REINITIALIZE FINAL LAYER TO HAVE NUMBER OF CURRENT CLASSES INSTEAD OF 1000 CLASSES IN DEFAULT RESNET50
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features=in_features, out_features=self.num_classes)

    def prepare_data(self):
        """ Convert dataframe to PyTorch Dataset and DataLoader class """
        prediction_data_df = LoadImages(self.prediction_images_path).load_images_as_dataframe()
        prediction_dataset = PredictionDataset(prediction_data_df)
        prediction_loader = DataLoader(prediction_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        return prediction_data_df, prediction_loader

    def predict_images(self):
        """ Predict classes for prediction images using trained model """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        prediction_data_df, prediction_loader = self.prepare_data()
        prediction_results = []
        self.model.load_state_dict(torch.load(self.best_weights_path, device))
        self.model.eval()

        for images in prediction_loader:
            images = images.to(device)
            with torch.no_grad():
                outputs = self.model(images)
                print(outputs)
                _, predictions = torch.max(outputs, dim=1)
                predictions_output = predictions.cpu().detach().numpy()
            for prediction in predictions_output:
                prediction_results.append(prediction)

        prediction_data_df["Predictions"] = prediction_results
        predictions_dict = dict(zip(prediction_data_df['ImageName'], prediction_data_df["Predictions"]))

        return predictions_dict

    def classify_images_and_save_predictions(self):
        """ Save images and corresponding predictions in a JSON file in the same folder"""
        self.check_paths()
        predictions_dict = self.predict_images()
        with open(os.path.join(self.prediction_images_path, "classifications.json"), "w") as file:
            json.dump(predictions_dict, file, indent=1)


def main():
    prediction_images_path = os.path.join(os.getcwd(), "data", "dataset_preparation", "test_prediction_data",
                                          "20_micron")
    best_weights_path = os.path.join(os.getcwd(), "model_attributes", "20_micron", "current", "best_weights",
                                     "best_weights.pth")
    classes = {0.0: "Bad", 1.0: "Marginal", 2.0: "Good"}
    current_model = Model(model=resnet50, classes=classes, weights=None, freeze_weights=False).output_model()

    ModelPredict(prediction_images_path=prediction_images_path, model=current_model,
                 best_weights_path=best_weights_path).classify_images_and_save_predictions()


if __name__ == "__main__":
    main()
