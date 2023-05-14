import os
import glob
import json
import copy
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# from training_model import ModelTrain
from torchvision.models import resnet50


class LoadImages:
    def __init__(self, prediction_images_path):
        self.prediction_images_path = prediction_images_path

    def load_images_as_dataframe(self):
        prediction_images = glob.glob(os.path.join(self.prediction_images_path, "*.jpg"))
        prediction_images_df = pd.DataFrame(prediction_images, columns=["ImageName"])
        return prediction_images_df


class PredictionDataset(Dataset):
    """
    Convert images and labels to pytorch dataset
    """
    def __init__(self, data_to_dataset):
        self.data = data_to_dataset
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        self.simple_transformation = transforms.Compose([transforms.Resize((1056, 1056),
                                                                           interpolation=transforms.InterpolationMode.BICUBIC),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(mean=self.mean, std=self.std)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data.loc[idx, "ImageName"])
        image = self.simple_transformation(img)
        return image


class ModelPredict:
    def __init__(self, prediction_images_path, model, best_weights_path, batch_size=64):
        super().__init__()
        self.prediction_images_path = prediction_images_path
        self.batch_size = batch_size
        self.current_model = model
        self.best_weights_path = best_weights_path

    def check_paths(self):
        if not os.path.exists(self.prediction_images_path):
            raise FileNotFoundError(f"{self.prediction_images_path} doesn't exist")

        if not os.path.exists(self.best_weights_path):
            raise FileNotFoundError(f"{self.best_weights_path} doesn't exist")

    def prepare_data(self):
        prediction_data_df = LoadImages(self.prediction_images_path).load_images_as_dataframe()
        prediction_dataset = PredictionDataset(prediction_data_df)
        prediction_loader = DataLoader(prediction_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        return prediction_data_df, prediction_loader

    def predict_images(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        prediction_data_df, prediction_loader = self.prepare_data()
        prediction_results = []
        current_model = self.current_model
        current_model.load_state_dict(torch.load(self.best_weights_path, device))
        current_model.eval()

        for images in prediction_loader:
            images = images.to(device)
            with torch.no_grad():
                outputs = current_model(images)
                _, predictions = torch.max(outputs, dim=1)
                predictions_output = predictions.cpu().detach().numpy()
            for prediction in predictions_output:
                prediction_results.append(prediction)

        prediction_data_df["Predictions"] = prediction_results
        predictions_dict = dict(zip(prediction_data_df['ImageName'], prediction_data_df["Predictions"]))

        return predictions_dict

    def classify_images_and_save_predictions(self):
        self.check_paths()
        predictions_dict = self.predict_images()
        with open(os.path.join(self.prediction_images_path, "classifications.json"), "w") as file:
            json.dump(predictions_dict, file, indent=1)


def main():
    prediction_images_path = os.path.join(os.getcwd(), "data", "dataset_preparation", "test_prediction_data",
                                          "20_micron")
    best_weights_path = os.path.join(os.getcwd(), "model_attributes", "20_micron", "current", "best_weights",
                                     "best_weights.pth")
    current_model = resnet50()
    # current_model = ModelTrain().output_model()

    ModelPredict(prediction_images_path=prediction_images_path,
                 model=current_model,
                 best_weights_path=best_weights_path).classify_images_and_save_predictions()


if __name__ == "__main__":
    main()
