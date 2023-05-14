import os
import glob
import json
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class LoadImages:
    def __init__(self, partition_images_path):
        self.partitionImagesPath = partition_images_path

    def loadImagesAsDF(self):
        predictionImages = glob.glob(os.path.join(self.partitionImagesPath, "*.jpg"))
        predictionImagesDf = pd.DataFrame(predictionImages, columns=["ImageName"])
        return predictionImagesDf


class PredictionDataset(Dataset):
    """
    Convert images and labels to pytorch dataset
    """

    def __init__(self, dataToDataset):
        self.data = dataToDataset
        self.convertToTensor = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data.loc[idx, "ImageName"])
        image = self.convertToTensor(img)
        return image


class ModelPredict:
    def __init__(self, partition_images_path, current_model, batch_size=64):
        super().__init__()
        self.partitionImagesPath = partition_images_path
        self.batchSize = batch_size
        self.currentModel = current_model

    def prepareData(self):
        predictionDataDf = LoadImages(self.partitionImagesPath).loadImagesAsDF()
        predictionDataset = PredictionDataset(predictionDataDf)
        predictionLoader = DataLoader(predictionDataset, batch_size=self.batchSize, shuffle=False, num_workers=4)
        return predictionDataDf, predictionLoader

    def predictImages(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        predictionDataDf, predictionLoader = self.prepareData()
        predictionResults = []
        currentModel = self.currentModel.load_state_dict(torch.load("*_best-weights.pth"))
        currentModel.to(device)
        currentModel.eval()

        for images in predictionLoader:
            images = images.to(device)
            with torch.no_grad():
                outputs = currentModel(images)
                _, predictions = torch.max(outputs, dim=1)
                predictionsOutput = predictions.cpu().detach().numpy()
            for prediction in predictionsOutput:
                predictionResults.append(prediction)

        predictionDataDf["Predictions"] = predictionResults
        predictionsDict = dict(zip(predictionDataDf['ImageName'], predictionDataDf["Predictions"]))

        return predictionsDict

    def classifyImagesAndSavePredictions(self, classificationFilePath):
        predictionsDict = self.predictImages()
        with open(os.path.join(classificationFilePath, "classifications.json"), "w") as file:
            json.dump(predictionsDict, file, indent=1)


def main():
    ModelPredict().classifyImagesAndSavePredictions()


if __name__ == "__main__":
    main()
