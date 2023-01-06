from __future__ import print_function, division
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class InvalidLabel(Exception):
    """
    Exception raised when label is None for assigning to an image

    Arguments:
        label: label to be assigned to the image
    """
    def __init__(self):
        self.message = "Label cannot be None"
        super().__init__(self.message)


class ImagesAndLabels:
    def __init__(self, dataPath):
        super(ImagesAndLabels, self).__init__()
        self.datasetPath = dataPath

    def appendImagesAndLabels(self):
        allImages = []
        subFolders = [subFolder.path for subFolder in os.scandir(self.datasetPath) if subFolder.is_dir()]

        labelCatEncoding = {"unsatisfactory": 0.0, "moderatelysatisfactory": 1.0, "satisfactory": 2.0}
        label = None
        for subFolder in subFolders:
            label = labelCatEncoding[os.path.basename(subFolder)]
            imagePaths = glob.glob(os.path.join(self.datasetPath, subFolder, "*.jpg"))
            for imagePath in imagePaths:
                if label is not None:
                    allImages.append([imagePath, label])
                else:
                    raise InvalidLabel()
        return allImages


class TrainData:
    def __init__(self):
        super().__init__()
        self.trainRatio = 0.8
        self.valRatio = 0.2
        self.trainDataPath = os.path.join(os.getcwd(), "data", "dataset_for_model", "train_data")

    def trainValDataPrep(self):
        trainImages = ImagesAndLabels(self.trainDataPath).appendImagesAndLabels()
        trainDf = pd.DataFrame(trainImages, columns=['ImageName', 'Label'])
        trainData, valData = train_test_split(trainDf, test_size=0.2)
        trainData, valData = trainData.reset_index(), valData.reset_index()
        return trainData, valData


class TestData:
    def __init__(self):
        super().__init__()
        self.testDataPath = os.path.join(os.getcwd(), "data", "dataset_for_model", "test_data")

    def testDataPrep(self):
        testImages = ImagesAndLabels(self.testDataPath).appendImagesAndLabels()
        testData = pd.DataFrame(testImages, columns=["ImageName", "Label"])
        return testData


class SandingCanopyDataset(Dataset):
    def __init__(self, dataToDataset):
        self.data = dataToDataset
        self.transformation = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                                  transforms.RandomVerticalFlip(p=0.5),
                                                  transforms.ToTensor()])
        self.convertToTensor = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = torch.tensor(self.data.loc[idx, "Label"])
        img = Image.open(self.data.loc[idx, "ImageName"])
        if self.transformation is not None:
            image = self.transformation(img)
        else:
            image = self.convertToTensor(img)
        return image, label


class TrainingModel:
    def __init__(self):
        self.batchSize = 16

        # PREPARATION OF TRAIN, VAL AND TEST DATA FOR DATASET CONVERSION
        self.trainDataToDataset, self.valDataToDataset = TrainData().trainValDataPrep()
        self.testDataToDataset = TestData().testDataPrep()

        # CONVERT PREPARED DATA INTO DATASET
        self.trainData = SandingCanopyDataset(self.trainDataToDataset)
        self.valData = SandingCanopyDataset(self.valDataToDataset)
        self.testData = SandingCanopyDataset(self.testDataToDataset)

        # LOAD DATA INTO TRAIN, VAL AND TEST DATALOADERS
        self.trainLoader = DataLoader(dataset=self.trainData, batch_size=self.batchSize, shuffle=True, num_workers=4)
        self.valLoader = DataLoader(dataset=self.valData, batch_size=self.batchSize, shuffle=True, num_workers=4)
        self.testLoader = DataLoader(dataset=self.testData, batch_size=self.batchSize, shuffle=True, num_workers=4)

        # CATEGORICAL ENCODING OF CLASSES
        self.classes = {0.0: "unsatisfactory", 1.0: "moderatelysatisfactory", 2.0: "satisfactory"}

    def showImg(self, img):
        img = img
        npimg = img.numpy()
        npimg = np.transpose(npimg, (1, 2, 0))
        return npimg

    def imgAndLabelVisualization(self):
        dataiter = iter(self.valLoader)
        images, labels = next(dataiter)
        # Viewing data examples used for training
        fig, axis = plt.subplots(3, 5, figsize=(15, 10))
        for i, ax in enumerate(axis.flat):
            image, label = images[i], labels[i]
            ax.imshow(self.showImg(image))
            ax.set(title=f"{self.classes[label.item()]}")
        plt.show()


def main():
    TrainingModel().imgAndLabelVisualization()


if __name__ == "__main__":
    main()







