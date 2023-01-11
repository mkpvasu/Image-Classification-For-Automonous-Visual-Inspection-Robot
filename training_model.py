from __future__ import print_function, division
import os
import glob
import time
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
from PIL import Image
import torch.nn as nn
from torch import optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from torch.optim import lr_scheduler


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


class LoaderVisualization:
    def __init__(self, dataloader, classes):
        self.dataLoader = dataloader
        self.classes = classes

    # CONVERT IMAGE FROM FLAT IMAGE BACK TO ORIGINAL IMAGE
    def showImg(self, img):
        img = img
        npimg = img.numpy()
        npimg = np.transpose(npimg, (1, 2, 0))
        return npimg

    # DISPLAY IMAGES IN THE DATALOADER WITH LABELS
    def imgAndLabelVisualization(self):
        dataiter = iter(self.dataLoader)
        images, labels = next(dataiter)
        # Viewing data examples used for training
        fig, axis = plt.subplots(4, 5, figsize=(15, 10))
        for i, ax in enumerate(axis.flat):
            image, label = images[i], labels[i]
            ax.imshow(self.showImg(image))
            ax.set(title=f"{self.classes[label.item()]}")
        plt.show()


# CLASS TO TRAIN RESNET MODEL ON INPUT DATASET
class TrainingModel:
    def __init__(self, batch_size):

        # SELECT GPU IF AVAILABLE ELSE CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # SIZE OF THE BATCH
        self.batchSize = batch_size

        # PREPARATION OF TRAIN, VAL AND TEST DATA FOR DATASET CONVERSION
        self.trainDataToDataset, self.valDataToDataset = TrainData().trainValDataPrep()
        self.testDataToDataset = TestData().testDataPrep()
        self.datasetSize = {"train": len(self.trainDataToDataset), "val": len(self.valDataToDataset)}

        # CONVERT PREPARED DATA INTO DATASET
        self.trainData = SandingCanopyDataset(self.trainDataToDataset)
        self.valData = SandingCanopyDataset(self.valDataToDataset)
        self.testData = SandingCanopyDataset(self.testDataToDataset)

        # LOAD DATA INTO TRAIN, VAL AND TEST DATALOADERS
        self.trainLoader = DataLoader(dataset=self.trainData, batch_size=self.batchSize, shuffle=True, num_workers=4)
        self.valLoader = DataLoader(dataset=self.valData, batch_size=self.batchSize, shuffle=True, num_workers=4)
        self.testLoader = DataLoader(dataset=self.testData, batch_size=self.batchSize, shuffle=True, num_workers=4)
        self.dataLoaders = {"train": len(self.trainDataToDataset), "val": len(self.valDataToDataset)}

        # CATEGORICAL ENCODING OF CLASSES
        self.classes = {0.0: "unsatisfactory", 1.0: "moderatelysatisfactory", 2.0: "satisfactory"}

        # TRACK LOSS AND ACCURACY FOR EACH EPOCH
        self.Loss = {"train": [], "val": []}
        self.Accuracy = {"train": [], "val": []}

        # VISUALIZATION OF IMAGES AND LABELS IN DATALOADER
        LoaderVisualization(self.valLoader, self.classes).imgAndLabelVisualization()

    def trainModel(self, model=resnet50(), loss=nn.CrossEntropyLoss(), optimizer=optim.Adam, lr_scheduler, n_epochs=10):
        # LOAD MODEL
        model = model(weights=None)

        # EPOCHS TO BE TRAINED
        epochs = n_epochs

        # LOSS FUNCTION
        criterion = loss

        # OPTIMIZER FUNCTION TO UPDATE WEIGHTS
        optimizer = optimizer(model.parameters(), lr=0.001)

        # UPDATE LEARNING RATE USING SCHEDULER
        scheduler = lr_scheduler

        # TO ANALYZE THE DURATION OF TRAINING MODEL
        startTime = time.time()

        # TO KEEP TRACK OF BEST MODEL WEIGHT
        bestModelWeights = copy.deepcopy(model.state_dict())

        # BEST ACCURACY
        bestAcc = 0.0

        # TRAINING MODEL FOR MENTIONED EPOCHS
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}")
            print("=" * 15)

            for mode in ["train", "val"]:
                if mode == "train":
                    model.train()
                else:
                    model.eval()

                runningLoss = 0.0
                runningCorrects = 0

                for batchIdx, (images, labels) in enumerate(self.dataLoaders[mode]):
                    images = images.to_device(self.device)
                    labels = labels.to_device(self.device)

                    # INITIALIZE GRADIENTS TO BE ZERO
                    optimizer.zero_grad()

                    # FORWARD PASSING IMAGES TO TRAIN
                    with torch.set_grad_enabled(mode == "train"):
                        output = model(images)
                        _, predictions = torch.max(output, dim=1)
                        loss = criterion(predictions, labels)

                        # BACKPROPAGATION AND UPDATE PARAMETERS DURING TRAIN
                        if mode == "train":
                            loss.backward()
                            optimizer.step()

                    # PRINT BATCH LOSS


                    # CALCULATE OVERALL LOSS
                    runningLoss += loss.item() * images.size(0)
                    runningCorrects += torch.sum(predictions == labels.data)

                    if batchIdx % 5 == 0:
                        print(f"    Batch [{batchIdx}/{len(self.dataLoaders[mode])}], Batch Loss: {loss.item():.4f}, "
                              f"Batch Accuracy: {}")

                if model == "train":
                    scheduler.step()

                epochLoss = runningLoss / self.datasetSize[mode]
                self.Loss[mode].append(epochLoss)
                epochAccuracy = runningCorrects / self.datasetSize[mode]
                self.Accuracy[mode].append(epochAccuracy)

                if mode == "train":
                    phase = "Train"
                else:
                    phase = "Validation"

                print(f"{phase}, Epoch Loss: {epochLoss}, Epoch Accuracy: {epochAccuracy}\n\n")

                # COPY WEIGHTS TO BEST MODEL WEIGHTS FOR HIGHEST ACCURACY EPOCHS
                if (mode == "val") and (epochAccuracy > bestAcc):
                    bestAcc = epochAccuracy
                    bestModelWeights = copy.deepcopy(model.state_dict())

        timeElapsed = time.time() - startTime
        print(f"Training Completed in {timeElapsed // 60}mins {timeElapsed % 60}secs")

        # LOAD BEST WEIGHTS
        model.load_state_dict(bestModelWeights)
        return model


def main():
    TrainingModel()


if __name__ == "__main__":
    main()







