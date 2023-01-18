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
from torch import nn
from torch import optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim import lr_scheduler


class InvalidLabel(Exception):
    """
    Exception raised when label is None for assigning to an image
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

        # CHANGE TRAIN DATA PATH ACCORDING TO SET UP
        self.trainDataPath = os.path.join(os.getcwd(), "data", "dataset_for_model", "train_data")

    def trainValDataPrep(self):
        trainImages = ImagesAndLabels(self.trainDataPath).appendImagesAndLabels()
        trainDf = pd.DataFrame(trainImages, columns=['ImageName', 'Label'])
        trainData, valData = train_test_split(trainDf, test_size=0.2)
        trainData, valData = trainData.reset_index(), valData.reset_index()
        return trainData, valData


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
        npImg = img.numpy()
        npImg = np.transpose(npImg, (1, 2, 0))
        return npImg

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
    def __init__(self, batch_size=8):

        # SELECT GPU IF AVAILABLE ELSE CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # SIZE OF THE BATCH
        self.batchSize = batch_size

        # PREPARATION OF TRAIN, VAL AND TEST DATA FOR DATASET CONVERSION
        self.trainDataToDataset, self.valDataToDataset = TrainData().trainValDataPrep()

        # CONVERT PREPARED DATA INTO DATASET
        self.trainData = SandingCanopyDataset(self.trainDataToDataset)
        self.valData = SandingCanopyDataset(self.valDataToDataset)

        # LOAD DATA INTO TRAIN, VAL AND TEST DATALOADERS
        self.trainLoader = DataLoader(dataset=self.trainData, batch_size=self.batchSize, shuffle=True, num_workers=4)
        self.valLoader = DataLoader(dataset=self.valData, batch_size=self.batchSize, shuffle=True, num_workers=4)
        self.dataLoaders = {"train": self.trainLoader, "val": self.valLoader}

        # CATEGORICAL ENCODING OF CLASSES
        self.classes = {0.0: "unsatisfactory", 1.0: "moderatelysatisfactory", 2.0: "satisfactory"}

        # TRACK LOSS AND ACCURACY FOR EACH EPOCH
        self.EpochLoss = {"train": [], "val": []}
        self.EpochAccuracy = {"train": [], "val": []}

        # VISUALIZATION OF IMAGES AND LABELS IN DATALOADER
        # LoaderVisualization(self.valLoader, self.classes).imgAndLabelVisualization()

    def outputTrainedModel(self, model=resnet50, loss=nn.CrossEntropyLoss(), optimizer=optim.Adam, n_epochs=10):
        return self.trainModel(model, loss, optimizer, n_epochs)

    def trainModel(self, model, loss, optimizer, n_epochs):
        # LOAD MODEL
        model = model(weights=ResNet50_Weights.IMAGENET1K_V2)

        # EPOCHS TO BE TRAINED
        epochs = n_epochs

        # LOSS FUNCTION
        criterion = loss

        # OPTIMIZER FUNCTION TO UPDATE WEIGHTS
        optimizer = optimizer(model.parameters(), lr=0.005)

        # UPDATE LEARNING RATE USING SCHEDULER
        scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

        # TO ANALYZE THE DURATION OF TRAINING MODEL
        startTime = time.time()

        # TO KEEP TRACK OF BEST MODEL WEIGHT
        bestModelWeights = copy.deepcopy(model.state_dict())

        # BEST ACCURACY
        bestAcc = 0.0

        # TRAINING MODEL FOR MENTIONED EPOCHS
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1} / {epochs}")
            print("=" * 100, "\n")

            for mode in ["train", "val"]:
                if mode == "train":
                    model.train()
                else:
                    model.eval()

                runningLoss = 0.0
                runningCorrects = 0
                runningTotal = 0

                for batchIdx, (images, labels) in enumerate(self.dataLoaders[mode]):
                    images = images.to(self.device)
                    labels = labels.type(torch.LongTensor).to(self.device)

                    # INITIALIZE GRADIENTS TO BE ZERO
                    optimizer.zero_grad()

                    # FORWARD PASSING IMAGES
                    with torch.set_grad_enabled(mode == "train"):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        _, predictions = torch.max(outputs, dim=1)

                        # BACKPROPAGATION AND UPDATE PARAMETERS DURING TRAIN
                        if mode == "train":
                            loss.backward()
                            optimizer.step()

                    # PRINT BATCH LOSS
                    if (mode == "train") and (batchIdx % 5 == 0):
                        print(f"Batch [{batchIdx}/{len(self.dataLoaders[mode])}]: Batch Loss: {loss.item():.4f}, "
                              f"Batch Accuracy: {100 * (torch.sum(predictions == labels.data)/len(labels)):.4f}")

                    # CALCULATE TOTAL LOSS AND CORRECTS OF EPOCH
                    runningLoss += loss.item() * images.size(0)
                    runningCorrects += torch.sum(predictions == labels.data)
                    runningTotal += len(labels.data)

                if model == "train":
                    scheduler.step()

                epochLoss = runningLoss / runningTotal
                self.EpochLoss[mode].append(epochLoss)
                epochAccuracy = 100 * (runningCorrects / runningTotal)
                self.EpochAccuracy[mode].append(epochAccuracy)

                if mode == "train":
                    phase = "Training"
                else:
                    phase = "Validation"

                print(f"\n--- {phase} ----\nEpoch Loss: {self.EpochLoss[mode][epoch]:.4f}, "
                      f"Epoch Accuracy: {self.EpochAccuracy[mode][epoch]:.4f}\n")

                # COPY WEIGHTS TO BEST MODEL WEIGHTS FOR HIGHEST ACCURACY EPOCHS
                if (mode == "val") and (self.EpochAccuracy[mode][epoch] > bestAcc):
                    print("\n--- Model Performance Improved: Saving Weights ---\n\n")
                    bestAcc = epochAccuracy
                    bestModelWeights = copy.deepcopy(model.state_dict())

        timeElapsed = time.time() - startTime
        print("\n", "*" * 100)
        print(f"\nTraining and Validation Completed in {timeElapsed // 60} minutes and {timeElapsed % 60} secs\n")
        print("*" * 100, "\n")

        # LOAD BEST WEIGHTS
        model.load_state_dict(bestModelWeights)

        return model


# def main():
#     resnet50Model = TrainingModel().trainModel()
#
#
# if __name__ == "__main__":
#     main()







