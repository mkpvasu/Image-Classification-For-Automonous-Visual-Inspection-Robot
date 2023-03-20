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
    """
    Combining sub-images with its respective labels
    Categorical Encoding:
        Bad - 0.0
        Marginal - 1.0
        Good - 2.0
    """

    def __init__(self, dataPath):
        super(ImagesAndLabels, self).__init__()
        self.datasetPath = dataPath

    def append_images_and_labels(self):
        allImages = []
        subFolders = [subFolder.path for subFolder in os.scandir(self.datasetPath) if subFolder.is_dir()]

        labelCatEncoding = {"Bad": 0.0, "Marginal": 1.0, "Good": 2.0}
        for subFolder in subFolders:
            label = labelCatEncoding[os.path.basename(subFolder)]
            imagePaths = glob.glob(os.path.join(self.datasetPath, subFolder, "*.jpg"))
            for imagePath in imagePaths:
                if label is not None:
                    allImages.append([imagePath, label])
                else:
                    raise InvalidLabel()
        return allImages


class SandingCanopyDataset(Dataset):
    """ Convert images and labels to pytorch dataset """
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
    """ Visualize the images and labels loaded to dataset loader """
    def __init__(self, dataloader, classes):
        self.dataLoader = dataloader
        self.classes = classes

    # CONVERT IMAGE FROM FLAT IMAGE BACK TO ORIGINAL IMAGE
    def format_img(self, img):
        """ Converts to-be displayed image into RGB format """
        img = img
        npImg = img.numpy()
        npImg = np.transpose(npImg, (1, 2, 0))
        return npImg

    # DISPLAY IMAGES IN THE DATALOADER WITH LABELS
    def img_and_label_visualization(self):
        """
        To display images and its labels present in the dataloader
        """
        dataiter = iter(self.dataLoader)
        images, labels = next(dataiter)
        # Viewing data examples used for training
        fig, axis = plt.subplots(4, 5, figsize=(15, 10))
        for i, ax in enumerate(axis.flat):
            image, label = images[i], labels[i]
            ax.imshow(self.format_img(image))
            ax.set(title=f"{self.classes[label.item()]}")
        plt.show()


class ModelTrain:
    """ To train the ResNet50 model using the custom prepared sub-images dataset """
    def __init__(self, save_model_attributes_path, micron="30_micron", model=resnet50, weights=ResNet50_Weights.IMAGENET1K_V2,
                 loss=nn.CrossEntropyLoss(), optimizer=optim.Adam, model_lr_scheduler=lr_scheduler.ExponentialLR,
                 model_lr_scheduler_gamma=0.95, n_epochs=5, batch_size=64, train_val_split=0.2):

        # ASSIGN INPUTS TO INSTANCE VARIABLES
        # MICRON OF SANDING TO BE TRAINED
        self.micron = micron
        # DIRECTORY CONTAINING TRAIN DATA
        self.train_data_path = os.path.join(os.getcwd(), "data", "dataset_preparation",
                                            "dataset_for_model", "Train_Set", self.micron)
        # WEIGHTS INITIALIZATION FOR TRANSFER LEARNING
        self.weights = weights
        # MODEL
        self.model = model(weights=self.weights)
        # LOSS FUNCTION
        self.loss = loss
        # OPTIMIZER FUNCTION TO UPDATE WEIGHTS
        self.optimizer = optimizer(params=self.model.parameters(), lr=0.005)
        # EPOCHS TO BE TRAINED
        self.n_epochs = n_epochs
        # GAMMA FOR LEARNING RATE SCHEDULER
        self.model_lr_scheduler_gamma = model_lr_scheduler_gamma
        # UPDATE LEARNING RATE USING SCHEDULER
        self.model_lr_scheduler = model_lr_scheduler(optimizer=self.optimizer, gamma=self.model_lr_scheduler_gamma)
        # BATCH SIZE FOR TRAINING
        self.batchSize = batch_size
        # RATIO OF TRAINING AND VALIDATION SPLIT FROM THE TRAIN DATA
        self.trainValSplit = train_val_split

        # SELECT GPU IF AVAILABLE ELSE CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # PREPARATION OF TRAIN, VAL AND TEST DATA FOR DATASET CONVERSION
        self.trainDataToDataset, self.valDataToDataset = self.train_val_data_prep()

        # CONVERT PREPARED DATA INTO DATASET
        self.trainDataset = SandingCanopyDataset(self.trainDataToDataset)
        self.valDataset = SandingCanopyDataset(self.valDataToDataset)

        # LOAD DATA INTO TRAIN, VAL AND TEST DATALOADERS
        self.trainLoader = DataLoader(dataset=self.trainDataset, batch_size=self.batchSize, shuffle=True, num_workers=4)
        self.valLoader = DataLoader(dataset=self.valDataset, batch_size=self.batchSize, shuffle=True, num_workers=4)
        self.dataLoaders = {"train": self.trainLoader, "val": self.valLoader}

        # CATEGORICAL ENCODING OF CLASSES
        self.classes = {0.0: "Bad", 1.0: "Marginal", 2.0: "Good"}

        # TRACK LOSS AND ACCURACY FOR EACH EPOCH
        self.epochLoss = {"train": {}, "val": {}}
        self.epochAccuracy = {"train": {}, "val": {}}

        # VISUALIZATION OF IMAGES AND LABELS IN DATALOADER
        # LoaderVisualization(self.valLoader, self.classes).img_and_label_visualization()

        # SAVE WEIGHTS OF MODEL IN RESPECTIVE FOLDERS
        self.save_model_attributes_path = save_model_attributes_path

    def output_trained_model(self):
        """
        Outputs trained model

        Returns:
            self.trainModel: trained model
        """
        modelAttributes = dict()
        modelAttributes["dataset"] = "DatasetForModel_v" + time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(os.path.getmtime(os.path.join(os.getcwd(), "data", "dataset_preparation", "dataset_for_model"))))
        modelAttributes["model"] = self.model
        modelAttributes["weights"] = self.weights
        modelAttributes["loss"] = self.loss
        modelAttributes["optimizer"] = self.optimizer
        modelAttributes["model_lr_scheduler"] = self.model_lr_scheduler
        modelAttributes["model_lr_scheduler_gamma"] = self.model_lr_scheduler_gamma
        modelAttributes["classes"] = self.classes

        return modelAttributes, self.train_model()

    def train_model(self):
        """
        Train image classification model depending on input arguments specified by the user
        Returns:
            model: trained deep learning model
        """

        # TO ANALYZE THE DURATION OF TRAINING MODEL
        startTime = time.time()

        # TO KEEP TRACK OF BEST MODEL WEIGHT
        bestModelWeights = copy.deepcopy(self.model.state_dict())

        # BEST ACCURACY
        bestAcc = 0.0

        # TRAINING MODEL FOR SPECIFIED # OF EPOCHS
        for epoch in range(self.n_epochs):
            print(f"\nEpoch {epoch + 1}:")
            print("=" * 100, "\n")

            for mode in ["train", "val"]:
                if mode == "train":
                    self.model.train()
                else:
                    self.model.eval()

                runningLoss = 0.0
                runningCorrects = 0
                runningTotal = 0

                for batchIdx, (images, labels) in enumerate(self.dataLoaders[mode]):
                    # TRANSFER THE IMAGES AND CORRESPONDING LABELS TO GPU IF AVAILABLE
                    images = images.to(self.device)
                    labels = labels.type(torch.LongTensor).to(self.device)

                    # INITIALIZE GRADIENTS TO BE ZERO
                    self.optimizer.zero_grad()

                    # FORWARD PASSING IMAGES
                    with torch.set_grad_enabled(mode == "train"):
                        # PREDICT THE MODELS OUTPUTS FOR INPUT IMAGES
                        outputs = self.model(images)

                        # CALCULATE THE LOSS
                        loss = self.loss(outputs, labels)

                        # TAKE THE CLASS WITH MAXIMUM PROBABILITY AS THE PREDICTION
                        _, predictions = torch.max(outputs, dim=1)

                        # BACK-PROPAGATE AND UPDATE MODEL PARAMETERS IF TRAINING
                        if mode == "train":
                            loss.backward()

                            # UPDATE THE WEIGHTS WITH OPTIMIZER TAKING SMALL STEP AT A TIME
                            self.optimizer.step()

                    # PRINT BATCH LOSS FOR EVERY 5 BATCHES IN A EPOCH
                    if (mode == "train") and (batchIdx % 5 == 0):
                        print(f"\nBatch [{batchIdx}/{len(self.dataLoaders[mode])}]: Batch Loss: {loss.item():.4f}, "
                              f"Batch Accuracy: {100 * (torch.sum(predictions == labels.data)/len(labels)):.4f}")

                    # CALCULATE TOTAL LOSS AND CORRECT PREDICTIONS OF EACH EPOCH
                    runningLoss += loss.item() * images.size(0)
                    runningCorrects += torch.sum(predictions == labels.data)
                    runningTotal += len(labels.data)

                if self.model == "train":
                    self.model_lr_scheduler.step()

                # EPOCH LOSS IS AVG. LOSS OF ALL IMAGES
                epochLoss = runningLoss / runningTotal
                self.epochLoss[mode][epoch+1] = epochLoss

                # EPOCH ACCURACY IS TOTAL CORRECT PREDICTIONS BY TOTAL PREDICTIONS
                epochAccuracy = 100 * (runningCorrects / runningTotal)
                self.epochAccuracy[mode][epoch+1] = epochAccuracy

                if mode == "train":
                    phase = "Training"
                else:
                    phase = "Validation"

                print(f"\n--------------------- {phase} ---------------------\nEpoch Loss: {self.epochLoss[mode][epoch+1]:.4f}, "
                      f"Epoch Accuracy: {self.epochAccuracy[mode][epoch+1]:.4f}\n")

                # UPDATE BEST MODEL WEIGHTS FOR EPOCHS WITH IMPROVED VALIDATION ACCURACY AND SAVE ITS WEIGHTS
                if (mode == "val") and (self.epochAccuracy[mode][epoch+1] > bestAcc):
                    print("\n--- Model Performance Improved: Saving Weights ---\n")
                    torch.save(self.model.state_dict(), self.save_model_attributes_path + f"-epoch_{epoch}.pth")
                    bestAcc = epochAccuracy
                    bestModelWeights = copy.deepcopy(self.model.state_dict())

        torch.save(bestModelWeights, self.save_model_attributes_path + "-best_weights.pth")

        # EMPTY THE CUDA MEMORY AFTER TRAINING
        with torch.no_grad():
            torch.cuda.empty_cache()

        timeElapsed = time.time() - startTime
        print("\n", "*" * 100)
        print(f"\nTraining and Validation Completed in {timeElapsed // 60} minutes and {timeElapsed % 60} secs\n")
        print("*" * 100)

        # SAVE TRAINING AND VALIDATION LOSS CURVE AND ACCURACY WITH EPOCHS
        self.plot_loss_accuracy_curve()

        # LOAD BEST WEIGHTS
        self.model.load_state_dict(bestModelWeights)

        return self.model

    def plot_loss_accuracy_curve(self):
        for performanceMetrics in [self.epochLoss, self.epochAccuracy]:
            plt.figure()
            for mode in ["train", "val"]:
                plt.plot(performanceMetrics[mode].keys(), performanceMetrics[mode].values(), label=mode)
            plt.xlabel("# of Epochs")
            plt.ylabel("Epoch Loss")
            plt.title("Training and Validation Loss vs Epochs")
            if performanceMetrics == self.epochLoss:
                plt.savefig(os.path.join(self.save_model_attributes_path, "loss_curve"), format="png", bbox_inches="tight")
            else:
                plt.savefig(os.path.join(self.save_model_attributes_path, "accuracy_curve"), format="png", bbox_inches="tight")

    def train_val_data_prep(self):
        train_images = ImagesAndLabels(self.train_data_path).append_images_and_labels()
        train_df = pd.DataFrame(train_images, columns=['ImageName', 'Label'])
        train_data, val_data = train_test_split(train_df, test_size=self.trainValSplit)
        train_data, val_data = train_data.reset_index(), val_data.reset_index()
        return train_data, val_data