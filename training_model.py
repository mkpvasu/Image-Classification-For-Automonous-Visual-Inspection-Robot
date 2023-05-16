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
    """ Exception raised when label is None for assigning to an image """
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

    def __init__(self, data_path):
        super(ImagesAndLabels, self).__init__()
        self.datasetPath = data_path

    def append_images_and_labels(self):
        all_images = []
        sub_folders = [subFolder.path for subFolder in os.scandir(self.datasetPath) if subFolder.is_dir()]

        label_map = {"Bad": 0.0, "Marginal": 1.0, "Good": 2.0}
        for subFolder in sub_folders:
            label = label_map[os.path.basename(subFolder)]
            image_paths = glob.glob(os.path.join(self.datasetPath, subFolder, "*.jpg"))
            for imagePath in image_paths:
                if label is not None:
                    all_images.append([imagePath, label])
                else:
                    raise InvalidLabel()
        return all_images


class SandingCanopyDataset(Dataset):
    """ Convert images and labels to pytorch dataset """
    def __init__(self, data_to_dataset):
        self.data = data_to_dataset
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        self.transformation = transforms.Compose([transforms.Resize((1056, 1056),
                                                                    interpolation=transforms.InterpolationMode.BICUBIC),
                                                  transforms.RandomHorizontalFlip(p=0.5),
                                                  transforms.RandomVerticalFlip(p=0.5), transforms.ToTensor(),
                                                  transforms.Normalize(mean=self.mean, std=self.std)])
        self.simple_transformation = transforms.Compose([transforms.Resize((1056, 1056), interpolation=transforms.
                                                                           InterpolationMode.BICUBIC),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(mean=self.mean, std=self.std)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = torch.tensor(self.data.loc[idx, "Label"])
        img = Image.open(self.data.loc[idx, "ImageName"])
        if self.transformation is not None:
            image = self.transformation(img)
        else:
            image = self.simple_transformation(img)
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
        np_img = img.numpy()
        np_img = np.transpose(np_img, (1, 2, 0))
        return np_img

    # DISPLAY IMAGES IN THE DATALOADER WITH LABELS
    def img_and_label_visualization(self):
        """ To display images and its labels present in the dataloader """
        dataiter = iter(self.dataLoader)
        images, labels = next(dataiter)
        # Viewing data examples used for training
        fig, axis = plt.subplots(4, 5, figsize=(15, 10))
        for i, ax in enumerate(axis.flat):
            image, label = images[i], labels[i]
            ax.imshow(self.format_img(image))
            ax.set(title=f"{self.classes[label.item()]}")
        plt.show()


# UPDATE MODEL FOR CURRENT NECESSITIES IF NECESSARY
class Model:
    def __init__(self, model, classes, weights=None, freeze_weights=False):
        self.weights = weights
        self.model = model(weights=self.weights)
        self.freeze_weights = freeze_weights
        self.num_classes = len(classes)

    # UPDATE DEFAULT RESNET50 MODEL TO THE REQUIREMENTS AND OUTPUT THE CUSTOM MODEL
    def output_model(self):
        self.update_model()
        return self.model

    # FREEZE THE WEIGHTS OF ALL LAYERS EXCEPT FC AND UPDATE ONLY FC WEIGHTS
    def set_parameter_requires_grad(self):
        """ Sets all layer parameters has to be updated by optimizer if false else only fc parameters will be updated
        """
        if self.freeze_weights:
            for param in self.model.parameters():
                param.requires_grad = False

    # TO REPLACE FINAL FC LAYER FOR CURRENT REQUIREMENTS
    def update_model(self):
        """ Update final fc to the number of output classes and freeze pretrained weights if necessary """
        # IF FINE TUNING MODEL SET FREEZE WEIGHTS = FALSE
        # ELIF USING EXTRACTED FEATURES (FEATURE EXTRACTING) SET FREEZE WEIGHT = TRUE (ONLY LEARNS FC LAYER WEIGHTS)
        self.set_parameter_requires_grad()

        # REINITIALIZE FINAL LAYER TO HAVE NUMBER OF CURRENT CLASSES INSTEAD OF 1000 CLASSES IN DEFAULT RESNET50
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features=in_features, out_features=self.num_classes)


class ModelTrain:
    """ To train the ResNet50 model using the custom prepared sub-images dataset """
    def __init__(self, train_data_path, save_model_attributes_path, micron, model=resnet50,
                 weights=ResNet50_Weights.IMAGENET1K_V2, loss=nn.CrossEntropyLoss(), optimizer=optim.Adam,
                 model_lr_scheduler=lr_scheduler.ExponentialLR, optimizer_lr=0.005, model_lr_scheduler_gamma=0.95,
                 n_epochs=5, batch_size=64, num_workers=2, train_val_split=0.2, freeze_weights=False):

        # ASSIGN INPUTS TO INSTANCE VARIABLES
        # MICRON OF SANDING TO BE TRAINED
        self.micron = micron
        # DIRECTORY CONTAINING TRAIN DATA
        self.train_data_path = train_data_path
        # WEIGHTS INITIALIZATION FOR TRANSFER LEARNING
        self.weights = weights
        # FREEZE WEIGHTS
        self.freeze_weights = freeze_weights
        # CATEGORICAL ENCODING OF CLASSES
        self.classes = {0.0: "Bad", 1.0: "Marginal", 2.0: "Good"}
        # MODEL WITH TRANSFER LEARNING WEIGHTS
        self.model = Model(model=model, weights=self.weights, classes=self.classes,
                           freeze_weights=self.freeze_weights).output_model()
        # LOSS FUNCTION
        self.loss = loss
        # OPTIMIZER FUNCTION TO UPDATE WEIGHTS
        self.optimizer_lr = optimizer_lr
        self.optimizer = optimizer(params=self.model.parameters(), lr=self.optimizer_lr)
        # EPOCHS TO BE TRAINED
        self.n_epochs = n_epochs
        # GAMMA FOR LEARNING RATE SCHEDULER
        self.model_lr_scheduler_gamma = model_lr_scheduler_gamma
        # UPDATE LEARNING RATE USING SCHEDULER
        self.model_lr_scheduler = model_lr_scheduler(optimizer=self.optimizer, gamma=self.model_lr_scheduler_gamma)
        # BATCH SIZE FOR TRAINING
        self.batchSize = batch_size
        # RATIO OF TRAINING AND VALIDATION SPLIT FROM THE TRAIN DATA
        self.train_val_split = train_val_split
        # NUMBER OF WORKERS FOR DATALOADER
        self.num_workers = num_workers

        # SELECT GPU IF AVAILABLE ELSE CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # PREPARATION OF TRAIN, VAL AND TEST DATA FOR DATASET CONVERSION
        self.train_data_to_dataset, self.val_data_to_dataset = self.train_val_data_prep()

        # CONVERT PREPARED DATA INTO DATASET
        self.train_dataset = SandingCanopyDataset(self.train_data_to_dataset)
        self.val_dataset = SandingCanopyDataset(self.val_data_to_dataset)

        # LOAD DATA INTO TRAIN, VAL AND TEST DATALOADERS
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batchSize, shuffle=True,
                                       num_workers=self.num_workers)
        self.val_loader = DataLoader(dataset=self.val_dataset, batch_size=self.batchSize, shuffle=True,
                                     num_workers=self.num_workers)
        self.data_loaders = {"train": self.train_loader, "val": self.val_loader}

        # TRACK LOSS AND ACCURACY FOR EACH EPOCH
        self.epoch_loss = {"train": {}, "val": {}}
        self.epoch_accuracy = {"train": {}, "val": {}}

        # VISUALIZATION OF IMAGES AND LABELS IN DATALOADER
        # LoaderVisualization(self.val_loader, self.classes).img_and_label_visualization()

        # SAVE WEIGHTS OF MODEL IN RESPECTIVE FOLDERS
        self.save_model_attributes_path = save_model_attributes_path

    def output_trained_model(self):
        """
        Saves all model parameters in a dict to be copied to a JSON file for reproduction

        Returns:
            self.trainModel: trained model
        """
        model_attributes = dict()
        model_attributes["dataset"] = "DatasetForModel_v" + \
                                      time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(
                                          os.path.getmtime(os.path.join(os.getcwd(), "data", "dataset_preparation",
                                                                        "dataset_for_model"))))
        trained_model = self.train_model()
        model_attributes["train_data_path"] = str(self.train_data_path)
        model_attributes["save_model_attributes_path"] = str(self.save_model_attributes_path)
        model_attributes["micron"] = str(self.micron)
        model_attributes["model"] = str(trained_model)
        model_attributes["weights"] = str(self.weights)
        model_attributes["loss"] = str(self.loss)
        model_attributes["optimizer"] = str(self.optimizer)
        model_attributes["optimizer_lr"] = float(self.optimizer_lr)
        model_attributes["model_lr_scheduler"] = str(self.model_lr_scheduler)
        model_attributes["model_lr_scheduler_gamma"] = float(self.model_lr_scheduler_gamma)
        model_attributes["num_epochs"] = int(self.n_epochs)
        model_attributes["batch_size"] = int(self.batchSize)
        model_attributes["num_workers"] = int(self.num_workers)
        model_attributes["train_val_split"] = float(self.train_val_split)
        model_attributes["freeze_weights"] = str(self.freeze_weights)
        model_attributes["classes"] = str(self.classes)

        return model_attributes, trained_model

    def train_model(self):
        """
        Train image classification model depending on input arguments specified by the user
        Returns:
            model: trained deep learning model
        """

        # TO ANALYZE THE DURATION OF TRAINING MODEL
        start_time = time.time()

        # TO KEEP TRACK OF BEST MODEL WEIGHT
        best_model_weights = copy.deepcopy(self.model.state_dict())

        # BEST ACCURACY
        best_accuracy = 0.0

        # TRAINING MODEL FOR SPECIFIED # OF EPOCHS
        for epoch in range(self.n_epochs):
            print(f"\nEpoch {epoch + 1}:")
            print("=" * 100, "\n")

            for mode in ["train", "val"]:
                if mode == "train":
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0
                running_total = 0

                for batchIdx, (images, labels) in enumerate(self.data_loaders[mode]):
                    # TRANSFER THE IMAGES AND CORRESPONDING LABELS TO GPU IF AVAILABLE
                    images = (images - 127.5) / 127.5
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
                        print(f"\nBatch [{batchIdx}/{len(self.data_loaders[mode])}]: Batch Loss: {loss.item():.4f}, "
                              f"Batch Accuracy: {100 * (torch.sum(predictions == labels.data)/len(labels)):.4f}")

                    # CALCULATE TOTAL LOSS AND CORRECT PREDICTIONS OF EACH EPOCH
                    running_loss += loss.item() * images.size(0)
                    running_corrects += torch.sum(predictions == labels.data)
                    running_total += len(labels.data)

                if self.model == "train":
                    self.model_lr_scheduler.step()

                # EPOCH LOSS IS AVG. LOSS OF ALL IMAGES
                epoch_loss = running_loss / running_total
                self.epoch_loss[mode][epoch+1] = epoch_loss

                # EPOCH ACCURACY IS TOTAL CORRECT PREDICTIONS BY TOTAL PREDICTIONS
                epoch_accuracy = 100 * (running_corrects / running_total)
                self.epoch_accuracy[mode][epoch+1] = epoch_accuracy

                if mode == "train":
                    phase = "Training"
                else:
                    phase = "Validation"

                print(f"\n--------------------- {phase} ---------------------\nEpoch Loss: "
                      f"{self.epoch_loss[mode][epoch+1]:.4f}, "
                      f"Epoch Accuracy: {self.epoch_accuracy[mode][epoch+1]:.4f}\n")

                # UPDATE BEST MODEL WEIGHTS FOR EPOCHS WITH IMPROVED VALIDATION ACCURACY AND SAVE ITS WEIGHTS
                if (mode == "val") and (self.epoch_accuracy[mode][epoch+1] > best_accuracy):
                    print("\n--- Model Performance Improved: Saving Weights ---\n")
                    torch.save(self.model.state_dict(), os.path.join(self.save_model_attributes_path,
                                                                     f"weights_epoch_{epoch}.pth"))
                    best_accuracy = epoch_accuracy
                    best_model_weights = copy.deepcopy(self.model.state_dict())

        torch.save(best_model_weights, os.path.join(self.save_model_attributes_path, f"best_weights.pth"))

        # EMPTY THE CUDA MEMORY AFTER TRAINING
        with torch.no_grad():
            torch.cuda.empty_cache()

        time_elapsed = time.time() - start_time
        print("\n", "*" * 100)
        print(f"\nTraining and Validation Completed in {time_elapsed // 60} minutes and {time_elapsed % 60} secs\n")
        print("*" * 100)

        # SAVE TRAINING AND VALIDATION LOSS CURVE AND ACCURACY WITH EPOCHS
        self.plot_loss_accuracy_curve()

        # LOAD BEST WEIGHTS
        self.model.load_state_dict(best_model_weights)

        torch.cuda.empty_cache()
        return self.model

    def plot_loss_accuracy_curve(self):
        for performance_metrics in [self.epoch_loss, self.epoch_accuracy]:
            plt.figure()
            for mode in ["train", "val"]:
                plt.plot(performance_metrics[mode].keys(), performance_metrics[mode].values(), label=mode)

            if performance_metrics == self.epoch_loss:
                plt.xlabel("# of Epochs")
                plt.ylabel("Epoch Loss")
                plt.title("Training and Validation Loss vs Epochs")
                plt.legend()
                plt.savefig(os.path.join(self.save_model_attributes_path, "loss_curve.png"), bbox_inches="tight")
            else:
                plt.xlabel("# of Epochs")
                plt.ylabel("Epoch Accuracy")
                plt.title("Training and Validation Accuracy vs Epochs")
                plt.legend()
                plt.savefig(os.path.join(self.save_model_attributes_path, "accuracy_curve.png"), bbox_inches="tight")
            plt.close()

    def train_val_data_prep(self):
        train_images = ImagesAndLabels(self.train_data_path).append_images_and_labels()
        train_df = pd.DataFrame(train_images, columns=['ImageName', 'Label'])
        train_data, val_data = train_test_split(train_df, test_size=self.train_val_split)
        train_data, val_data = train_data.reset_index(), val_data.reset_index()
        return train_data, val_data
