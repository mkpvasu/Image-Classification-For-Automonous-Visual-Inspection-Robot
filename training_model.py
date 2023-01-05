from __future__ import print_function, division
import os
import glob
from torch.utils import data
from torchvision.models import resnet50
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import SGD


class InvalidLabel(Exception):
    """
    Exception raised when label is None for assigning to an image

    Arguments:
        label: label to be assigned to the image
    """
    def __init__(self):
        self.message = "Label cannot be None"
        super().__init__(self.message)


class SandingCanopyDataset(Dataset):
    def __init__(self):
        self.Images = ImagesAndLabels.combineImagesAndLabels()
        self.transformation = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                  transforms.RandomVerticalFlip(),
                                                  transforms.ToTensor()])
        self.convertToTensor = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.Images)

    def __getitem__(self, idx):
        label = self.Images[idx][1]
        npImg = io.imread(self.Images[idx][0]).astype(float)
        npImg = npImg / 255
        if self.transformation is not None:
            image = self.transformation(npImg)
        else:
            image = self.convertToTensor(npImg)
        return (image, label)


class ImagesAndLabels:
    def __int__(self):
        super(ImagesAndLabels, self).__init__()
        self.datasetPath = os.path.join(os.getcwd(), "data", "dataset")
        self.trainRatio = 0.75
        self.valRatio = 0.15
        self.testRatio = 0.1

    def combineImagesAndLabels(self):
        allImages = []
        subFolders = [subFolder.path for subFolder in os.scandir(self.datasetPath) if subFolder.isdir()]

        label = None
        for subFolder in subFolders:
            if subFolder == "unsatisfactory":
                label = 0.0
            elif subFolder == "moderatelysatisfactory":
                label = 1.0
            elif subFolder == "satisfactory":
                label = 2.0

            imagePaths = glob.glob(os.path.join(self.datasetPath, subFolder, "*.jpg"))
            for imagePath in imagePaths:
                if label is not None:
                    allImages.append([imagePath, label])
                else:
                    raise InvalidLabel().__init__()

        return allImages

    def trainValTestSplit(self):
        totalImages = self.combineImagesAndLabels()
        trainData, valData, testData = data.random_split(totalImages, [self.trainRatio, self.valRatio, self.testRatio])
        return trainData, valData, testData


class TrainingModel:
    def __init__(self):
        trainData, valData, testData = ImagesAndLabels.trainValTestSplit()
        trainLoader = DataLoader(dataset=trainData, batch_size=64, shuffle=True, num_workers=4)
        valLoader = DataLoader(dataset=valData, batch_size=64, shuffle=True, num_workers=4)
        testLoader = DataLoader(dataset=testData, batch_size=64, shuffle=True, num_workers=4)
        classes = ("unsatisfactory", "moderatelysatisfactory", "satisfactory")

    for images, labels in datasetLoader:
        Resnet50 = resnet50()







