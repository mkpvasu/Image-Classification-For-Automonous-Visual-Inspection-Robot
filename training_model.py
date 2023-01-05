from __future__ import print_function, division
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
from torchvision.models import resnet50
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
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
        return image, label


class ImagesAndLabels:
    def __init__(self):
        super(ImagesAndLabels, self).__init__()
        self.datasetPath = os.path.join(os.getcwd(), "data", "dataset")
        self.trainRatio = 0.5
        self.valRatio = 0.25
        self.testRatio = 0.25

    def combineImagesAndLabels(self):
        allImages = []
        subFolders = [subFolder.path for subFolder in os.scandir(self.datasetPath) if subFolder.is_dir()]

        label = None
        for subFolder in subFolders:
            if os.path.basename(subFolder) == "unsatisfactory":
                label = 0.0
            elif os.path.basename(subFolder) == "moderatelysatisfactory":
                label = 1.0
            elif os.path.basename(subFolder) == "satisfactory":
                label = 2.0

            imagePaths = glob.glob(os.path.join(self.datasetPath, subFolder, "*.jpg"))
            for imagePath in imagePaths:
                if label is not None:
                    allImages.append([imagePath, label])
                else:
                    raise InvalidLabel()

        return allImages

    def trainValTestSplit(self):
        totalImages = self.combineImagesAndLabels()
        trainData, valData, testData = data.random_split(totalImages, [self.trainRatio, self.valRatio, self.testRatio])
        return trainData, valData, testData


class TrainingModel:
    def __init__(self):
        trainData, valData, testData = ImagesAndLabels().trainValTestSplit()
        print(len(trainData))
        print(len(valData))
        print(len(testData))
        self.batchSize = 64
        self.trainLoader = DataLoader(dataset=trainData, batch_size=self.batchSize, shuffle=True, num_workers=4)
        self.valLoader = DataLoader(dataset=valData, batch_size=self.batchSize, shuffle=True, num_workers=4)
        self.testLoader = DataLoader(dataset=testData, batch_size=self.batchSize, shuffle=True, num_workers=4)
        self.classes = ("unsatisfactory", "moderatelysatisfactory", "satisfactory")

    def imshow(self, img):
        img = img / 2 + 0.5  # unnormalize
        npImg = img.numpy()
        plt.imshow(np.transpose(npImg, (1, 2, 0)))
        plt.show()

    def displayImages(self):
        dataiter = iter(self.trainLoader)
        images, labels = next(dataiter)

        # show images
        self.imshow(utils.make_grid(images))
        # print labels
        print(' '.join(f'{self.classes[labels[j]]:5s}' for j in range(self.batchSize)))


def main():
    TrainingModel().displayImages()


if __name__ == "__main__":
    main()







