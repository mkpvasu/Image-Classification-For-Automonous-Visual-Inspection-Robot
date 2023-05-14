import glob
import json
import os
import cv2
import numpy as np
from PIL import Image
from transforms import RGBTransform


class RestoreToOriginalImage:
    def __init__(self, sub_images_dir, original_image_size: tuple, image_channels=3):
        super(RestoreToOriginalImage, self).__init__()
        self.subImagesDir = os.path.normpath(sub_images_dir)
        self.originalImageHeight, self.originalImageWidth = original_image_size
        self.originalImageChannels = image_channels

    # RESTORE TO ORIGINAL IMAGE
    def restoreImage(self):
        subImagesPath = sorted(glob.glob(os.path.join(self.subImagesDir, "*.jpg")))
        if not len(subImagesPath):
            raise RuntimeError(f"No sub-images in the [{self.subImagesDir}] directory")
        sampleSubImage = cv2.imread(subImagesPath[0])
        patchHeight, patchWidth, patchChannels = sampleSubImage.shape
        classificationOutputs = self.readClassifications()

        if patchChannels != self.originalImageChannels:
            self.originalImageChannels = patchChannels
        originalImage = Image.new("RGB", (self.originalImageWidth, self.originalImageHeight))

        subImageCols = self.originalImageWidth / patchWidth
        subImageRows = self.originalImageHeight / patchHeight

        if not subImageCols.is_integer() or not subImageRows.is_integer():
            raise RuntimeError("All patches should have same size")

        subImageRows = int(subImageRows)
        subImageCols = int(subImageCols)

        for rows in range(0, self.originalImageHeight, patchHeight):
            for cols in range(0, self.originalImageWidth, patchWidth):
                subImgPath = subImagesPath[(rows // patchHeight) * subImageCols + (cols // patchWidth)]
                subImgClassification = classificationOutputs[os.path.basename(subImgPath)]
                subImg = Image.open(subImgPath)
                subImg = subImg.convert("RGB")
                tintedSubImg = None

                if float(subImgClassification) == 0.0:
                    tintedSubImg = RGBTransform().tintWith((255, 0, 0), factor=0.2).applyToImg(subImg)
                elif float(subImgClassification) == 1.0:
                    tintedSubImg = RGBTransform().tintWith((255, 255, 0), factor=0.2).applyToImg(subImg)
                elif float(subImgClassification) == 2.0:
                    # tintedSubImg = RGBTransform().mix_with((0, 255, 0), factor=0.4).applied_to(subImg)
                    tintedSubImg = subImg

                originalImage.paste(tintedSubImg, (cols, rows))

        imagePath = os.path.join(self.subImagesDir, os.path.basename(self.subImagesDir) + "_restored.jpg")
        originalImage.save(imagePath)

    def readClassifications(self):
        classificationJsonPath = os.path.join(self.subImagesDir, "classification.json")
        if not os.path.exists(classificationJsonPath):
            raise RuntimeWarning(f"Classification JSON file doesn't exist in {self.subImagesDir}")
        with open(classificationJsonPath, "r") as inputFile:
            classifications = json.load(inputFile)
        return classifications


def main():
    RestoreToOriginalImage(
        sub_images_dir=r"C:\Macs Lab\imageClassification\data\partitioned_images\batch0\part0\30_micron\pass_1\capt0000",
        original_image_size=(6336, 9504)).restoreImage()


if __name__ == "__main__":
    main()
