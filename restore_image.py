import glob
import os
import cv2
import numpy as np


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

        if patchChannels != self.originalImageChannels:
            self.originalImageChannels = patchChannels
        originalImage = np.ndarray(shape=(self.originalImageHeight, self.originalImageWidth, self.originalImageChannels))

        subImageCols = self.originalImageWidth / patchWidth
        subImageRows = self.originalImageHeight / patchHeight

        if not subImageCols.is_integer() or not subImageRows.is_integer():
            raise RuntimeError("All patches should have same size")

        subImageRows = int(subImageRows)
        subImageCols = int(subImageCols)

        for rowIdx in range(subImageRows):
            for colIdx in range(subImageCols):
                subImg = cv2.imread(subImagesPath[rowIdx * subImageCols + colIdx])
                for channel in range(self.originalImageChannels):
                    originalImage[rowIdx * patchHeight:(rowIdx + 1) * patchHeight, colIdx * patchWidth:(colIdx + 1) * patchWidth, channel] = subImg[:, :, channel]

        imagePath = os.path.join(r"G:\Macs Lab\imageClassification\data\Partitioned_Images", os.path.basename(self.subImagesDir) + ".jpg")
        cv2.imwrite(imagePath, originalImage)


def main():
    RestoreToOriginalImage(
        sub_images_dir=r"G:\Macs Lab\imageClassification\data\Partitioned_Images\batch0\part0\30\pass_1\capt0000",
        original_image_size=(6336, 9504)).restoreImage()


if __name__ == "__main__":
    main()