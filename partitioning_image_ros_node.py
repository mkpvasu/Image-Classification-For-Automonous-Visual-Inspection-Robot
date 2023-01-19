import os
import cv2
import shutil


class PartitionImages:
    def __init__(self, image_path, sub_image_height=528, sub_image_width=528):
        super(PartitionImages, self).__init__()
        self.partitionImagesDir = None
        self.capturedImgPath = image_path
        self.capturedImgName = os.path.splitext(os.path.basename(image_path))[0]
        self.rowsPImg = sub_image_height
        self.columnsPImg = sub_image_width
        self.setPartitionImagesPath()

    # PARTITION FULL RESOLUTION IMAGE
    def partitionImage(self):
        
        if not os.path.exists(self.capturedImgPath):
            raise FileNotFoundError(f"[{self.capturedImgName}]: Doesn't Exist in the Specified Folder")
        
        savePartitionedImagePath = os.path.join(self.partitionImagesDir, self.capturedImgName)

        # IF PARTITIONED IMAGE FOLDER ALREADY EXISTS, REMOVE IT AND CREATE NEW ONE
        if os.path.exists(savePartitionedImagePath):
            shutil.rmtree(savePartitionedImagePath)
        os.makedirs(savePartitionedImagePath)

        img = cv2.imread(self.capturedImgPath)
        partitionedImgs = [img[x:x+self.rowsPImg, y:y+self.columnsPImg] for x in range(0, img.shape[0], self.rowsPImg)
                           for y in range(0, img.shape[1], self.columnsPImg)]
        for i in range(len(partitionedImgs)):
            PImgName = self.capturedImgName + '_' + str(i + 1).zfill(3) + '.jpg'
            PImgPath = os.path.join(savePartitionedImagePath, PImgName)
            cv2.imwrite(PImgPath, partitionedImgs[i])

    # FIND PATH TO SAVE SUB-IMAGES FOR EACH IMAGE BASED ON FOLDERS CONVENTIONS
    def setPartitionImagesPath(self):
        capturedImagePath = os.path.normpath(os.path.dirname(self.capturedImgPath))
        capturedImagePathComponents = capturedImagePath.split(os.sep)
        savePartitionImagePathComponents = [component.replace("Full_Resolution_Images", "Partitioned_Images")
                                          for component in capturedImagePathComponents]
        partitionImagePassDir = savePartitionImagePathComponents[0] + os.sep
        savePartitionImagePathComponents.pop(0)
        for component in savePartitionImagePathComponents:
            partitionImagePassDir = os.path.join(partitionImagePassDir, component)
        self.partitionImagesDir = partitionImagePassDir


def main():
    PartitionImages(image_path=r"G:\Macs Lab\imageClassification\data\Full_Resolution_Images\batch0\part0\30\pass_1\capt0001.jpg").partitionImage()


if __name__ == "__main__":
    main()