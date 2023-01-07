import os
import glob
import cv2
import shutil


class PartitionImages:
    def __init__(self):
        super(PartitionImages, self).__init__()
        self.imgsPath = os.path.join(os.getcwd(), 'data', 'high_resolution_images')
        self.PImgsPath = os.path.join(os.getcwd(), 'data', 'partitioned_images')
        self.fullResolutionImgsPath = None
        self.rowsPImg = 528
        self.columnsPImg = 528

    def partitionImage(self):
        self.readImagesList()
        if os.path.exists(self.PImgsPath):
            shutil.rmtree(self.PImgsPath)
            os.mkdir(self.PImgsPath)
        else:
            os.mkdir(self.PImgsPath)
        for imgPath in self.fullResolutionImgsPath:
            imgName = os.path.basename(imgPath).split('.')[0]
            img = cv2.imread(imgPath)
            PImgs = [img[x:x+self.rowsPImg, y:y+self.columnsPImg] for x in range(0, img.shape[0], self.rowsPImg) for y in range(0, img.shape[1], self.columnsPImg)]
            for i in range(len(PImgs)):
                PImgName = imgName + '_' + str(i + 1) + '.jpg'
                PImgPath = os.path.join(self.PImgsPath, PImgName)
                cv2.imwrite(PImgPath, PImgs[i])

    # READING ALL FULL RESOLUTIONS IMAGES (9504 * 6336) AS LIST
    def readImagesList(self):
        fullResolutionImgPaths = glob.glob(os.path.join(self.imgsPath, '*.jpg'))
        if not fullResolutionImgPaths:
            print("No full resolution images present in the folder")
            return None
        self.fullResolutionImgsPath = fullResolutionImgPaths


def main():
    PartitionImages().partitionImage()


if __name__ == "__main__":
    main()