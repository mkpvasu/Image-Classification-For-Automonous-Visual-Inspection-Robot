import os
import cv2


class PartitionImages:
    def __init__(self):
        super(PartitionImages, self).__init__()
        self.capturedImgName = "image_1.jpg"
        self.capturedImgPath = None
        self.PImgsPath = os.path.join(os.getcwd(), 'Partitioned_Images')
        self.rowsPImg = 528
        self.columnsPImg = 528

    def partitionImage(self):
        self.readCapturedImage()
        if not os.path.exists(self.PImgsPath):
            os.mkdir(self.PImgsPath)
        imgName = os.path.basename(self.capturedImgName).split('.')[0]
        img = cv2.imread(self.capturedImgPath)
        partitionedImgs = [img[x:x+self.rowsPImg, y:y+self.columnsPImg] for x in range(0, img.shape[0], self.rowsPImg)
                           for y in range(0, img.shape[1], self.columnsPImg)]
        for i in range(len(partitionedImgs)):
            PImgName = imgName + '_' + str(i + 1) + '.jpg'
            PImgPath = os.path.join(self.PImgsPath, PImgName)
            cv2.imwrite(PImgPath, partitionedImgs[i])

    # READING ALL FULL RESOLUTIONS IMAGES (9504 * 6336) AS LIST
    def readCapturedImage(self):
        capturedImgPath = os.path.join(os.getcwd(), 'Inspection', self.capturedImgName)
        if not os.path.exists(capturedImgPath):
            raise FileNotFoundError(f"[{self.capturedImgName}] Doesn't Exist in Inspection Directory")
        self.capturedImgPath = capturedImgPath


def main():
    PartitionImages().partitionImage()


if __name__ == "__main__":
    main()