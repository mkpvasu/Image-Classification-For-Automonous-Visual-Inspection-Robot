import os
import glob
import cv2


def main():
    imagesPath = r"G:\Macs Lab\imageClassification\data\Dataset_Preparation\DatasetForModel\Test_Set\30_micron"
    images = glob.glob(os.path.join(imagesPath, "**", "*.jpg"))

    for image in images:
        img = cv2.imread(image)
        if img.shape != (528, 528, 3):
            os.remove(image)


if __name__ == "__main__":
    main()