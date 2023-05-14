import os
import glob
import shutil
import pandas as pd


class ImagesToLabelsFolder:
    def __init__(self, excel_file):
        self.excel_file = excel_file
        self.micron_paths = {}

    def copy_images_to_micron_folder(self):
        image_paths = glob.glob(os.path.join(os.getcwd(), 'images', '*.jpg'))
        # micron_paths = {}

        for micron in ["20_micron", "10_micron", "5_micron"]:
            micron_path = os.path.join(os.getcwd(), micron)
            if not os.path.exists(micron_path):
                os.mkdir(micron_path)
            self.micron_paths[micron] = micron_path

        # for image_path in image_paths:
        #     if "ImagingLayer_3" in image_path:
        #         shutil.copy(image_path, micron_paths["20_micron"])
        #     elif "ImagingLayer_6" in image_path:
        #         shutil.copy(image_path, micron_paths["10_micron"])
        #     elif "ImagingLayer_9" in image_path:
        #         shutil.copy(image_path, micron_paths["5_micron"])

        # return micron_paths

    def convert_images_to_doe_folders(self):
        for micron_path in self.micron_paths:
            image_paths = glob.glob(os.path.join(self.micron_paths[micron_path], "*.jpg"))
            for i in range(1, 16):
                doe_number = "DoE-" + str(i)
                doe_path = os.path.join(self.micron_paths[micron_path], doe_number)
                if not os.path.exists(doe_path):
                    os.mkdir(doe_path)
                for path in image_paths:
                    if doe_number+"_" in path:
                        shutil.move(path, doe_path)

    def read_excel_and_create_label_folders(self):
        xls = pd.ExcelFile(self.excel_file)
        label_map = {1: "Good", 2: "Marginal", 3: "Bad"}

        for micron in self.micron_paths:
            row_number = 0
            if micron == "20_micron":
                row_number = 7
            elif micron == "10_micron":
                row_number = 10
            elif micron == "5_micron":
                row_number = 13

            for val in label_map.values():
                label_path = os.path.join(self.micron_paths[micron], val)
                if not os.path.exists(label_path):
                    os.mkdir(label_path)

            for i in range(1, 16):
                doe = f"DoE-{i}"
                df = pd.read_excel(xls, sheet_name=f"DOE-{i}")
                images = glob.glob(os.path.join(self.micron_paths[micron], doe, "*.jpg"))
                for j in range(len(images)):
                    label = int(df.iat[row_number, 18+j])
                    label_path = os.path.join(self.micron_paths[micron], label_map[label])
                    shutil.copy(images[j], label_path)

    def execute(self):
        self.copy_images_to_micron_folder()
        # self.convert_images_to_doe_folders()
        self.read_excel_and_create_label_folders()


def main():
    excel_file = os.path.join(os.getcwd(), "Flat Sheet DOE.xlsx")
    ImagesToLabelsFolder(excel_file=excel_file).execute()


if __name__ == "__main__":
    main()
