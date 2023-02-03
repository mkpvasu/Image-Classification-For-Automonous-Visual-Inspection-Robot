import os
import glob
import shutil
import random


class DatasetForModel:
    def __init__(self, past_data_inclusion_ratio: float = 0.5, current_data_train_test_split:tuple = (0.9, 0.1)):
        if not isinstance(past_data_inclusion_ratio, (int, float)):
            raise RuntimeError("past_data_inclusion_ratio - input argument should be int or float")
        if not isinstance(current_data_train_test_split, tuple):
            raise RuntimeError("current_data_train_test_split should be a tuple")

        self.pastDataRatio = past_data_inclusion_ratio
        self.currentDataPath = os.path.join(os.getcwd(), "data", "Dataset_Preparation")
        self.datasetForModelPath = os.path.join(os.getcwd(), "data", "Dataset_Preparation", "DatasetForModel")
        self.currentDataTrainingRatio, self.currentDataTestRatio = current_data_train_test_split

    def prepareDataset(self):
        data, combinedData = {}, {
                                  "Train_Set": {"30_micron": {"Good": [], "Marginal": [], "Bad": []},
                                                "15_micron": {"Good": [], "Marginal": [], "Bad": []},
                                                "9_micron": {"Good": [], "Marginal": [], "Bad": []}},
                                  "Test_Set": {"30_micron": {"Good": [], "Marginal": [], "Bad": []},
                                               "15_micron": {"Good": [], "Marginal": [], "Bad": []},
                                               "9_micron": {"Good": [], "Marginal": [], "Bad": []}}
                                  }
        dataTypeFolders = [directory for directory in glob.glob(os.path.join(self.currentDataPath, "*")) if os.path.isdir(directory)]
        for dataTypeFolder in dataTypeFolders:
            dataType = os.path.basename(dataTypeFolder)
            if dataType in ["Past_Data", "Current_Data"]:
                dataPath = os.path.join(self.currentDataPath, dataType)
                dataFolders = [path for path in glob.glob(os.path.join(dataPath, "*")) if os.path.isdir(path)]

                for setPath in dataFolders:
                    setName = os.path.basename(setPath)
                    if setName in ["Train_Set", "Test_Set", "Total_Set"]:
                        data[setName] = {}
                        setFolders = [path for path in glob.glob(os.path.join(setPath, "*")) if os.path.isdir(path)]

                        for micronPath in setFolders:
                            micronName = os.path.basename(micronPath)
                            if micronName in ["30_micron", "15_micron", "9_micron"]:
                                data[setName][micronName] = {}
                                micronFolder = [path for path in glob.glob(os.path.join(micronPath, "*")) if os.path.isdir(path)]

                                for classPath in micronFolder:
                                    className = os.path.basename(classPath)
                                    if className in ["Good", "Marginal", "Bad"]:
                                        if dataType == "Past_Data" and setName in ["Train_Set", "Test_Set"]:
                                            data[setName][micronName][className] = \
                                                glob.glob(os.path.join(classPath, "*.jpg"))
                                            random.shuffle(data[setName][micronName][className])
                                            numberOfImages = \
                                                int(self.pastDataRatio * len(data[setName][micronName][className]))
                                            combinedData[setName][micronName][className].extend(
                                                data[setName][micronName][className][:numberOfImages])

                                        elif dataType == "Current_Data" and setName == "Total_Set":
                                            data[setName][micronName][className] = \
                                                glob.glob(os.path.join(classPath, "*.jpg"))
                                            random.shuffle(data[setName][micronName][className])
                                            numberOfTrainImages = int(self.currentDataTrainingRatio *
                                                                      len(data[setName][micronName][className]))
                                            combinedData["Train_Set"][micronName][className].\
                                                extend(data[setName][micronName][className][:numberOfTrainImages])
                                            combinedData["Test_Set"][micronName][className]. \
                                                extend(data[setName][micronName][className][numberOfTrainImages:])
        
        return combinedData

    def makeDirsModelDatasetDir(self):
        preparedDataset = self.prepareDataset()
        for set in preparedDataset.keys():
            for micron in preparedDataset[set].keys():
                for cls in preparedDataset[set][micron].keys():
                    if os.path.exists(os.path.join(self.datasetForModelPath, set, micron, cls)):
                        shutil.rmtree(os.path.join(self.datasetForModelPath, set, micron, cls))
                    os.makedirs(os.path.join(self.datasetForModelPath, set, micron, cls))

    def createDirsAndCopyImages(self):
        preparedDataset = self.prepareDataset()
        self.makeDirsModelDatasetDir()
        for set in preparedDataset.keys():
            for micron in preparedDataset[set].keys():
                for cls in preparedDataset[set][micron].keys():
                    for image in preparedDataset[set][micron][cls]:
                        shutil.copy(image, os.path.join(self.datasetForModelPath, set, micron, cls))


def main():
    DatasetForModel().createDirsAndCopyImages()


if __name__ == "__main__":
    main()






        
                                        








