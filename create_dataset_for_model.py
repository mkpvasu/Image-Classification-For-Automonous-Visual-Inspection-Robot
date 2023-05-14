import os
import glob
import shutil
import random


class DatasetForModel:
    """
    Class to create model for dataset from newly collected data and also include past data if necessary
    """
    def __init__(self, data_path: str, past_data_inclusion_ratio: float = 0, 
                 current_data_train_test_split:tuple = (0.9, 0.1)):
        
        if not isinstance(past_data_inclusion_ratio, (int, float)):
            raise RuntimeError("past_data_inclusion_ratio - input argument should be int or float")
        if not isinstance(current_data_train_test_split, tuple):
            raise RuntimeError("current_data_train_test_split should be a tuple")

        self.past_data_inclusion_ratio = past_data_inclusion_ratio
        self.current_data_path = data_path
        self.dataset_for_model_path = os.path.join(data_path, "dataset_for_model")
        self.current_data_training_ratio, self.current_data_test_ratio = current_data_train_test_split
        self.microns = ["20_micron", "10_micron", "5_micron"]
        self.data_names = ["past_data", "current_data"]
        self.labels = ["Good", "Marginal", "Bad"]
        self.sets = ["train_set", "test_set"]
        self.all_sets = ["train_set", "test_set", "total_set"]

    def prepare_dataset(self):
        data, combined_data = {}, {}
        
        for micron in self.microns:
            combined_data[micron] = {}
            for _set in self.all_sets:
                combined_data[micron][_set] = {"Good": [], "Marginal": [], "Bad": []}
        
        data_type_folders = [directory for directory in glob.glob(os.path.join(self.current_data_path, "*")) if os.path.isdir(directory)]
        for data_type_folder in data_type_folders:
            data_type = os.path.basename(data_type_folder)
            if data_type in self.data_names:
                data_path = os.path.join(self.current_data_path, data_type)
                if not os.listdir(data_path):
                    continue
                micron_dirs = [path for path in glob.glob(os.path.join(data_path, "*")) if os.path.isdir(path)]

                for micron_dir in micron_dirs:
                    micron = os.path.basename(micron_dir)
                    if micron in self.microns:
                        data[micron] = {}
                        set_dirs = [path for path in glob.glob(os.path.join(micron_dir, "*")) if os.path.isdir(path)]

                        for set_dir in set_dirs:
                            _set = os.path.basename(set_dir)
                            if _set in self.all_sets:
                                data[micron][_set] = {}
                                label_dirs = [path for path in glob.glob(os.path.join(set_dir, "*")) if os.path.isdir(path)]

                                for label_dir in label_dirs:
                                    label = os.path.basename(label_dir)
                                    if label in self.labels:
                                        if data_type == "past_data" and _set in self.sets:
                                            data[micron][_set][label] = \
                                                glob.glob(os.path.join(label_dir, "*.jpg"))
                                            random.shuffle(data[micron][_set][label])
                                            number_of_images = \
                                                int(self.past_data_inclusion_ratio * len(data[micron][_set][label]))
                                            combined_data[micron][_set][label].extend(
                                                data[micron][_set][label][:number_of_images])

                                        elif data_type == "current_data" and _set == "total_set":
                                            data[micron][_set][label] = \
                                                glob.glob(os.path.join(label_dir, "*.jpg"))
                                            random.shuffle(data[micron][_set][label])
                                            number_of_train_images = int(self.current_data_training_ratio * 
                                                                         len(data[micron][_set][label]))
                                            combined_data[micron]["train_set"][label].\
                                                extend(data[micron][_set][label][:number_of_train_images])
                                            combined_data[micron]["test_set"][label]. \
                                                extend(data[micron][_set][label][number_of_train_images:])
        
        return combined_data

    def create_dirs_for_dataset(self):
        prepared_dataset = self.prepare_dataset()
        if not os.path.exists(self.dataset_for_model_path):
            os.mkdir(self.dataset_for_model_path)
        for micron in prepared_dataset.keys():
            for _set in self.sets:
                for cls in prepared_dataset[micron][_set].keys():
                    if os.path.exists(os.path.join(self.dataset_for_model_path, micron, _set, cls)):
                        shutil.rmtree(os.path.join(self.dataset_for_model_path, micron, _set, cls))
                    os.makedirs(os.path.join(self.dataset_for_model_path, micron, _set, cls))

    def copy_images(self):
        prepared_dataset = self.prepare_dataset()
        self.create_dirs_for_dataset()
        for micron in prepared_dataset.keys():
            for _set in prepared_dataset[micron].keys():
                for cls in prepared_dataset[micron][_set].keys():
                    for image in prepared_dataset[micron][_set][cls]:
                        shutil.copy(image, os.path.join(self.dataset_for_model_path, micron, _set, cls))


def main():
    data_path = os.path.join(os.getcwd(), "data", "dataset_preparation")
    DatasetForModel(data_path=data_path).copy_images()


if __name__ == "__main__":
    main()