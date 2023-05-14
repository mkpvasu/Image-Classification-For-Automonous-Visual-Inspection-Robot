import os
import json
import glob
import random


def main():
    path = "partitioned_image_folder_path"
    files = glob.glob(os.path.join(path, "*.jpg"))
    baseNames, jsonDict = [], {}
    for file in files:
        baseNames.append(os.path.basename(file))
    for baseName in baseNames:
        jsonDict[baseName] = random.randint(0, 2)

    jsonFile = os.path.join(path, "classification.json")

    with open(jsonFile, "w") as outputFile:
        json.dump(jsonDict, outputFile, indent=2)
    outputFile.close()


if __name__ == "__main__":
    main()
