import json
import os
import pickle
import pandas as pd


class DataSaverLoader(object):

    @staticmethod
    def load_pickle(path, filename):
        with open(path + filename + ".pkl", "rb") as f:
            file = pickle.load(f)
        return file

    @staticmethod
    def save_pickle(path, name, python_object):
        DataSaverLoader.directory_exists(path)

        with open(path + name + ".pkl", "wb") as handle:
            pickle.dump(python_object, handle)

    @staticmethod
    def save_csv(path, name, python_object):
        DataSaverLoader.directory_exists(path)

        python_object.to_csv(path+name, index=False)

    @staticmethod
    def load_file(path, filename):
        with open(path + filename, "r") as file:
            data = file.read()
        return data

    @staticmethod
    def save_file(path, name, python_object):
        DataSaverLoader.directory_exists(path)

        with open(path + name, "wb") as file:
            file.write(python_object)

    @staticmethod
    def directory_exists(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def save_json(json_data, path, filename):
        DataSaverLoader.directory_exists(path)
        json_file = json.dumps(json_data, sort_keys=True, indent=1)
        with open(path + filename, "w") as outfile:
            outfile.write(json_file)

    @staticmethod
    def load_json(path, filename):
        with open(path + filename, "r") as outfile:
            json_data = json.load(outfile)

        return json_data
