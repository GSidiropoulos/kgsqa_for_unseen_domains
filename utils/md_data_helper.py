import ast

import numpy as np
import pandas as pd
from keras.utils import to_categorical

from utils.data_io import DataSaverLoader
from utils.model_utils import ModelUtils


class DataHelperMD(object):

    def __init__(self, path_data,
                 max_seq_len=36, embed_dim=300, n_tags=2, ix2word=None, version=1, target_domain="all",
                 apply_on_train=True):

        if version == 1:
            self._max_seq_len = max_seq_len
            self._n_tags = n_tags
            self._apply_on_train = apply_on_train
            self._target_domain = target_domain

            self.ix2word = DataSaverLoader.load_pickle(path=path_data, filename="ix2word")

            self.x_train, self.y_train, self.relations_train = self._load_data(path_data, "train")
            self.x_test, self.y_test, self.relations_test = self._load_data(path_data, "test")
            self.x_valid, self.y_valid, self.relations_valid = self._load_data(path_data, "valid")

            if target_domain != "all":
                self.skip_rows = dict()
                if self._apply_on_train:
                    ids = []
                    for index, relation in enumerate(self.relations_train):
                        if target_domain == relation.split("/")[0]:
                            ids.append(index)
                    array_ids = np.array(ids)
                    self.x_train = np.delete(self.x_train, array_ids, axis=0)
                    self.y_train = np.delete(self.y_train, array_ids, axis=0)
                    self.relations_train = np.delete(np.array(self.relations_train), array_ids, axis=0)

                self.x_valid, self.y_valid, self.relations_valid = self._keep_on_inference(self.x_valid, self.y_valid,
                                                                                           self.relations_valid, "valid")
                self.x_test, self.y_test, self.relations_test = self._keep_on_inference(self.x_test, self.y_test,
                                                                                        self.relations_test, "test")

        elif version == 2:
            self.x_train, self.y_train = self._load(path_data, "train")
            self.x_test, self.y_test = self._load(path_data, "test")
            self.x_valid, self.y_valid = self._load(path_data, "valid")

    def _load_data(self, path, dataset_name):
        """
        :param path: path to data
        :param dataset_name: "train", "test", "validation"
        :return: np arrays of data, targets and relations (text)
        """

        df = pd.read_csv(path + dataset_name + "/data.csv",
                         usecols=[1, 3, 4])  # , names=["relation","data","annotation"])
        x = df["data"].apply(ast.literal_eval).to_list()
        y = df["annotation"].apply(ast.literal_eval).to_list()
        r = df["relation"].to_list()
        r = np.array(r)

        # padding
        x = ModelUtils.pad_seq(x, self._max_seq_len)
        y = ModelUtils.pad_seq(y, self._max_seq_len)

        # categorical values for labels
        y = [to_categorical(i, num_classes=self._n_tags) for i in y]
        y = np.array(y)

        return x, y, r

    def _load(self, path, dataset_name):
        x = DataSaverLoader.load_pickle(path=path + dataset_name + "/", filename="data")
        y = DataSaverLoader.load_pickle(path=path + dataset_name + "/", filename="targets")

        return x, y

    def _keep_on_inference(self, x, y, relations, set_name):
        ids = []
        skip_ids = []
        for index, relation in enumerate(relations):
            if self._target_domain == relation.split("/")[0]:
                ids.append(index)
            else:
                skip_ids.append(index+1)

        array_ids = np.array(ids)

        x = x[array_ids]
        y = y[array_ids]
        relations = relations[array_ids]

        self.skip_rows[set_name] = skip_ids

        return x, y, relations
