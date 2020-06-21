import keras
import numpy as np
import pandas as pd
import ast
from utils.data_io import DataSaverLoader
from utils.model_utils import ModelUtils


class DataHelper(object):

    def __init__(self, path_data, max_seq_len=36, embed_dim=300, num_classes=1837, max_label_words_len=17,
                 max_label_names_len=5, create_negatives=False, total_negatives=None, negatives_intersection=None,
                 domain_intersection=True, incl_names=False, remove_domains=None, one_go=False, noisy_questions=False):
        """
        :param path_data:
        :param max_seq_len: max length of question
        :param embed_dim: embedding size, 300 in our word2vec case
        :param num_classes: 1837 predicates
        :param max_label_words_len: max length of predicate token/word level
        :param max_label_names_len: max length of predicate name level
        :param create_negatives: create or not negative samples
        :param total_negatives: number of negatives
        :param negatives_intersection: how many should come from the intersection
        :param domain_intersection: if given then intersection is domain level, otherwise just lexical
        :param incl_names: include predicate names alongside word level
        :param remove_domains: which domain to remove
        :param one_go: creates a 3D matrix instead of a 2D
        :param noisy_questions: take into account synthetic generated data from the QG task
        """

        self._max_seq_len = max_seq_len
        self._embed_dim = embed_dim
        self._num_classes = num_classes
        self._max_label_words_len = max_label_words_len
        self._max_label_names_len = max_label_names_len
        self._create_negatives = create_negatives
        self._total_negatives = total_negatives
        self._negatives_intersection = negatives_intersection
        self._incl_names = incl_names
        self._remove_domains = remove_domains
        self._domain_intersection = domain_intersection
        self._one_go = one_go
        self._noisy_questions = noisy_questions

        self._print()

        self.ix2word = DataSaverLoader.load_pickle(path=path_data, filename="ix2word")
        self.word2ix = DataSaverLoader.load_pickle(path=path_data, filename="word2ix")
        self.ix2pred = DataSaverLoader.load_pickle(path=path_data, filename="ix2pred")
        self.pred2ix = DataSaverLoader.load_pickle(path=path_data, filename="pred2ix")
        self.pred_names = DataSaverLoader.load_pickle(path=path_data, filename="pred_names")

        if remove_domains is not None:
            self._pred_ids_in_domain = self._find_remove_domain_ids()
            self.ids_per_set = dict()

        if not create_negatives:
            self.x_train, self.y_train = self._load_data(path_data, "train")
            self.x_test, self.y_test = self._load_data(path_data, "test")
            self.x_valid, self.y_valid = self._load_data(path_data, "valid")

        else:
            self.pred_words = DataSaverLoader.load_pickle(path=path_data, filename="pred_words")
            self.pred_words_ids = [[self.word2ix[token] for token in predicate_label] for predicate_label in
                                   self.pred_words]

            if self._incl_names:
                # if model includes name level and word level for the predicate label
                self.pred_names_ids = [[self.word2ix[token] for token in predicate_label] for predicate_label in
                                       self.pred_names]

                self.x_train, self.y_train, self.pos_word_level_train, self.neg_word_level_train, self.pos_name_level_train, self.neg_name_level_train = self._load_data_with_neg(
                    path_data, "train")
                self.x_test, self.y_test, self.pos_word_level_test, self.neg_word_level_test, self.pos_name_level_test, self.neg_name_level_test = self._load_data_with_neg(
                    path_data, "test")
                self.x_valid, self.y_valid, self.pos_word_level_valid, self.neg_word_level_valid, self.pos_name_level_valid, self.neg_name_level_valid = self._load_data_with_neg(
                    path_data, "valid")

            else:
                # if model includes only word level for the predicate label
                self.x_train, self.y_train, self.pos_word_level_train, self.neg_word_level_train = self._load_data_with_neg(
                    path_data, "train")
                self.x_test, self.y_test, self.pos_word_level_test, self.neg_word_level_test, = self._load_data_with_neg(
                    path_data, "test")
                self.x_valid, self.y_valid, self.pos_word_level_valid, self.neg_word_level_valid = self._load_data_with_neg(
                    path_data, "valid")

        print(self.x_train.shape, self.y_train.shape)
        print(self.x_test.shape, self.y_test.shape)
        print(self.x_valid.shape, self.y_valid.shape)

    def _load_data(self, path, dataset_name):

        df = pd.read_csv(path + dataset_name + "/data.csv", usecols=[3, 4])
        x = df["data"].apply(ast.literal_eval).to_list()
        y = df["targets"]
        y = np.array(y)
        # padding
        x = keras.preprocessing.sequence.pad_sequences(x, maxlen=self._max_seq_len, dtype="int32", padding="post",
                                                       value=0)

        # remove domains if specified
        if self._remove_domains is not None:
            x, y = self._zero_shot_setting(path, dataset_name, x, y)

        # one hot
        y = keras.utils.to_categorical(y, num_classes=self._num_classes)

        return x, y

    def _load_data_with_neg(self, path, dataset_name):
        df = pd.read_csv(path + dataset_name + "/data.csv", usecols=[3, 4])
        x = df["data"].apply(ast.literal_eval).to_list()
        y = df["targets"]
        y = np.array(y)
        # padding
        x = keras.preprocessing.sequence.pad_sequences(x, maxlen=self._max_seq_len, dtype="int32", padding="post",
                                                       value=0)

        pred_intersect = ModelUtils.predicate_intersection_wrt_words(self.pred_words, self._domain_intersection)

        if self._remove_domains is not None:
            x, y = self._zero_shot_setting(path, dataset_name, x, y)
        if self._one_go and self._incl_names:
            # creates dataset in a way that distances of
            # negative samples per question can be computed in one-go
            pos_w_l, neg_w_l, pos_n_l, neg_n_l = ModelUtils.create_negatives2_onego(x, y, self._max_seq_len,
                                                                                    self._max_label_words_len,
                                                                                    self._max_label_names_len,
                                                                                    self._total_negatives[dataset_name],
                                                                                    self._negatives_intersection[
                                                                                        dataset_name],
                                                                                    pred_intersect, self.pred_words_ids,
                                                                                    self.pred_names_ids)

            # one hot
            y = keras.utils.to_categorical(y, num_classes=self._num_classes)

            return x, y, pos_w_l, neg_w_l, pos_n_l, neg_n_l

        elif (not self._one_go) and self._incl_names:
            # include both word level and name level
            x, y, pos_w_l, neg_w_l, pos_n_l, neg_n_l = ModelUtils.create_negatives2(x, y, self._max_seq_len,
                                                                                    self._max_label_words_len,
                                                                                    self._max_label_names_len,
                                                                                    self._total_negatives[dataset_name],
                                                                                    self._negatives_intersection[
                                                                                        dataset_name],
                                                                                    pred_intersect, self.pred_words_ids,
                                                                                    self.pred_names_ids)

            # one hot
            y = keras.utils.to_categorical(y, num_classes=self._num_classes)

            return x, y, pos_w_l, neg_w_l, pos_n_l, neg_n_l

        elif (not self._one_go) and (not self._incl_names):
            # include only word level
            x, y, pos_labels, neg_labels = ModelUtils.create_negatives(x, y, self._max_seq_len,
                                                                       self._max_label_words_len,
                                                                       self._total_negatives[dataset_name],
                                                                       self._negatives_intersection[dataset_name],
                                                                       pred_intersect, self.pred_words_ids)

            # one hot
            y = keras.utils.to_categorical(y, num_classes=self._num_classes)

            return x, y, pos_labels, neg_labels

    def _find_remove_domain_ids(self):
        ids = []
        for k, v in enumerate(self.pred_names):
            if v[0] in self._remove_domains:
                ids.append(k)

        return ids

    def _remove_from_train(self, x, y, datest):
        array_ids = [k for k, v in enumerate(y) if v in self._pred_ids_in_domain]
        x = np.delete(x, array_ids, axis=0)
        y = np.delete(y, array_ids, axis=0)
        self.ids_per_set[datest] = array_ids
        return x, y

    def _zero_shot_setting(self, path, dataset_name, x, y):
        # remove domains if specified
        if dataset_name == "train":
            x, y = self._remove_from_train(x, y, dataset_name)

            if self._noisy_questions:
                x_noisy = DataSaverLoader.load_pickle(path=path + dataset_name + "/", filename="x_noisy")
                y_noisy = DataSaverLoader.load_pickle(path=path + dataset_name + "/", filename="y_noisy")

                x_noisy = keras.preprocessing.sequence.pad_sequences(x_noisy, maxlen=self._max_seq_len, dtype="int32",
                                                                     padding="post", value=0)
                y_noisy = np.array(y_noisy)

                x = np.concatenate((x, x_noisy), axis=0)
                y = np.concatenate((y, y_noisy), axis=0)

        return x, y

    def _print(self):
        print("max_seq_len: ", self._max_seq_len)
        print("embed_dim: ", self._embed_dim)
        print("num_classes: ", self._num_classes)
        print("max_label_words_len: ", self._max_label_words_len)
        print("max_label_names_len: ", self._max_label_names_len)
        print("create_negatives: ", self._create_negatives)
        print("total_negatives: ", self._total_negatives)
        print("negatives_intersection: ", self._negatives_intersection)
        print("domain_intersection: ", self._domain_intersection)
        print("incl_names: ", self._incl_names)
        print("remove_domains: ", self._remove_domains)
        print("one_go: ", self._one_go)
        print("noisy_questions: ", self._noisy_questions)
