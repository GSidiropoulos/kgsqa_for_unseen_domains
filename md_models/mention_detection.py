import random as rn
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from keras_contrib.utils import save_load_utils

from utils.data_io import DataSaverLoader
from utils.model_utils import ModelUtils


class MentionDetection(ABC):

    def __init__(self, max_seq_len=36, embed_dim=300, n_tags=2, ix2word=None):

        self._max_seq_len = max_seq_len
        self._embed_dim = embed_dim
        self._n_tags = n_tags
        self._ix2word = ix2word
        self._embedding_matrix = ModelUtils.emb_matrix(self._ix2word)
        self._nb_words = self._embedding_matrix.shape[0]

        super().__init__()

    @abstractmethod
    def create_model(self):
        pass

    def train(self, x_train, y_train, x_valid, y_valid, batch_size=256, epochs=100):
        # define callbacks
        #    early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=3, verbose=1)
        #    callbacks_list = [early_stopping]

        # model training with batch normalization
        hist = self.model.fit(x_train, y_train, batch_size=batch_size,
                              epochs=epochs, validation_data=(x_valid, y_valid),
                              shuffle=True, verbose=1)

    def evaluate_on_test_set(self, x_test, y_test):
        pred = self.model.predict(x_test)
        pred = np.argmax(pred, axis=-1)

        count = 0
        for i in range(pred.shape[0]):
            if np.array_equal(pred[i], np.argmax(y_test[i], axis=-1)):
                count += 1

        print("Accuracy: ", count / pred.shape[0])
        return pred

    def save_crf_model(self, path, name):
        # check if dir exist
        DataSaverLoader.directory_exists(path)
        # Save model
        save_load_utils.save_all_weights(self.model, path + name)

    def save_model_h5(self, path, name):
        # check if dir exist
        DataSaverLoader.directory_exists(path)
        # save model
        self.model.save(path + name + ".h5")

    def load_trained_model(self, path):
        self.model = load_model(path)

    def clear(self):
        """Clear the session and set all the seeds"""
        K.clear_session()
        np.random.seed(42)
        rn.seed(12345)

        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                      inter_op_parallelism_threads=1)

        tf.set_random_seed(1234)

        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)
