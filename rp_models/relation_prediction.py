import numpy as np
import tensorflow as tf
import random as rn
from keras import backend as K

from abc import ABC, abstractmethod
from keras.models import load_model, save_model

from utils.data_io import DataSaverLoader
from utils.metrics import Metrics
from utils.model_utils import ModelUtils


class RelationPredictionModel(ABC):

    def __init__(self, max_seq_len=36, embed_dim=300, num_classes=1837, ix2word=None):

        self._max_seq_len = max_seq_len
        self._embed_dim = embed_dim
        self._num_classes = num_classes
        self._ix2word = ix2word

        self._embedding_matrix = ModelUtils.emb_matrix(self._ix2word)
        self._nb_words = self._embedding_matrix.shape[0]

        super().__init__()

    @abstractmethod
    def create_model(self):
        pass

    def load_trained_model(self, path, name="model", custom_obj=None):
        self.model = load_model(path + name + ".h5", custom_objects=custom_obj)

    def save_model_(self, path, name):
        # check if dir exist
        DataSaverLoader.directory_exists(path)
        # save model
        self.model.save(path + name + ".h5")

    def relation_pred_wrt_cand(self, predictions, y_true, predicates_per_question):
        """Relation prediction w.r.t subject entity candidates"""
        y_true = np.argmax(y_true, axis=-1)
        predictions_final = []
        for i in range(len(y_true)):
            sorted_predictions = predictions[i].argsort()[::-1]
            for p in sorted_predictions:

                if len(predicates_per_question[i]) < 1:
                    predictions_final.append(p)
                    break
                else:
                    if p in predicates_per_question[i]:
                        predictions_final.append(p)
                        break

        count = 0
        for i in range(len(predictions_final)):
            if predictions_final[i] == y_true[i]:
                count += 1
        print("Relation Prediction Accuracy: ", count / len(predictions_final))

        return predictions_final

    def get_metrics(self, predictions, y, ix2pred, pred2ix):
        # Metrics
        #    predictions = self.model.predict_classes(x)
        y_ = np.argmax(a=y, axis=-1)
        print(np.mean(np.equal(predictions, y_)))

        Metrics.report(y_, predictions, [i for i in ix2pred], [i for i in pred2ix], True)
        print(Metrics.pr_rc_fscore_sup(y_, predictions, "micro", True))
        print(Metrics.pr_rc_fscore_sup(y_, predictions, "macro", True))

    def clear(self):
        """Clear the session and set all the seeds"""
        K.clear_session()
        np.random.seed(42)
        rn.seed(12345)

        #    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
        #                                  inter_op_parallelism_threads=1)

        tf.set_random_seed(1234)

        #    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        #    K.set_session(sess)

        sess = tf.Session(graph=tf.get_default_graph())
        K.set_session(sess)
