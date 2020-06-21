import numpy as np

from rp_models.relation_prediction import RelationPredictionModel
from utils.model_utils import ModelUtils


class SimilarityModelWordsNames(RelationPredictionModel):

    def __init__(self, max_seq_len=36, embed_dim=300, num_classes=1837,
                 max_label_words_len=17, max_label_names_len=5, ix2word=None, pred_words_ids=None, pred_names_ids=None):

        self._max_label_words_len = max_label_words_len
        self._max_label_names_len = max_label_names_len
        self._pred_words_ids = pred_words_ids
        self._pred_names_ids = pred_names_ids

        super().__init__(max_seq_len, embed_dim, num_classes, ix2word)

    def train(self, x_train, y_train, pos_train, neg_train, pos_name_level_train, neg_name_level_train,
              x_valid, y_valid, pos_valid, neg_valid, pos_name_level_valid, neg_name_level_valid, num_epochs,
              batch_size=128):
        labels_word_tile_train = np.zeros((x_train.shape[0], 1, self._max_label_words_len))
        labels_name_tile_train = np.zeros((x_train.shape[0], 1, self._max_label_names_len))

        labels_word_tile_valid = np.zeros((x_valid.shape[0], 1, self._max_label_words_len))
        labels_name_tile_valid = np.zeros((x_valid.shape[0], 1, self._max_label_names_len))

        hist = self.model.fit(
            x=[x_train, pos_train, neg_train, labels_word_tile_train, pos_name_level_train, neg_name_level_train,
               labels_name_tile_train],
            y=y_train, batch_size=batch_size, epochs=num_epochs, verbose=1,
            validation_data=(
                [x_valid, pos_valid, neg_valid, labels_word_tile_valid, pos_name_level_valid, neg_name_level_valid,
                 labels_name_tile_valid], y_valid), shuffle=True)

    def evaluate_wrt_sbj_entity(self, x_test, y_test, pos_test, neg_test, pos_name_level_test, neg_name_level_test,
                                predicates_per_question):
        pred_words_ids = self._pred_words_ids
        pred_names_ids = self._pred_names_ids

        pred_words_ids = ModelUtils.pad_seq(pred_words_ids, self._max_label_words_len)
        pred_names_ids = ModelUtils.pad_seq(pred_names_ids, self._max_label_names_len)

        labels_tile_ = [np.expand_dims(np.array([pred_words_ids[cand] for cand in list(set(rel_cand))]), axis=0) for
                        rel_cand in predicates_per_question]

        labels_tile_names = [np.expand_dims(np.array([pred_names_ids[cand] for cand in list(set(rel_cand))]), axis=0)
                             for rel_cand in predicates_per_question]

        predictions = []
        for i in range(len(x_test)):
            if len(labels_tile_[i].shape) >= 3:

                prediction = self.model.predict(
                    [[x_test[i]], pos_test, neg_test, labels_tile_[i], pos_name_level_test, neg_name_level_test,
                     labels_tile_names[i]], batch_size=1)

                predictions.append(list(set(predicates_per_question[i]))[np.argmax(prediction, -1)[0]])
            else:
                predictions.append(0)

        print(np.mean(predictions == np.argmax(a=y_test, axis=-1)))

        return predictions

    def evaluate_test(self, x_test, y_test, pos_test, neg_test, pos_name_level_test, neg_name_level_test):
        pred_words_ids = self._pred_words_ids
        pred_names_ids = self._pred_names_ids

        pred_words_ids = ModelUtils.pad_seq(pred_words_ids, self._max_label_words_len)
        pred_names_ids = ModelUtils.pad_seq(pred_names_ids, self._max_label_names_len)

        labels_tile = np.tile(pred_words_ids, [x_test.shape[0], 1, 1])
        labels_tile_names = np.tile(pred_names_ids, [x_test.shape[0], 1, 1])

        predictions = self.model.predict(
            [x_test, pos_test, neg_test, labels_tile, pos_name_level_test, neg_name_level_test, labels_tile_names],
            batch_size=64, verbose=1)
        print('Accuracy: ', np.sum(np.argmax(predictions, axis=-1) == np.argmax(y_test, axis=-1)) / y_test.shape[0])
        return predictions
