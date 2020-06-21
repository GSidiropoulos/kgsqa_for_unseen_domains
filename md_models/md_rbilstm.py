from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional, Add
from keras.models import Model, Input

from md_models.mention_detection import MentionDetection


class MDResidualBiLSTM(MentionDetection):

    def create_model(self, layers=2, rec_dropout=[0.2, 0.2], rnn_dropout=[0.4, 0.4], units=[600, 600], lr=0.001,
                     train_emb=False):
        input = Input(shape=(self._max_seq_len,))
        embedding = Embedding(self._nb_words, self._embed_dim, weights=[self._embedding_matrix], mask_zero=True,
                              trainable=train_emb)(input)

        # first layer
        x = Bidirectional(
            LSTM(units=units[0], return_sequences=True, recurrent_dropout=rec_dropout[0], dropout=rnn_dropout[0]))(
            embedding)
        # second layer
        x_rnn = Bidirectional(
            LSTM(units=units[1], return_sequences=True, recurrent_dropout=rec_dropout[1], dropout=rnn_dropout[1]))(x)

        # residual connection
        x_add = Add()([x, x_rnn])

        out = TimeDistributed(Dense(self._n_tags, activation="softmax"))(x_add)
        model = Model(input, out)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        self.model = model

    def grid_search(self, x_train, y_train, x_valid, y_valid, param_grid):
        """
        Use this method in order to do a manual grid search of the parameters and hyperparameters.

        :param x_train: train data
        :param y_train: train targets
        :param x_valid: validation data
        :param y_valid: validation targets
        :param param_grid: a dictionary where key-> hyperparameter/parameter name and value-> the corresponding value
        :return: dictionary where key->param_grid combination to_string value-> trained model
        """

        models = dict()
        for epochs in param_grid["epochs"]:
            for batch in param_grid["batch"]:
                for train_emb in param_grid["train_emb"]:
                    for units in param_grid["units"]:
                        for rnn_dropout in param_grid["rnn_dropout"]:
                            for lr in param_grid["lr"]:
                                key = "Epochs: %d, Batch: %d, Train_emb: %s, Units: %d, RNN_drop: %s, Lr: %s," % (
                                    epochs, batch, train_emb, units, str(rnn_dropout), str(lr))

                                print(key)

                                self.create_model(rnn_dropout=rnn_dropout,
                                                  units=units, lr=lr, train_emb=train_emb)

                                self.train(x_train=x_train, y_train=y_train,
                                           x_valid=x_valid, y_valid=y_valid,
                                           batch_size=batch, epochs=epochs)
                                self.evaluate_on_test_set(x_valid, y_valid)

                                models[key] = self.model

                                self.clear()

        return models
