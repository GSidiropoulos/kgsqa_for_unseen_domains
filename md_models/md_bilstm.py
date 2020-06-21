from keras.layers import LSTM, Embedding, Dense, TimeDistributed
from keras.models import Model, Input

from md_models.mention_detection import MentionDetection


class MDBiLSTM(MentionDetection):

    def create_model(self, layers=2, rec_dropout=[0.2, 0.2], rnn_dropout=[0.5, 0], units=[300, 100], lr=0.001,
                     train_emb=False):
        print("training BiLSTM ...")
        input = Input(shape=(self._max_seq_len,))
        x_rnn = Embedding(self._nb_words, self._embed_dim,
                          weights=[self._embedding_matrix], mask_zero=True, trainable=train_emb)(input)

        for layer in range(layers):
            x_rnn = LSTM(units=units[layer], return_sequences=True,
                         recurrent_dropout=rec_dropout[layer], dropout=rnn_dropout[layer])(x_rnn)

        out = TimeDistributed(Dense(self._n_tags, activation="softmax"))(x_rnn)
        model = Model(input, out)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        self.model = model

    def grid_search(self, x_train, y_train, x_valid, y_valid, param_grid):

        models = dict()
        for epochs in param_grid["epochs"]:
            for batch in param_grid["batch"]:
                for train_emb in param_grid["train_emb"]:
                    for layers in param_grid["layers"]:
                        for units in param_grid["units"]:
                            for rec_dropout in param_grid["rec_dropout"]:
                                for rnn_dropout in param_grid["rnn_dropout"]:
                                    for lr in param_grid["lr"]:
                                        key = "Epochs: %d, Batch: %d, Train_emb: %s, Layers: %d, Units: %s, Rec_drop: %s, RNN_drop: %s, Lr: %s," % (
                                            epochs, batch, train_emb, layers, str(units), str(rec_dropout),
                                            str(rnn_dropout), str(lr))
                                        print(key)
                                        self.create_model(layers=layers, rec_dropout=rec_dropout,
                                                          rnn_dropout=rnn_dropout,
                                                          units=units, lr=lr, train_emb=train_emb)

                                        self.train(x_train=x_train, y_train=y_train,
                                                   x_valid=x_valid, y_valid=y_valid,
                                                   batch_size=batch, epochs=epochs)
                                        self.evaluate_on_test_set(x_valid, y_valid)

                                        models[key] = self.model
                                        self.clear()

        return models
