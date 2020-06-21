from keras import optimizers
from keras.layers import GRU, Bidirectional, Embedding, Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

from rp_models.classification.relation_classification import RelationClassification


class BiGRU(RelationClassification):

    def create_model(self, learning_rate, units, dropout, rnn_drop, train_emb, layers):
        model = Sequential()
        model.add(Embedding(self._nb_words, self._embed_dim,
                            weights=[self._embedding_matrix], mask_zero=True, input_length=self._max_seq_len,
                            trainable=train_emb))

        for l in range(layers - 1):
            print("More layers")
            model.add(Bidirectional(GRU(units=units[l], dropout=rnn_drop[l], return_sequences=True)))

        model.add(Bidirectional(GRU(units=units[-1], dropout=rnn_drop[-1])))

        model.add(Dense(units=units[-1] * 2))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(dropout, seed=123))

        model.add(Dense(self._num_classes, activation="softmax"))

        adam = optimizers.Adam(lr=learning_rate)  # ,amsgrad=True)
        model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

        self.model = model

    def grid_search(self, x_train, y_train, x_valid, y_valid, param_grid, none_sbj, predicates_per_question):
        models = dict()
        for opt in param_grid["optimizer"]:
            for batch in param_grid["batch_size"]:
                for epoch in param_grid["epoch"]:
                    for train_emb in param_grid["train_emb"]:
                        for layers in param_grid["layers"]:
                            for units in param_grid["units"]:
                                for lr in param_grid["learning_rate"]:
                                    for rnn_drop in param_grid["rnn_dropout"]:
                                        for drop in param_grid["dropout"]:
                                            print("Optimizer: ", opt, " Learning rate: ", lr, " Units: ", units,
                                                  " Dropout: ", drop, "RNN_Dropout: ", rnn_drop, " Epochs: ", epoch,
                                                  " Batch size: ", batch, " train_emb: ", train_emb, " Layers: ",
                                                  layers)
                                            self.create_model(optimizer=opt, learning_rate=lr, units=units,
                                                              dropout=drop, rnn_drop=rnn_drop, train_emb=train_emb,
                                                              layers=layers)
                                            self.train(x_train, y_train, x_valid, y_valid, epoch, batch)

                                            # predictions no constraint
                                            pred = self.evaluate_test(x_valid, y_valid)

                                            # predictions with constraint
                                            #                      y_valid_rm = np.delete(y_valid,none_sbj,axis=0)
                                            #                      pred = np.delete(pred,none_sbj,axis=0)
                                            self.relation_pred_wrt_cand(pred, y_valid, predicates_per_question)

                                            #                  model_name = opt+str(lr)+str(units)+str(drop)+str(batch)+str(epoch)
                                            #                  models[model_name] = self.model

                                            self.clear()

        return models
