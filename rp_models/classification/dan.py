from keras import optimizers
from keras.layers import Embedding, Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

from layers.custom_layer import AverageWords, WordDropout
from rp_models.classification.relation_classification import RelationClassification


class DAN(RelationClassification):

    def create_model(self, lr, layers, units, dropout, word_dropout, train_emb):
        model = Sequential()
        model.add(Embedding(self._nb_words, self._embed_dim,
                            weights=[self._embedding_matrix], mask_zero=True, input_length=self._max_seq_len,
                            trainable=train_emb))

        if word_dropout != 0:
            print("Dropping")
        model.add(WordDropout(word_dropout))  # drops word token

        # if you want a different dropout than the one above
        # just uncomment one of the following dropout variations

        #      model.add(TimestepDropout(word_dropout))# drops word type
        #      model.add(SpatialDropout1D(word_dropout))# drops whole row
        #    model.add(Masking(mask_value=0.))

        model.add(AverageWords())
        for i in range(layers):
            model.add(Dense(units[i]))
            model.add(BatchNormalization())
            model.add(Activation("relu"))
            model.add(Dropout(rate=dropout[i], seed=123))

        model.add(Dense(self._num_classes, activation="softmax"))

        adam = optimizers.Adam(lr=lr)
        model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

        self.model = model
