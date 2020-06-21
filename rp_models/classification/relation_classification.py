import numpy as np

from rp_models.relation_prediction import RelationPredictionModel


class RelationClassification(RelationPredictionModel):

    def train(self, x_train, y_train, x_valid, y_valid, epochs=10, batch_size=64):
        # define callbacks
        #    early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=5, verbose=1)
        #    callbacks_list = [early_stopping]

        hist = self.model.fit(x_train, y_train,
                              batch_size=batch_size, epochs=epochs,
                              #                     callbacks=callbacks_list,
                              validation_data=(x_valid, y_valid), shuffle=True, verbose=1)

    def evaluate_test(self, x, y):
        predictions = self.model.predict(x, batch_size=64)
        print("Accuracy: ", np.sum(np.argmax(predictions, axis=-1) == np.argmax(y, axis=-1)) / y.shape[0])

        return predictions
