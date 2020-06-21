import keras.backend as K


def acc_coef(y_true, y_pred, scores):
    pred_y = K.argmax(scores, -1)
    true_y = K.argmax(y_true, -1)
    count = K.equal(pred_y, true_y)

    return K.mean(K.cast(count, "float"))


def accuracy(scores):
    def acc(y_true, y_pred):
        return acc_coef(y_true, y_pred, scores)

    return acc