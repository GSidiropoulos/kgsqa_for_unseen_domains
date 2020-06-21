import keras.backend as K


def acc_coef(y_true, y_pred, scores_pos, scores_neg):
    count = K.greater(scores_pos, scores_neg)
    return K.mean(K.cast(count, "float"))


def accuracy(scores_pos, scores_neg):
    def acc(y_true, y_pred):
        return acc_coef(y_true, y_pred, scores_pos, scores_neg)

    return acc