from keras import backend as K


def hinge_coef(y_true, y_pred, scores_pos, scores_neg):
    return K.mean(K.maximum(0., (-scores_pos + scores_neg + 0.5)))


def hinge_loss(scores_pos, scores_neg):
    def hinge(y_true, y_pred):
        return hinge_coef(y_true, y_pred, scores_pos, scores_neg)

    return hinge