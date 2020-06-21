import keras.backend as K


def hinge_coef(scores_pos, scores_neg, num_pos, num_neg):
    metric_p = K.tile(K.expand_dims(scores_pos, axis=2), [1, 1, num_neg])
    metric_n = K.tile(K.expand_dims(scores_neg, axis=1), [1, num_pos, 1])
    delta = metric_n - metric_p

    loss_q_pos = K.sum(K.relu(0.5 + delta), axis=2)

    return K.sum(loss_q_pos)


def hinge_loss(scores_pos, scores_neg, num_pos, num_neg):
    def hinge(y_true, y_pred):
        return hinge_coef(scores_pos, scores_neg, num_pos, num_neg)
    return hinge