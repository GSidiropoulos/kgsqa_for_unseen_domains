import numpy as np
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics


class CRFModel(object):

    def create_model(self):
        crf = CRF(algorithm="lbfgs", c1=0.1,
                  c2=0.1,
                  all_possible_transitions=False)

        self.model = crf

    def train(self, x, y):
        self.model.fit(x, y)

    def evaluate(self, x, y):
        y_pred = self.model.predict(x)
        print(metrics.flat_accuracy_score(y, y_pred))

        count = 0
        for i in range(len(y_pred)):
            if np.array_equal(y_pred[i], y[i]):
                count += 1

        print("Acc:", count / len(y))
