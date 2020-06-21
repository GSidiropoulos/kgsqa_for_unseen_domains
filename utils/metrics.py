from functools import reduce

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support


class Metrics(object):

    @staticmethod
    def report(y_true, y_pred, labels, target_names, verbose=True):

        clf = classification_report(y_true=y_true, y_pred=y_pred, labels=labels, target_names=target_names)

        if verbose:
            print(clf)

        return clf

    @staticmethod
    def pr_rc_fscore_sup(y_true, y_pred, average="macro", verbose=True):

        precision, recall, fscore, support = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred,
                                                                             average=average)

        if verbose:
            print(average + ": ", precision, recall, fscore, support)

        return precision, recall, fscore, support

    def report_average(*args):
        """"https://datascience.stackexchange.com/questions/31134/python-sklearn-average-classification-reports """
        report_list = list()
        for report in args:
            splited = [" ".join(x.split()) for x in report.split("\n\n")]
            header = [x for x in splited[0].split(" ")]
            data = np.array(splited[1].split(" ")).reshape(-1, len(header) + 1)
            data = np.delete(data, 0, 1).astype(float)
            avg_total = np.array([x for x in splited[2].split(" ")][3:]).astype(float).reshape(-1, len(header))
            df = pd.DataFrame(np.concatenate((data, avg_total)), columns=header)
            report_list.append(df)
        res = reduce(lambda x, y: x.add(y, fill_value=0), report_list) / len(report_list)
        return res.rename(index={res.index[-1]: "avg / total"}), data
