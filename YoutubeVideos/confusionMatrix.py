__author__ = 'GongLi'

import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix

class ConfMatrix:

    def __init__(self, matrix=None, labels=None):
        self.m = None
        if matrix!=None:
            self.m = matrix
        if labels:
            self.labels = labels

    @staticmethod
    def load_matrix(self, fname):
        f = open(fname, "r")
        self.m = pickle.load(f)
        self.labels = pickle.load(f)
        f.close()

    def save_matrix(self, fname, labels=None):
        if '.' not in fname:
            fname+=".pickle"
        f = open(fname, "w")
        pickle.dump(self.m , f)
        pickle.dump(self.labels , f)
        print("saving %s" %fname)
        f.close()

    def get_classification(self, labels=None,
                           rm_labels=['tbd', '?','TBD']):
        labels = labels or self.labels
        idxs = [idx for idx, el in enumerate(labels) if el not in rm_labels]
        rm_idxs = [idx for idx, el in enumerate(labels) if el in rm_labels]
        if len(rm_idxs):
            print("don't consider %s" %(','.join(["%s->%s" %(label, idx) for label, idx in zip(rm_labels, rm_idxs)])))
        total = self.m[idxs,:].sum()
        target = 0.0
        for i in idxs:
            target+=self.m[i][i]
        print("global precision: %i/%i=%2.2f" %(target, total, target/total))
        return target/total

    def to_precision(self):
        precision = np.array(self.m, dtype=float)
        precision/=self.m.sum(axis=0)
        precision*=100
        return precision

    def to_recall(self):
        recall = np.array(self.m, dtype=float)
        recall/=self.m.sum(axis=1)[:,np.newaxis]
        recall*=100
        return recall

    def _gen_conf_matrix(self, fname, labels=None, title=None, threshold=.1, factor=1, normalize=""):
        plt.ioff()

        fname = fname or "conf_matrix"
        labels = labels or self.labels
        matrix = self.m
        if normalize!='':
            fname+= "_%s" %normalize
        title = title or fname
        #title += " %s" %normalize
        fig = plt.figure()
        plt.clf()
        ax = fig.add_subplot(111)
        ax.set_aspect(2)
        matrix = np.array(matrix, dtype=float)
        res = ax.imshow(matrix, cmap='Greys',
                        interpolation='nearest')

        size = matrix.shape[0]

        if normalize=='recall':
            matrix=self.to_recall()

        elif normalize=='precision':
            matrix=self.to_precision()

        for x in xrange(size):
            for y in xrange(size):
                if abs(matrix[x][y])>threshold:
                    value = "%2.0f" %(matrix[x][y]*factor)
                    ax.annotate(value, xy=(y, x),
                                horizontalalignment='center',
                                verticalalignment='center', color='green')

        cb = fig.colorbar(res)

        plt.xticks(range(size), labels, rotation='vertical')
        plt.yticks(range(size), labels)
        plt.title(title)
        plt.xlabel('Predicted class')
        plt.ylabel('Actual class')
        fname+=".png"
        fname = fname.replace(' ','')
        plt.savefig("/Users/GongLi/PycharmProjects/VisualEventRecognitionDemo/YoutubeVideos/static/confusionImages/"+fname, format='png')
        plt.close()
        print("generated : %s" %fname)

    def gen_conf_matrix(self, fname, labels=None, title=None, threshold=.1, factor=1):
        fname = fname or "conf_matrix"
        labels = labels or self.labels
        self._gen_conf_matrix(fname, labels, title=title, threshold=threshold, factor=factor)


if __name__ == "__main__":

    labels = ["birthday", "parade", "picnic", "show", "sports", "wedding"]
    t = ["birthday", "parade", "picnic", "show", "sports", "wedding"]
    p = ["birthday", "parade", "picnic", "show", "sports", "wedding"]

    cm = ConfMatrix(confusion_matrix(t, p), labels)
    cm.gen_conf_matrix('conf_matrix')