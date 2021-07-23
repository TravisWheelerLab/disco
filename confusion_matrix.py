import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools

class ConfusionMatrix:

    def __init__(self, num_classes=3):
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def increment_conf_matrix(self, target, pred):
        stacked = torch.stack((target, pred), dim=1).squeeze().numpy()
        for i in range(stacked.shape[0]):
            true_label = stacked[0][i]
            predicted_label = stacked[1][i]
            self.matrix[true_label, predicted_label] += 1

    def plot(self, classes, normalize=False, title='Confusion matrix', cmap='Blues'):
        if normalize:
            print(self.matrix)
            self.matrix = self.matrix.astype('float') / self.matrix.sum(axis=1)[:, np.newaxis]
            np.round(self.matrix, 3)
            print('Normalized confusion matrix.')
        else:
            print('Confusion matrix, without normalization.')

        print(self.matrix)
        plt.imshow(self.matrix, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = self.matrix.max() / 2.
        for i, j in itertools.product(range(self.matrix.shape[0]), range(self.matrix.shape[1])):
            plt.text(j, i, format(self.matrix[i, j], fmt), horizontalalignment="center",
                     color="white" if self.matrix[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('image_offload/conf_matrix.png')
        plt.close()
