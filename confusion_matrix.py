import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools


class ConfusionMatrix:

    def __init__(self, num_classes=3):
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.correct_labels = 0
        self.total = 0

    def increment(self, target, pred, device, is_a=False):

        # takes in target, the true labels, and pred, the predicted labels, for each time point in a given
        # spectrogram, and maps those labels to their correct places in the confusion matrix.
        if device == 'cuda':
            stacked = torch.stack((target, pred), dim=1).squeeze().cpu().numpy()
        else:
            stacked = torch.stack((target, pred), dim=1).squeeze().numpy()

        if is_a:
            count_stop_idx = round(target.shape[1]*1/2)

        for i in range(stacked.shape[1]):
            if is_a and i >= count_stop_idx:
                break
            true_label = stacked[0][i]
            predicted_label = stacked[1][i]
            self.matrix[true_label, predicted_label] += 1

            self.correct_labels += torch.sum(pred == target)
            self.total += torch.numel(target)

    def plot(self, classes, save_images):
        # this function generates and saves images of the three confusion matrices: counts and normalized across
        # rows/cols.

        # Creates three plots: totals, row-normalized (recall), and column-normalized (precision)
        print('UNNORMALIZED:\n', self.matrix)
        row_normalized_mat = np.round(self.matrix.astype('float') / self.matrix.sum(axis=1)[:, np.newaxis], 3)
        print('ROW-NORMALIZED:\n', row_normalized_mat)
        col_normalized_mat = np.round(self.matrix.astype('float') / self.matrix.sum(axis=0)[np.newaxis, :], 3)
        print('COLUMN-NORMALIZED:\n', col_normalized_mat)

        if save_images:
            matrices_to_save = [self.matrix, row_normalized_mat, col_normalized_mat]

            for matrix_idx in range(3):
                name = 'Counts'
                if matrix_idx == 1:
                    name = 'Recall'
                elif matrix_idx == 2:
                    name = 'Precision'

                plt.imshow(matrices_to_save[matrix_idx], interpolation='nearest', cmap='Blues')
                plt.title(name)
                tick_marks = np.arange(len(classes))
                plt.xticks(tick_marks, classes)
                plt.yticks(tick_marks, classes)

                fmt = 'd' if matrix_idx == 0 else '.3f'
                thresh = matrices_to_save[matrix_idx].max() / 2.
                for i, j in itertools.product(range(matrices_to_save[matrix_idx].shape[0]),
                                              range(matrices_to_save[matrix_idx].shape[1])):
                    plt.text(j, i, format(matrices_to_save[matrix_idx][i, j], fmt), horizontalalignment="center",
                             color="white" if matrices_to_save[matrix_idx][i, j] > thresh else "black")

                plt.tight_layout()
                plt.ylabel('True label')
                plt.xlabel('Predicted label')

                plt.savefig('image_offload/conf_matrix_' + name + '.png')
                plt.close()

            print("matrices saved in /image_offload/ directory.")
