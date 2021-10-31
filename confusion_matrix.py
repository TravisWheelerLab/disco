import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools


class ConfusionMatrix:

    def __init__(self, num_classes=3):
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.correct = 0
        self.total = 0
        self.doctored_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.doctored_correct = 0
        self.doctored_total = 0

    def increment(self, target, pred, device, is_a):
        cutoff_pct = 1/2
        # takes in target, the true labels, and pred, the predicted labels, for each time point in a given
        # spectrogram, and maps those labels to their correct places in the confusion matrix.
        if device == 'cuda':
            stacked = torch.stack((target, pred), dim=1).squeeze().cpu().numpy()
        else:
            stacked = torch.stack((target, pred), dim=1).squeeze().numpy()

        if is_a:
            count_stop_idx = round(target.shape[1] * cutoff_pct)
        else:
            count_stop_idx = stacked.shape[1]

        for i in range(stacked.shape[1]):
            true_label = stacked[0][i]
            predicted_label = stacked[1][i]
            self.matrix[true_label, predicted_label] += 1
            if i <= count_stop_idx:
                self.doctored_matrix[true_label, predicted_label] += 1

        self.correct += torch.sum(pred == target)
        self.total += torch.numel(target)
        self.doctored_correct += torch.sum(pred[:, :count_stop_idx] == target[:, :count_stop_idx])
        self.doctored_total += torch.numel(target[:, :count_stop_idx])

    def plot_matrices(self, classes, save_images, plot_undoctored, plot_doctored):

        if plot_undoctored:
            self.plot_indiv_matrix(classes, save_images, self.matrix)
        if plot_doctored:
            self.plot_indiv_matrix(classes, save_images, self.doctored_matrix)

    def plot_indiv_matrix(self, classes, save_images, matrix_to_plot):
        # this function generates and saves images of the three confusion matrices: counts and normalized across
        # rows/cols.

        # Creates three plots: totals, row-normalized (recall), and column-normalized (precision)
        print('UNNORMALIZED:\n', matrix_to_plot)
        row_normalized_mat = np.round(matrix_to_plot.astype('float') / matrix_to_plot.sum(axis=1)[:, np.newaxis], 3)
        print('ROW-NORMALIZED:\n', row_normalized_mat)
        col_normalized_mat = np.round(matrix_to_plot.astype('float') / matrix_to_plot.sum(axis=0)[np.newaxis, :], 3)
        print('COLUMN-NORMALIZED:\n', col_normalized_mat)

        if save_images:
            matrices_to_save = [matrix_to_plot, row_normalized_mat, col_normalized_mat]

            cmap = 'Blues' if matrix_to_plot is self.matrix else 'Purples'
            name_of_matrix = 'uncorrected' if matrix_to_plot is self.matrix else 'corrected'

            for matrix_idx in range(3):
                name = 'Counts'
                if matrix_idx == 1:
                    name = 'Recall'
                elif matrix_idx == 2:
                    name = 'Precision'

                plt.imshow(matrices_to_save[matrix_idx], interpolation='nearest', cmap=cmap)
                plt.title(name + ', ' + name_of_matrix.capitalize())
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

                plt.savefig('image_offload/conf_matrix_' + name + '_' + name_of_matrix + '.png')
                plt.close()

            print("matrices saved in /image_offload/ directory.")
