"""
Module: utils.py
Description:
    This module provides utility functions for the MODDA framework including:
      - Evaluation metric calculation (ROC, AUPR, Accuracy, F1, etc.)
      - Setting random seeds for reproducibility.
      - Early stopping mechanism during training.
      - Plotting ROC and Precision-Recall curves.
"""

import datetime
import numpy as np
import torch
import random
# import seaborn
import os
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt


def get_metrics_auc(real_score, predict_score):
    """
    Compute the Area Under the ROC Curve (AUC) and the Average Precision (AUPR).

    Parameters:
        real_score (array-like): True binary labels.
        predict_score (array-like): Predicted scores.

    Returns:
        tuple: (AUC, AUPR)
    """
    auc = roc_auc_score(real_score, predict_score)
    aupr = average_precision_score(real_score, predict_score)
    return auc, aupr


def get_metrics(real_score, predict_score):
    """
    Calculate various performance metrics including AUC, AUPR, Accuracy, F1-Score, Precision, Recall, and Specificity.
    
    The implementation is based on the method described in:
    Yu Z, Huang F, Zhao X et al. Predicting drug-disease associations through layer attention graph convolutional network,
    Brief Bioinform 2021;22.

    Parameters:
        real_score (array-like): True labels.
        predict_score (array-like): Predicted scores.

    Returns:
        tuple: (AUC, AUPR, Accuracy, F1-Score, Precision, Recall, Specificity)
    """
    # Obtain sorted unique predicted scores to generate thresholds
    sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.asmatrix(thresholds)
    thresholds_num = thresholds.shape[1]

    # Create a prediction matrix for all thresholds
    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1

    # Calculate TP, FP, FN, and TN for each threshold
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

    recall_list = tpr
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack((recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return auc[0, 0], aupr[0, 0], accuracy, f1_score, precision, recall, specificity


def set_seed(seed=0):
    """
    Set the random seed for Python, NumPy, and PyTorch for reproducibility.

    Parameters:
        seed (int): The random seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class EarlyStopping(object):
    """
    Early stopping utility to halt training when the validation performance stops improving.
    """

    def __init__(self, patience=10, saved_path="."):
        """
        Parameters:
            patience (int): Number of epochs with no improvement after which training is stopped.
            saved_path (str): Directory path to save the model checkpoint.
        """
        dt = datetime.datetime.now()
        self.filename = os.path.join(
            saved_path, "early_stop_{}_{}-{}-{}.pth".format(dt.date(), dt.hour, dt.minute, dt.second)
        )
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        """
        Check if early stopping condition is met based on current loss and accuracy.

        Parameters:
            loss (float): Current loss value.
            acc (float): Current accuracy value.
            model (torch.nn.Module): The model being trained.

        Returns:
            bool: True if early stopping condition is met, otherwise False.
        """
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            # Uncomment the following line for debugging
            # print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Save the current model state as a checkpoint."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the best model checkpoint."""
        model.load_state_dict(torch.load(self.filename))


def plot_result_auc(args, label, predict, auc):
    """
    Plot and save the Receiver Operating Characteristic (ROC) curve.

    Parameters:
        args: Argument object containing saved_path.
        label (array-like): True labels.
        predict (array-like): Predicted scores.
        auc (float): Computed AUC value.
    """
    # seaborn.set_style()
    fpr, tpr, _ = roc_curve(label, predict)
    plt.figure(figsize=(8, 8))
    lw = 2
    plt.plot(fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.4f)" % auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(args.saved_path, "result_auc.png"))
    plt.clf()


def plot_result_aupr(args, label, predict, aupr):
    """
    Plot and save the Precision-Recall (PR) curve.

    Parameters:
        args: Argument object containing saved_path.
        label (array-like): True labels.
        predict (array-like): Predicted scores.
        aupr (float): Computed AUPR value.
    """
    # seaborn.set_style()
    precision, recall, _ = precision_recall_curve(label, predict)
    plt.figure(figsize=(8, 8))
    lw = 2
    plt.plot(precision, recall, color="darkorange", lw=lw, label="AUPR (area = %0.4f)" % aupr)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(args.saved_path, "result_aupr.png"))
    plt.clf()

def get_metrics_with_threshold(real_score, predict_score):
    """
    Same as get_metrics(), but additionally returns the threshold that maximizes F1.

    Returns:
        auc, aupr, accuracy, f1_score, precision, recall, specificity, best_threshold
    """
    sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)

    thresholds = sorted_predict_score[np.int32(sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.asmatrix(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1

    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)

    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

    recall_list = tpr
    precision_list = TP / (TP + FP)

    PR_dot_matrix = np.mat(sorted(np.column_stack((recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)

    max_index = np.argmax(f1_score_list)

    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]

    best_threshold = float(np.asarray(thresholds).flatten()[max_index])

    return (
        auc[0, 0],
        aupr[0, 0],
        accuracy,
        f1_score,
        precision,
        recall,
        specificity,
        best_threshold,
    )