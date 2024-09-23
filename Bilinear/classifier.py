# -*- coding: utf-8 -*-
# --------------------------------------------------
#
# classifier.py
#
# - Attribute Label Embedding (ALE) compatibility function
# - Normalized Zero-Shot evaluation
#
# Written by cetinsamet -*- cetin.samet@metu.edu.tr
# December, 2019
# --------------------------------------------------
import torch
import torch.nn as nn


def init_layer(layer):
    torch.nn.init.xavier_normal_(layer.weight, gain=0.1)

    if layer.bias is not None:
        layer.bias.data.fill_(0.0)


class Compatibility(nn.Module):
    def __init__(self, d_in, d_out):
        super(Compatibility, self).__init__()

        hidden_units = d_in
        self.fc1 = nn.Linear(d_in, hidden_units, bias=False)
        self.fc2 = nn.Linear(hidden_units, d_out, bias=False)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc1)
        init_layer(self.fc2)

    def forward(self, x1, x2):
        """
        :param x1: (num_samples, num_segments, d_in)
        :param x2: (num_classes, d_out)
        :return: (num_samples, num_classes)
        """

        x1 = torch.tanh(self.fc1(x1))
        x1 = self.fc2(x1)  # (num_samples, d_out)

        x2 = x2.transpose(0, 1)  # (d_out, num_classes)

        output = x1.matmul(x2)  # (num_samples, num_classes)

        return output


def evaluate(model, x, y, attrs):
    """Normalized Zero-Shot Evaluation"""

    classes = torch.unique(y)
    n_class = len(classes)
    t_acc = 0.0
    y_ = torch.argmax(model(x, attrs), dim=1)

    for _class in classes:
        idx_sample = [i for i, _y in enumerate(y) if _y == _class]
        n_sample = len(idx_sample)

        y_sample_ = y_[idx_sample]
        y_sample = y[idx_sample].long()

        scr_sample = torch.sum(y_sample_ == y_sample).item()
        acc_sample = scr_sample / n_sample
        t_acc += acc_sample

    acc = t_acc / n_class
    return acc
