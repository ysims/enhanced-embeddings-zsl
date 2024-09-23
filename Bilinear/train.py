# -*- coding: utf-8 -*-
# --------------------------------------------------
#
# train.py
#
# Performs zero-shot training
#
# Written by cetinsamet -*- cetin.samet@metu.edu.tr
# December, 2019
# --------------------------------------------------
import numpy as np
import argparse
import random
import torch
import math
import os
import statistics

from torch.utils.data import TensorDataset, DataLoader

from classifier import Compatibility, evaluate
from data_loader import DatasetESC50, index_labels

FN = torch.from_numpy


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--fold", type=str)
parser.add_argument("--mode", type=str)
parser.add_argument("--optim_type", type=str)
parser.add_argument("--lr", type=float)
parser.add_argument("--wd", type=float)
parser.add_argument("--lr_decay", type=float)
parser.add_argument("--n_epoch", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--folder", type=str)
parser.add_argument("--device", type=str)
args = parser.parse_args()

# These are for testing with fold 4, and doing 4-fold cross-validation on the remaining folds

if args.dataset == "ESC-50":
    test_classes = [43, 5, 37, 12, 9, 0, 11, 8, 15, 16]
    val_classes = [43, 5, 37, 12, 9, 0, 11, 8, 15, 16]
    
    if args.fold == "fold04":
        val_classes = [27, 46, 38, 3, 29, 48, 40, 31, 2, 35]
    elif args.fold == "fold14":
        val_classes = [22, 13, 39, 49, 32, 26, 42, 21, 19, 36]
    elif args.fold == "fold24":
        val_classes = [23, 41, 14, 24, 33, 30, 4, 17, 10, 45]
    elif args.fold == "fold34":
        val_classes = [47, 34, 20, 44, 25, 6, 7, 1, 28, 18]
elif args.dataset == "FSC22":
    print("FSC22")
    #  train_classes = [0, 1, 2, 3, 4, 10, 11, 14, 16, 19, 20, 24, 25]
    val_classes = [6, 8, 9, 12, 13, 18, 22]
    test_classes = [5, 7, 15, 17, 21, 23, 26]

device_type = args.device
device = torch.device(device_type)

if device_type == "cpu":  # CUDA IS NOT AVAILABLE
    import psutil

    n_cpu = psutil.cpu_count()
    n_cpu_to_use = n_cpu // 4
    torch.set_num_threads(n_cpu_to_use)
    os.environ["MKL_NUM_THREADS"] = str(n_cpu_to_use)
    os.environ["KMP_AFFINITY"] = "compact"

if args.mode == "test":
    verbose = True
else:
    verbose = False

if verbose:
    print(
        "%s dataset running on %s mode with %s device"
        % (args.dataset.upper(), args.mode.upper(), device_type.upper())
    )

if args.dataset == "ESC-50":
    dset = DatasetESC50(args.folder, val_classes, test_classes, args.fold)

x_s_train = FN(dset.x_s_train).float().to(device)
y_s_train = FN(dset.y_s_train).to(device)
y_s_train_ix = FN(index_labels(dset.y_s_train, dset.s_class)).to(device)

x_s_test = FN(dset.x_s_test).float().to(device)
y_s_test = FN(dset.y_s_test).to(device)

x_u_test = FN(dset.x_u_test).float().to(device)
y_u_test = FN(dset.y_u_test).to(device)
y_u_test_ix = FN(index_labels(dset.y_u_test, dset.u_class)).to(device)

attr = FN(dset.attr).float().to(device)
s_attr = FN(dset.s_attr).float().to(device)
u_attr = FN(dset.u_attr).float().to(device)

n_s_train = len(x_s_train)

n_class = len(attr)
n_s_class = len(s_attr)
n_u_class = len(u_attr)

if verbose:
    print("Seen train 	:", x_s_train.size())
    print("Seen test 	:", x_s_test.size())
    print("Unseen test 	:", x_u_test.size())
    print("Attrs 		:", attr.size())
    print("Seen Attrs 	:", s_attr.size())
    print("Unseen Attrs	:", u_attr.size())

# seeds = [123]
seeds = [
    random.randrange(0, 9999999) for _ in range(0, 10)
]  # <- Train several times randomly
n_trials = len(seeds)
print("Running {} trials.".format(n_trials))

accs = np.zeros([n_trials, args.n_epoch, 4], "float32")

best_accs = []
for trial, seed in enumerate(seeds):
    best_acc = 0.0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # init classifier
    clf = Compatibility(d_in=dset.d_ft, d_out=dset.d_attr).to(device)

    # init loss
    # ce_loss = WARPLoss(device)
    # ce_loss = WARPLoss2(device)
    # ce_loss = objective
    ce_loss = torch.nn.CrossEntropyLoss()

    # init optimizer
    if args.optim_type == "adam":
        optimizer = torch.optim.Adam(
            params=clf.parameters(), lr=args.lr, weight_decay=args.wd
        )
    elif args.optim_type == "sgd":
        optimizer = torch.optim.SGD(
            params=clf.parameters(), lr=args.lr, weight_decay=args.wd
        )
    else:
        raise NotImplementedError

    # init scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay) # <- lr_scheduler

    data = TensorDataset(x_s_train, y_s_train_ix)
    data_loader = DataLoader(
        data, batch_size=args.batch_size, shuffle=True, drop_last=False
    )

    for epoch_idx in range(args.n_epoch):
        clf.train()  # Classifer train mode: ON

        running_loss = 0.0

        for x, y in data_loader:  # (x, y) <-> (image feature, image label)
            y_ = clf(x, s_attr)  # <- forward pass
            batch_loss = ce_loss(
                y_, y
            )  # + torch.linalg.norm(clf.fc1.weight) + torch.linalg.norm(clf.fc2.weight)  # <- calculate loss

            optimizer.zero_grad()  # <- set gradients to zero
            batch_loss.backward()  # <- calculate gradients
            optimizer.step()  # <- update weights

            running_loss += batch_loss.item() * args.batch_size  # <- cumulative loss

        # scheduler.step() # <- update schedular

        epoch_loss = running_loss / n_s_train  # <- calculate epoch loss

        print("Epoch %4d\tLoss : %s" % (epoch_idx + 1, epoch_loss))

        if math.isnan(epoch_loss):
            continue  # if loss is NAN, skip!

        if (epoch_idx + 1) % 1 == 0:
            clf.eval()  # Classifier evaluation mode: ON

            # ----------------------------------------------------------------------------------------------- #
            # ZERO-SHOT ACCURACY
            acc_zsl = evaluate(model=clf, x=x_u_test, y=y_u_test_ix, attrs=u_attr)
            # ------------------------------------------------------- #
            # * ----- * ----- * ----- * ----- * ----- * ----- * ----- *
            # ------------------------------------------------------- #
            # GENERALIZED SEEN ACCURACY
            acc_g_seen = evaluate(model=clf, x=x_s_test, y=y_s_test, attrs=attr)
            # ------------------------------------------------------- #
            # * ----- * ----- * ----- * ----- * ----- * ----- * ----- *
            # ------------------------------------------------------- #
            # GENERALIZED UNSEEN ACCURACY
            acc_g_unseen = evaluate(model=clf, x=x_u_test, y=y_u_test, attrs=attr)
            # ------------------------------------------------------- #
            # * ----- * ----- * ----- * ----- * ----- * ----- * ----- *
            # ------------------------------------------------------- #
            # GENERALIZED ZERO-SHOT ACCURACY
            if acc_g_seen + acc_g_unseen == 0.0:  # avoid divide by zero error!
                h_score = 0.0
            else:
                h_score = (2 * acc_g_seen * acc_g_unseen) / (acc_g_seen + acc_g_unseen)
            # ----------------------------------------------------------------------------------------------- #

            accs[trial, epoch_idx, :] = (
                acc_zsl,
                acc_g_seen,
                acc_g_unseen,
                h_score,
            )  # <- save accuracy values

            if verbose:
                print("Zero-Shot acc            : %f" % acc_zsl)
                print("Generalized Seen acc     : %f" % acc_g_seen)
                print("Generalized Unseen acc   : %f" % acc_g_unseen)
                print("H-Score                  : %f" % h_score)

            if best_acc < acc_zsl:
                best_acc = acc_zsl

    best_accs.append(best_acc)
    print("Best for run {} is {}.".format(trial, best_acc))

mean_best = statistics.mean(best_accs)
stdev_best = statistics.stdev(best_accs)

print("Mean best acc is", mean_best)
print("Standard deviation best acc is", stdev_best)
print("Best of the best is {}".format(max(best_accs)))


zsl_mean = accs[:, :, 0].mean(axis=0)
zsl_std = accs[:, :, 0].std(axis=0)
gzsls_mean = accs[:, :, 1].mean(axis=0)
gzsls_std = accs[:, :, 1].std(axis=0)
gzslu_mean = accs[:, :, 2].mean(axis=0)
gzslu_std = accs[:, :, 2].std(axis=0)
gzslh_mean = accs[:, :, 3].mean(axis=0)
gzslh_std = accs[:, :, 3].std(axis=0)

print("Zsl 	:: average: {mean:} +- {std:}".format(mean=zsl_mean[-1], std=zsl_std[-1]))
print(
    "Gzsls 	:: average: {mean:} +- {std:}".format(
        mean=gzsls_mean[-1], std=gzsls_std[-1]
    )
)
print(
    "Gzslu 	:: average: {mean:} +- {std:}".format(
        mean=gzslu_mean[-1], std=gzslu_std[-1]
    )
)
print(
    "Gzslh 	:: average: {mean:} +- {std:}".format(
        mean=gzslh_mean[-1], std=gzslh_std[-1]
    )
)

print(
    "Experiment was type {} : epoch {} : folder {}".format(
        args.fold, args.n_epoch, args.folder
    )
)
