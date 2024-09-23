import numpy as np
import pickle
import random
import sys

np.set_printoptions(threshold=sys.maxsize)


class DatasetESC50:
    def __init__(self, data, val, test, fold):
        # Get the train class numbers
        train = list(range(50))
        train = [t for t in train if t not in val and t not in test]
        train.sort()
        val.sort()
        test.sort()

        # Get the pickled data
        with open(data, "rb") as f:
            data = pickle.load(f)

        all_labels = np.array(data["labels"])
        all_features = np.array([list(d.to("cpu")[0]) for d in data["features"]])
        all_auxiliary = np.array(data["auxiliary"])

        train_labels = []
        train_features = []
        train_auxiliary = []

        val_labels = []
        val_features = []
        val_auxiliary = []

        for i in range(len(all_labels)):
            if (fold == "test"):
                if all_labels[i] in test:
                    val_labels.append(all_labels[i])
                    val_features.append(all_features[i])
                    val_auxiliary.append(all_auxiliary[i])
                else:
                    train_labels.append(all_labels[i])
                    train_features.append(all_features[i])
                    train_auxiliary.append(all_auxiliary[i])

            if all_labels[i] in test:
                continue
            elif all_labels[i] in val:
                val_labels.append(all_labels[i])
                val_features.append(all_features[i])
                val_auxiliary.append(all_auxiliary[i])
            else:
                train_labels.append(all_labels[i])
                train_features.append(all_features[i])
                train_auxiliary.append(all_auxiliary[i])

        seen_labels = train_labels
        seen_aux = train_auxiliary
        seen_features = train_features
        unseen_features = val_features
        unseen_aux = val_auxiliary

        # A list of all indices, to be shuffled to make a test/train split for seen data
        indices = [i for i in range(len(seen_labels))]
        random.shuffle(indices)
        test_num = int(len(indices) * 0.2)  # 20-80 split on test-train

        # Partition the seen data into test and train
        seen_test_labels = [seen_labels[i] for i in indices[:test_num]]
        seen_test_features = [seen_features[i] for i in indices[:test_num]]

        seen_train_labels = [seen_labels[i] for i in indices[test_num:]]
        seen_train_features = [seen_features[i] for i in indices[test_num:]]

        self.x_s_train = np.array(seen_train_features)
        self.y_s_train = np.array(seen_train_labels)

        self.x_s_test = np.array(seen_test_features)
        self.y_s_test = np.array(seen_test_labels)

        self.x_u_test = np.array(unseen_features)
        self.y_u_test = np.array(val_labels)

        self.d_ft = self.x_s_train.shape[1]

        self.s_class = np.array(train)
        self.u_class = np.array(val)

        self.s_attr = np.array([seen_aux[seen_labels.index(t)] for t in train])
        self.u_attr = np.array([unseen_aux[val_labels.index(v)] for v in val])

        self.attr = list(self.u_attr)
        for aux in list(self.s_attr):
            self.attr.append(aux)
        self.attr = np.array(self.attr)

        self.d_attr = self.attr.shape[1]


def index_labels(labels, classes, check=True):
    """
    Indexes labels in classes.

    Arg:
    labels:  [batch_size]
    classes: [n_class]
    index_labels(dset.y_s_train, dset.s_class)
    """
    print(len(labels), len(classes))
    indexed_labels = np.searchsorted(classes, labels)
    if check:
        assert np.all(np.equal(classes[indexed_labels], labels))

    return indexed_labels
