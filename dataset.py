import os
import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset
import random


def print_statistics(X, string):
    print(">" * 10 + string + ">" * 10)
    print("Average interactions", X.sum(1).mean(0).item())
    nonzero_row_indice, nonzero_col_indice = X.nonzero()
    unique_nonzero_row_indice = np.unique(nonzero_row_indice)
    unique_nonzero_col_indice = np.unique(nonzero_col_indice)
    print("Non-zero rows", len(unique_nonzero_row_indice) / X.shape[0])
    print("Non-zero columns", len(unique_nonzero_col_indice) / X.shape[1])
    print("Matrix density", len(nonzero_row_indice) / (X.shape[0] * X.shape[1]))


class BasicDataset(Dataset):
    """
    generate dataset from raw *.txt
    contains:
        tensors like (`user`, `bundle_p`, `bundle_n1`, `bundle_n2`, ...) for BPR (use `self.user_bundles`)
    Args:
    - `path`: the path of dir that contains dataset dir
    - `name`: the name of dataset (used as the name of dir)
    - `neg_sample`: the number of negative samples for each user-bundle_p pair
    - `seed`: seed of `np.random`
    """

    def __init__(self, path, name, task, neg_sample):
        self.path = path
        self.name = name
        self.task = task
        self.neg_sample = neg_sample
        self.num_users, self.num_bundles, self.num_items = self.__load_data_size()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __load_data_size(self):
        with open(
            os.path.join(self.path, self.name, "{}_data_size.txt".format(self.name)),
            "r",
        ) as f:
            return [int(s) for s in f.readline().split("\t")][:3]

    def load_U_B_interaction(self):
        with open(
            os.path.join(self.path, self.name, "user_bundle_{}.txt".format(self.task)),
            "r",
        ) as f:
            return list(
                map(lambda s: tuple(int(i) for i in s[:-1].split("\t")), f.readlines())
            )

    def load_U_I_interaction(self):
        with open(os.path.join(self.path, self.name, "user_item.txt"), "r") as f:
            return list(
                map(lambda s: tuple(int(i) for i in s[:-1].split("\t")), f.readlines())
            )

    def load_B_I_affiliation(self):
        with open(os.path.join(self.path, self.name, "bundle_item.txt"), "r") as f:
            return list(
                map(lambda s: tuple(int(i) for i in s[:-1].split("\t")), f.readlines())
            )


class BundleTrainDataset(BasicDataset):
    def __init__(self, path, name):
        super().__init__(path, name, "train", 1)
        # U-B
        self.U_B_pairs = self.load_U_B_interaction()
        indice = np.array(self.U_B_pairs, dtype=np.int32)
        values = np.ones(len(self.U_B_pairs), dtype=np.float32)
        self.ground_truth_u_b = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])),
            shape=(self.num_users, self.num_bundles),
        ).tocsr()
        print_statistics(self.ground_truth_u_b, "U-B statistics in train")

    def __getitem__(self, index):
        user_b, pos_bundle = self.U_B_pairs[index]
        all_bundles = [pos_bundle]
        while True:
            i = np.random.randint(self.num_bundles)
            if self.ground_truth_u_b[user_b, i] == 0 and not i in all_bundles:
                all_bundles.append(i)
                if len(all_bundles) == self.neg_sample + 1:
                    break
        return torch.LongTensor([user_b]), torch.LongTensor(all_bundles)

    def __len__(self):
        return len(self.U_B_pairs)


class BundleTestDataset(BasicDataset):
    def __init__(self, path, name, train_dataset, task="test"):
        super().__init__(path, name, task, None)
        # U-B
        self.U_B_pairs = self.load_U_B_interaction()
        indice = np.array(self.U_B_pairs, dtype=np.int32)
        values = np.ones(len(self.U_B_pairs), dtype=np.float32)
        self.ground_truth_u_b = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])),
            shape=(self.num_users, self.num_bundles),
        ).tocsr()

        self.train_mask_u_b = train_dataset.ground_truth_u_b
        print_statistics(self.ground_truth_u_b, "U-B statistics in test")
        self.users = torch.arange(self.num_users, dtype=torch.long).unsqueeze(dim=1)
        self.bundles = torch.arange(self.num_bundles, dtype=torch.long)
        assert self.train_mask_u_b.shape == self.ground_truth_u_b.shape

    def __getitem__(self, index):
        return (
            index,
            torch.from_numpy(self.ground_truth_u_b[index].toarray()).squeeze(),
            torch.from_numpy(self.train_mask_u_b[index].toarray()).squeeze(),
        )

    def __len__(self):
        return self.ground_truth_u_b.shape[0]


class ItemDataset(BasicDataset):
    def __init__(self, path, name):
        super().__init__(path, name, None, None)
        # U-I
        self.U_I_pairs = self.load_U_I_interaction()
        indice = np.array(self.U_I_pairs, dtype=np.int32)
        values = np.ones(len(self.U_I_pairs), dtype=np.float32)
        self.ground_truth_u_i = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])),
            shape=(self.num_users, self.num_items),
        ).tocsr()

        print_statistics(self.ground_truth_u_i, "U-I statistics")


class AssistDataset(BasicDataset):
    def __init__(self, path, name):
        super().__init__(path, name, None, None)
        # B-I
        self.B_I_pairs = self.load_B_I_affiliation()
        indice = np.array(self.B_I_pairs, dtype=np.int32)
        values = np.ones(len(self.B_I_pairs), dtype=np.float32)
        self.ground_truth_b_i = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])),
            shape=(self.num_bundles, self.num_items),
        ).tocsr()

        print_statistics(self.ground_truth_b_i, "B-I statistics")


def get_dataset(name, path="./data/"):
    assist_data = AssistDataset(path, name)
    print("finish loading assist data")
    item_data = ItemDataset(path, name)
    print("finish loading item data")

    bundle_train_data = BundleTrainDataset(path, name)
    print("finish loading bundle train data")
    bundle_test_data = BundleTestDataset(path, name, bundle_train_data)
    print("finish loading bundle test data")

    return bundle_train_data, bundle_test_data, item_data, assist_data

