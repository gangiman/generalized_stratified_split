#!/usr/bin/env python
# coding: utf-8
from typing import Optional
from tqdm import trange
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from stratified_split_for_segmentation import make_stratified_split_of_segmentation_dataset


class MockSegmentationDataset(Dataset):
    def __init__(self, n_samples=100, n_classes=20,
                 class_distribution='exp', sample_shape=(100,)):
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.class_distribution = class_distribution
        self.sample_shape = sample_shape
        if class_distribution == 'exp':
            p = np.exp(-np.arange(self.n_classes))
            p /= p.sum()
        else:
            p = np.ones(self.n_classes)/self.n_classes
        self.p = p

    def __getitem__(self, item):
        label = np.random.choice(
            np.arange(self.n_classes, dtype=np.int),
            size=self.sample_shape,
            p=self.p)
        return {
            'input': None,
            'label': label
        }

    def __len__(self):
        return self.n_samples


def test_split():
    num_classes = 20
    dataset = MockSegmentationDataset(n_classes=num_classes)
    split = make_stratified_split_of_segmentation_dataset(dataset, num_classes,
                                                          max_optimization_iterations=100,
                                                          verbose=True)
    # TODO make test for output of function


def main():
    test_split()


if __name__ == '__main__':
    main()
