#!/usr/bin/env python
# coding: utf-8
from typing import Optional
from tqdm import trange
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset


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

    def __iter__(self):
        for i in range(self.n_samples):
            yield self[i]

    def __len__(self):
        return self.n_samples


def count_classes(dataset, num_classes, disable=True):
    class_counts = torch.zeros(num_classes, dtype=torch.long)
    sample_with_class_count = torch.zeros(num_classes, dtype=torch.long)
    for _labels in tqdm(dataset, disable=disable):
        _ind, _counts = _labels['label'].unique(return_counts=True)
        class_counts[_ind] += _counts
        sample_with_class_count[_ind] += 1
    return class_counts, sample_with_class_count


def instance_class_matrix(dataset, num_classes, disable=True):
    class_counts = torch.zeros((len(dataset), num_classes), dtype=torch.long)
    for _scene_id, _label in enumerate(tqdm(dataset, disable=disable)):
        _ind, _counts = _label['label'].unique(return_counts=True)
        class_counts[_scene_id, _ind] += _counts
    return class_counts


def make_stratified_split_of_segmentation_dataset(
        dataset: Dataset,
        num_classes: int,
        split_ratio: Optional[float] = 0.2,
        names_of_classes: Optional[int] = None,
        verbose: bool = False,
        ignore_index: Optional[bool] = None,
        max_optimization_iterations: int = 1000000,
        split_n_sample_slack: int = 0,
):
    disable_tqdm = not verbose
    ds_cc, ds_swcc = count_classes(dataset, num_classes, disable=disable_tqdm)
    if names_of_classes is None:
        names_of_classes = [f"class_{_i}" for _i in range(num_classes)]
    dataset_stats = pd.DataFrame({
            'class_count': ds_cc,
            'samples_with_class_count': ds_swcc
        }, index=names_of_classes)
    if verbose:
        print(dataset_stats.sort_values('samples_with_class_count', ascending=False))

    icm = instance_class_matrix(dataset, num_classes, disable=disable_tqdm)
    # TODO: remove columns with ignore_index
    # icm = icm[:, 1:].numpy()

    optimization_weights_for_classes = np.zeros(icm.shape[1], dtype=np.float)
    # TODO: override weights (importance of classes)
    # for _idx, (_k, _s) in enumerate(dataset_stats.iterrows()):
    #     if _s.total_scenes_count > 20 and _k != 'Background':
    #         optimization_weights_for_classes[_idx - 1] = 1 / _s.total_voxel_count

    optimization_weights_for_classes /= optimization_weights_for_classes.sum()

    if verbose:
        print('\n'.join(f"{_f:1.9f}" for _f in optimization_weights_for_classes))

    num_samples = icm.shape[0]
    testset_size = int(np.floor(num_samples * split_ratio))
    total_class_voxels = icm.sum(axis=0)

    def calc_cost(subsample):
        subset_class_voxels = icm[subsample].sum(axis=0)
        per_class_ratios = subset_class_voxels / total_class_voxels.astype(np.float)
        return (optimization_weights_for_classes * np.abs(split_ratio - per_class_ratios)).sum()

    cost_stats = []
    best_cost = np.inf
    best_testset = None

    for _ in trange(max_optimization_iterations):
        subsample_size = np.random.randint(testset_size - split_n_sample_slack, testset_size + split_n_sample_slack)
        random_testset = np.random.permutation(num_samples)[:subsample_size]
        _cost = calc_cost(random_testset)
        if _cost < best_cost:
            best_cost = _cost
            best_testset = random_testset
        cost_stats.append(_cost)

    subset_class_stats = icm[best_testset].sum(axis=0)
    per_class_ratios = subset_class_stats / total_class_voxels.astype(np.float)
    residual = np.abs(split_ratio - per_class_ratios)
    # TODO: need to account for ignore_index
    # optimization_results = pd.DataFrame({
    #     'weights': optimization_weights_for_classes,
    #     'ratios': per_class_ratios
    # }, index=names_of_classes[1:])
    # TODO: plot histograms of splits
    # if verbose:
    #     pd.Series(cost_stats).plot(kind='hist')
    #     pd.Series(cost_stats).plot(kind='hist', bins=50)
    # icm[:, optimization_weights_for_classes == 0].sum(axis=1)
    # optimization_weights_for_classes == 0
    # removed_classes = np.where(optimization_weights_for_classes==0)[0] + 1
    # scenes_with_no_classes_but_removed = np.where(icm[:,optimization_weights_for_classes!=0].sum(axis=1)==0)[0]
    # for _scene_id in scenes_with_no_classes_but_removed:
    #     print(f"scene_id={_scene_id}: {labels[_scene_id]['semantic'].unique()}")
    return best_testset


def main():
    num_classes = 20
    dataset = MockSegmentationDataset(n_classes=num_classes)
    split = make_stratified_split_of_segmentation_dataset(dataset, num_classes)
    # TODO make test for output of function


if __name__ == '__main__':
    main()
