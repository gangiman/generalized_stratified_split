#!/usr/bin/env python
# coding: utf-8
from typing import Optional, Tuple, Dict, Callable, List, Union, Sequence
from tqdm import trange
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from operator import itemgetter
from torch.utils.data import Dataset


def compose(f, g):
    return lambda x: f(g(x))


def identity(x):
    return x


def _calc_label_fn(sample: Union[Tuple, Dict, torch.Tensor, np.ndarray]) -> Callable:
    if isinstance(sample, tuple):
        sample = sample[-1]
        deconstruct = itemgetter(-1)
    elif isinstance(sample, dict):
        label_key = None
        for _key in sample:
            if 'label' in _key.lower():
                label_key = _key
                break
        if label_key is None:
            raise AssertionError("Sample of dataset is a dict. Label key of sample is ambiguous.")
        else:
            sample = sample[label_key]
            deconstruct = itemgetter(label_key)
    else:
        deconstruct = identity

    if isinstance(sample, torch.Tensor):
        def label_fn(_sample):
            return _sample.cpu().detach().numpy()
    elif isinstance(sample, np.ndarray):
        label_fn = identity
    else:
        raise AssertionError("Unknown type of samples in dataset.")
    return compose(label_fn, deconstruct)


def instance_class_matrix(dataset: Sequence[np.ndarray], num_classes: int, disable: bool = True) -> np.ndarray:
    num_samples = len(dataset)
    inst_x_class_counts = np.zeros((num_samples, num_classes), dtype=np.long)
    for _sample_id in trange(num_samples, disable=disable):
        _sample = dataset[_sample_id]
        _ind, _counts = np.unique(_sample, return_counts=True)
        inst_x_class_counts[_sample_id, _ind] += _counts
    return inst_x_class_counts


def make_stratified_split_of_segmentation_dataset(
        dataset: Union[Dataset, np.ndarray, List],
        num_classes: int,
        split_ratio: Optional[float] = 0.2,
        names_of_classes: Optional[int] = None,
        verbose: bool = False,
        ignore_index: Optional[bool] = None,
        max_optimization_iterations: int = 1000000,
        split_n_sample_slack: int = 0,
):
    """
    Finds some split of samples that tries to balance class proportions
    :param dataset:
    :param num_classes:
    :param split_ratio:
    :param names_of_classes:
    :param verbose:
    :param ignore_index:
    :param max_optimization_iterations:
    :param split_n_sample_slack:
    :return:
    """
    disable_tqdm = not verbose
    if isinstance(dataset, Dataset):
        label_fn = _calc_label_fn(dataset[0])
        dataset = [label_fn(dataset[_i]) for _i in trange(len(dataset), disable=disable_tqdm)]
    icm = instance_class_matrix(dataset, num_classes, disable=disable_tqdm)
    # TODO: remove columns with ignore_index
    # icm = icm[:, 1:].numpy()
    ds_cc = icm.sum(axis=0)
    ds_swcc = (icm > 0).astype(np.long).sum(axis=0)
    if names_of_classes is None:
        names_of_classes = [f"class_{_i}" for _i in range(num_classes)]
    dataset_stats = pd.DataFrame({
        'class_count': ds_cc,
        'samples_with_class_count': ds_swcc
    }, index=names_of_classes)
    if verbose:
        print(dataset_stats.sort_values('samples_with_class_count', ascending=False))
    optimization_weights_for_classes = np.zeros(icm.shape[1], dtype=np.float)
    # TODO: override weights (importance of classes)
    optimization_weights_for_classes = 1.0 / ds_cc
    optimization_weights_for_classes[ds_cc == 0] = 0
    optimization_weights_for_classes /= optimization_weights_for_classes.sum()
    if verbose:
        print('\n'.join(f"{_f:1.9f}" for _f in optimization_weights_for_classes))
    num_samples = icm.shape[0]
    testset_size = int(np.floor(num_samples * split_ratio))

    def calc_cost(subsample):
        subset_class_voxels = icm[subsample].sum(axis=0)
        per_class_ratios = subset_class_voxels / ds_cc.astype(np.float)
        return (optimization_weights_for_classes * np.abs(split_ratio - per_class_ratios)).sum()

    cost_stats = []
    best_cost = np.inf
    best_testset = None
    for _ in trange(max_optimization_iterations):
        if split_n_sample_slack:
            subsample_size = np.random.randint(testset_size - split_n_sample_slack, testset_size + split_n_sample_slack)
        else:
            subsample_size = testset_size
        random_testset = np.random.permutation(num_samples)[:subsample_size]
        _cost = calc_cost(random_testset)
        if _cost < best_cost:
            best_cost = _cost
            best_testset = random_testset
        cost_stats.append(_cost)

    subset_class_stats = icm[best_testset].sum(axis=0)
    per_class_ratios = subset_class_stats / ds_cc.astype(np.float)
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
