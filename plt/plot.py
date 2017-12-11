# -*- coding: utf-8 -*-

"""Main module."""

from odo.numpy_dtype import dshape_to_pandas
from odo import convert, Temp, CSV
from odo.utils import keywords

import pandas as pd
import datashape
from datashape.predicates import isrecord
from toolz import keyfilter

from functools import wraps
import warnings
import numpy as np
import re


# def _csv_to_dataframe(c, dshape=None, chunksize=None, **kwargs):
#     header = {False: None, True: 0}.get(
#         kwargs.pop('has_header', c.has_header), 'infer')
#
#     sep = kwargs.pop(
#         'sep', kwargs.pop('delimiter', c.dialect.get('delimiter', ',')))
#     encoding = kwargs.pop('encoding', c.encoding)
#
#     if dshape:
#         dtypes, parse_dates = dshape_to_pandas(dshape)
#         if isrecord(dshape.measure):
#             names = kwargs.get('names', dshape.measure.names)
#         else:
#             names = kwargs.get('names')
#     else:
#         dtypes = parse_dates = names = None
#
#     usecols = kwargs.pop('usecols', None)
#     if parse_dates and usecols:
#         parse_dates = [col for col in parse_dates if col in usecols]
#
#     # See read_csv docs for header for reasoning
#     if names:
#         try:
#             with c.open() as f:
#                 found_names = pd.read_csv(f,
#                                           nrows=1,
#                                           encoding=encoding,
#                                           sep=sep)
#         except StopIteration:
#             with c.open() as f:
#                 found_names = pd.read_csv(f, encoding=encoding, sep=sep)
#     if names and header == 'infer':
#         if [n.strip() for n in found_names] == [n.strip() for n in names]:
#             header = 0
#         elif (all(re.match('^\s*\D\w*\s*$', n) for n in found_names) and
#                 not all(dt == datashape.string for dt in dshape.measure.types)):
#             header = 0
#         else:
#             header = None
#
#     kwargs = keyfilter(keywords(pd.read_csv).__contains__, kwargs)
#     with c.open() as f:
#         return pd.read_csv(f,
#                            header=header,
#                            sep=sep,
#                            encoding=encoding,
#                            dtype=dtypes,
#                            parse_dates=parse_dates,
#                            names=names,
#                            chunksize=chunksize,
#                            usecols=usecols,
#                            **kwargs)
#
#
# @convert.register(pd.DataFrame, (Temp(CSV), CSV), cost=10.0)
# def csv_to_dataframe(c, dshape=None, chunksize=None, nrows=None, **kwargs):
#     try:
#         return _csv_to_dataframe(c, dshape=dshape, chunksize=chunksize,
#                                  nrows=nrows, **kwargs)
#     except StopIteration:
#         if nrows:
#             return _csv_to_dataframe(c, dshape=dshape, chunksize=chunksize,
#                                      **kwargs)
#         else:
#             raise
#
#


def _binary_clf_curve(y_true, y_score, pos_label=None):
    if y_true.shape[0] != y_score.shape[0]:
        raise Exception('Sizes do not match ytrue:{} != y_scor:{}'.format(y_true.shape, y_score.shape))

    # ensure binary classification if pos_label is not specified
    classes = np.unique(y_true)
    if (pos_label is None and
        not (np.array_equal(classes, [0, 1]) or
             np.array_equal(classes, [-1, 1]) or
             np.array_equal(classes, [0]) or
             np.array_equal(classes, [-1]) or
             np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    return fps, tps, y_score[threshold_idxs]


def precision_recall_curve(y_true, y_preds, pos_label=None):
    fps, tps, thresholds = _binary_clf_curve(y_true,
                                             y_preds,
                                             pos_label=pos_label)

    precision = tps / (tps + fps)
    recall = tps / len(y_true)
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def roc_curve(y_true,
              y_score,
              pos_label=None,
              drop_intermediate=True):
    fps, tps, thresholds = _binary_clf_curve(y_true,
                                             y_score,
                                             pos_label=pos_label)

    if drop_intermediate and len(fps) > 2:
        optimal_idxs = np.where(np.r_[True,
                                      np.logical_or(np.diff(fps, 2),
                                                    np.diff(tps, 2)),
                                      True])[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

    if tps.size == 0 or fps[0] != 0:
        # Add an extra threshold position if necessary
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        thresholds = np.r_[thresholds[0] + 1, thresholds]

    if fps[-1] <= 0:
        warnings.warn("No negative samples in y_true, "
                      "false positive value should be meaningless",
                      Exception)
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        warnings.warn("No positive samples in y_true, "
                      "true positive value should be meaningless",
                      Exception)
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    return fpr, tpr, thresholds
