# -*- coding: utf-8 -*-
"""
Functions for loading test data setss
"""

import os.path as op
from pkg_resources import resource_filename
import numpy as np
from sklearn.utils import Bunch


def _load_data(dfiles, dlabels):
    """
    Loads `dfiles` for `dset` and return Bunch with data and labels

    Parameters
    ----------
    dset : {'sim', 'digits'}
        Dataset to load
    dfiles : list of str
        Data files in `dset`

    Returns
    -------
    data : :obj:`sklearn.utils.Bunch`
        With keys `data` and `labels`
    """

    # space versus comma-delimited files (ugh)
    try:
        data = [np.loadtxt(fn) for fn in dfiles]
    except ValueError:
        data = [np.loadtxt(fn, delimiter=',') for fn in dfiles]

    return Bunch(data=data, labels=np.loadtxt(dlabels,dtype=int))

def load_data(file_list, labels):
    """
    Loads "digits" dataset with four datatypes

    Returns
    -------
    digits : :obj:`sklearn.utils.Bunch`
        Dictionary-like object with keys ['data', 'labels']
    """

    return _load_data(file_list, labels)
