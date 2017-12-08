# -*- coding: utf-8 -*-

"""Main module."""

from odo.numpy_dtype import dshape_to_pandas
from odo import convert, Temp, CSV
from odo.utils import keywords

import pandas as pd
from datashape.predicates import isrecord
from toolz import keyfilter
import pylab as P


def _csv_to_dataframe(c, dshape=None, chunksize=None, **kwargs):
    header = {False: None, True: 0}.get(
        kwargs.pop('has_header', c.has_header), 'infer')

    sep = kwargs.pop(
        'sep', kwargs.pop('delimiter', c.dialect.get('delimiter', ',')))
    encoding = kwargs.pop('encoding', c.encoding)

    if dshape:
        dtypes, parse_dates = dshape_to_pandas(dshape)
        if isrecord(dshape.measure):
            names = kwargs.get('names', dshape.measure.names)
        else:
            names = kwargs.get('names')
    else:
        dtypes = parse_dates = names = None

    usecols = kwargs.pop('usecols', None)
    if parse_dates and usecols:
        parse_dates = [col for col in parse_dates if col in usecols]

    # See read_csv docs for header for reasoning
    if names:
        try:
            with c.open() as f:
                found_names = pd.read_csv(f,
                                          nrows=1,
                                          encoding=encoding,
                                          sep=sep)
        except StopIteration:
            with c.open() as f:
                found_names = pd.read_csv(f, encoding=encoding, sep=sep)
    if names and header == 'infer':
        if [n.strip() for n in found_names] == [n.strip() for n in names]:
            header = 0
        elif (all(re.match('^\s*\D\w*\s*$', n) for n in found_names) and
                not all(dt == datashape.string for dt in dshape.measure.types)):
            header = 0
        else:
            header = None

    kwargs = keyfilter(keywords(pd.read_csv).__contains__, kwargs)
    with c.open() as f:
        return pd.read_csv(f,
                           header=header,
                           sep=sep,
                           encoding=encoding,
                           dtype=dtypes,
                           parse_dates=parse_dates,
                           names=names,
                           chunksize=chunksize,
                           usecols=usecols,
                           **kwargs)

@convert.register(pd.DataFrame, (Temp(CSV), CSV), cost=10.0)
def csv_to_dataframe(c, dshape=None, chunksize=None, nrows=None, **kwargs):
    try:
        return _csv_to_dataframe(c, dshape=dshape, chunksize=chunksize,
                                 nrows=nrows, **kwargs)
    except StopIteration:
        if nrows:
            return _csv_to_dataframe(c, dshape=dshape, chunksize=chunksize,
                                     **kwargs)
        else:
            raise
