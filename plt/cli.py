# -*- coding: utf-8 -*-

"""Console script for plot."""

import ipdb

import click
import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.plot as rplt

from .plot import precision_recall_curve
from collections import defaultdict
from glob import glob
from pathlib import Path
from itertools import repeat
from odo import odo
from skimage import exposure, util

N_COLORS = 6
N_MARKERS = 3


@click.group(invoke_without_command=True)
@click.argument('src', required=False)
@click.option('-x', default=None,
              help="Column to be used as x-axis (Integer or String)")
@click.option('-y', default=None,
              help="Column to be used as y-axis (Integer or String)")
@click.option('--xlabel', default=None, help="Name of x-axis")
@click.option('--ylabel', default=None,  help="Name of y-axis")
@click.option('--context', default='paper',
              help="Preset for the plot (paper, notebook, talk, poster)")
@click.option('--style', default='whitegrid',
              help="Axes style (whitegrid, darkgrid, dark, white, ticks)")
@click.option('--palette', default='deep',
              help="Seaborn palette (deep, muted, bright, pastel, dark, colorblind)")
@click.option('--font', default='sans-sherif')
@click.option('--font_scale', default=1)
@click.option('--rc', default=None,
              help="Path the configuration file in json format")
@click.option('--kind', default='line',
              help="Kind of plot (line, bar, barh, hist, box, kde, density,"
                   "area, pie, scatter, hexbin)")
@click.option('--subplots', is_flag=True, default=False,
              help="Make each plots on independant axes")
@click.option('--sharex', is_flag=True, default=False,
              help="With subplots, use a shared x-axis")
@click.option('--sharey', is_flag=True, default=False,
              help="With subplots, use a shared y-axis")
@click.option('--title', default=None, help="Title of the plot")
@click.option('--use_index', is_flag=True, default=True,
              help="Use the dataframe index for x-axis")
@click.option('--legend/--no-legend', is_flag=True, default=True,
              help="Plot the legend on the graph")
@click.option('--logx', is_flag=True, default=False,
              help="Use log scale for x-axis")
@click.option('--logy', is_flag=True, default=False,
              help="Use log scale for y-axis")
@click.option('--loglog', is_flag=True, default=False,
              help="Use log scale for both axis")
@click.option('--xlim', default=None, help="Set x-axis limit")
@click.option('--ylim', default=None, help="Set x-axis limit")
@click.option('--sep', default=',', help="Separator between columns in the file")
@click.pass_context
def plot(ctx,
         src,
         x,
         y,
         context,
         style,
         palette,
         font,
         font_scale,
         rc,
         kind,
         subplots,
         sharex,
         sharey,
         title,
         use_index,
         legend,
         logx,
         logy,
         loglog,
         xlim,
         ylim,
         xlabel,
         ylabel,
         sep):

    if rc is not None:
        rc = json.load(open(Path(rc).absolute(), 'r'))

    color_codes = palette
    sns.set(context=context,
            font_scale=font_scale,
            style=style,
            palette=palette,
            font=font,
            color_codes=color_codes,
            rc=rc)

    if xlim is not None:
        xlim = tuple(map(float, xlim.split(',')))
    if ylim is not None:
        ylim = tuple(map(float, ylim.split(',')))
    plot_args = {
        'subplots': subplots,
        'sharex': sharex,
        'sharey': sharey,
        'use_index': use_index,
        'logx': logx,
        'logy': logy,
        'loglog': loglog,
        'xlim': xlim,
        'ylim': ylim,
    }

    src = Path(src).expanduser().absolute()
    files = glob(str(src))
    # colors = sns.color_palette(palette, N_COLORS)
    # makers = ['s', 'x', 'o', '^']
    # styles = product(colors, makers)

    ctx.obj = defaultdict(lambda: None)
    ctx.obj['title'] = title
    ctx.obj['legend'] = legend
    ctx.obj['xlabel'] = xlabel or 'x'
    ctx.obj['ylabel'] = ylabel or 'y'
    ctx.obj['plot_args'] = plot_args
    ctx.obj['files'] = files
    ctx.obj['sep'] = sep
    # ctx.obj['styles'] = styles

    if ctx.invoked_subcommand is None:
        if subplots and len(files) > 1:
            plot_args['subplots'] = False
            fig, axes = plt.subplots(len(files), sharex=sharex, sharey=sharey)
        else:
            fig, axes = plt.subplots()
            axes = repeat(axes)
        for f, a in zip(files, axes):
            index = x or 0
            data = odo(str(f), pd.DataFrame, delimiter=sep, index_col=index)

            if y is not None:
                if y.isdigit():
                    y = data.columns[int(y)]
                if x is None:
                    x = '0'
                if x.isdigit():
                    x = data.columns[int(x)]

                data = data[[x, y]]
            data.plot(ax=a, **plot_args)
        on_end(ctx)


@plot.command()
@click.option('--gamma', default=.5)
@click.option('--gain', default=1.)
@click.pass_context
def raster(ctx,
           gamma: float,
           gain: float):
    f = ctx.obj['files'][0]

    with rasterio.open(f, 'r') as src:
        data = src.read()
        data = util.img_as_float(data)
        for ii, band in enumerate(data):
            band = exposure.adjust_gamma(band, gamma=gamma, gain=gain)
            data[ii] = exposure.equalize_adapthist(band)
        rplt.show(data)


@plot.command()
@click.option('--resample', default=None,
              help="Frequency offset (http://pandas.pydata.org/pandas-docs/"
                   "stable/timeseries.html#offset-aliases)")
@click.option('--method', default='sum',
              help="Method to aggregate (sum, mean, count)")
@click.pass_context
def ts(ctx,
       resample,
       method):
    data = ctx.obj['data']
    files = ctx.obj['files']
    sep = ctx.obj['sep']
    plot_args = ctx.obj['plot_args']
    subplots = plot_args['subplots']
    sharex = plot_args['sharex']
    sharey = plot_args['sharey']

    if subplots and len(files) > 1:
        plot_args['subplots'] = False
        fig, axes = plt.subplots(len(files), sharex=sharex, sharey=sharey)
    else:
        fig, axes = plt.subplots()
        axes = repeat(axes)

    for f, a in zip(files, axes):
        data = odo(str(f), pd.DataFrame, delimiter=sep)
        data = data.set_index(data.columns[0])
        if resample is not None:
            data = data.resample(resample)
            if method == 'count':
                data = data.count()
            elif method == 'mean':
                data = data.mean()
            elif method == 'sum':
                data = data.sum()
            else:
                pass

        ctx.obj['ylabel'] = ctx.obj['ylabel'] or method
        data.plot(ax=a, **ctx.obj['plot_args'])
    on_end(ctx)


@plot.command()
@click.option('--pos', default=None, help="Positive label")
@click.option('-m', is_flag=True, default=False,
              help="Add mean average accuracy in the legend")
@click.option('--gt', default=0, help="Column  of the groundtruth")
@click.option('--preds', default=1, help="Column of the predictions")
@click.pass_context
def ap(ctx, pos, m, gt, preds):
    sep = ctx.obj['sep']
    files = ctx.obj['files']

    for f in files:
        data = odo(str(f), pd.DataFrame, delimiter=sep)
        y_true = data[data.columns[gt]].values
        y_preds = data[data.columns[preds]].values
        precision, recall, _ = precision_recall_curve(y_true, y_preds, pos)
        label = None
        if m:
            label = 'mAP: {:0.3f}'.format(np.trapz(recall, precision))
        xlim = ctx.obj['plot_args']['xlim'] or (0, 1)
        plt.xlim(xlim)
        ylim = ctx.obj['plot_args']['ylim'] or (0, 1)
        plt.ylim(ylim)
        plt.plot(recall, precision, label=label)
        ctx.obj['xlabel'] = ctx.obj['xlabel'] or 'Recall'
        ctx.obj['ylabel'] = ctx.obj['ylabel'] or 'Precision'
        ctx.obj['title'] = ctx.obj['title'] or 'Precision/Recall curve'

    on_end(ctx)


def on_end(ctx):
    plt.xlabel(ctx.obj['xlabel'])
    plt.ylabel(ctx.obj['ylabel'])
    title = ctx.obj['title']
    if title is not None:
        plt.title(title)
    legend = ctx.obj['legend']
    if legend:
        plt.legend()
    plt.show()


if __name__ == "__main__":
    plot()
