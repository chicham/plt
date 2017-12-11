# -*- coding: utf-8 -*-

"""Console script for plot."""

import click
import os
from odo import odo
import pandas as pd
from pathlib import Path
from .plot import *
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np
from glob import glob
from itertools import repeat


@click.group(invoke_without_command=True)
@click.argument('src', required=False)
@click.option('-x', default=None)
@click.option('-y', default=None)
@click.option('--xlabel', default=None)
@click.option('--ylabel', default=None)
@click.option('--context', default='paper')
@click.option('--style', default='whitegrid')
@click.option('--palette', default='deep')
@click.option('--font', default='sans-sherif')
@click.option('--font_scale', default=1)
@click.option('--color_codes', default=False)
@click.option('--rc', default=None)
@click.option('--kind', default='line')
@click.option('--subplots', is_flag=True, default=False)
@click.option('--sharex', is_flag=True, default=False)
@click.option('--sharey', is_flag=True, default=False)
@click.option('--title', default=None)
@click.option('--use_index', is_flag=True, default=True)
@click.option('--legend/--no-legend', is_flag=True, default=True)
@click.option('--logx', is_flag=True, default=False)
@click.option('--logy', is_flag=True, default=False)
@click.option('--loglog', is_flag=True, default=False)
@click.option('--xlim', default=None)
@click.option('--ylim', default=None)
@click.option('--sep', default=',')
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
         color_codes,
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

    sns.set(context=context,
            font_scale=font_scale,
            style=style,
            palette=palette,
            font=font,
            color_codes=color_codes,
            rc=rc)

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

    ctx.obj = defaultdict(lambda: None)
    ctx.obj['title'] = title
    ctx.obj['legend'] = legend
    ctx.obj['xlabel'] = xlabel or 'x'
    ctx.obj['ylabel'] = ylabel or 'y'
    ctx.obj['plot_args'] = plot_args
    ctx.obj['files'] = files
    ctx.obj['sep'] = sep


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
@click.option('--resample', default=None)
@click.option('--method', default='sum')
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
@click.option('--pos', default=None)
@click.option('-m', is_flag=True, default=False)
@click.option('--gt', default=0)
@click.option('--preds', default=1)
@click.pass_context
def ap(ctx, pos, m, gt, preds):
    sep = ctx.obj['sep']
    files = ctx.obj['files']
    plot_args = ctx.obj['plot_args']

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
