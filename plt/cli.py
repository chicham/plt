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

    src = Path(src).absolute()
    data = odo(str(src), pd.DataFrame, delimiter=sep)

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

    if y is not None:
        if y.isdigit():
            y = data.columns[int(y)]
        if x is None:
            x = '0'
        if x.isdigit():
            x = data.columns[int(x)]

        data = data[[x, y]]

    ctx.obj = defaultdict(lambda: None)
    ctx.obj['title'] = title
    ctx.obj['legend'] = legend

    if ctx.invoked_subcommand is None:
        data.plot(**plot_args)
        ctx.obj['xlabel'] = xlabel or 'x'
        ctx.obj['ylabel'] = ylabel or 'y'
        on_end(ctx)
    else:
        ctx.obj['data'] = data
        ctx.obj['plot_args'] = plot_args


@plot.command()
@click.option('--resample', default=None)
@click.option('--method', default='sum')
@click.pass_context
def ts(ctx,
       resample,
       method):
    data = ctx.obj['data']
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
    data.plot(**ctx.obj['plot_args'])
    on_end(ctx)


@plot.command()
@click.option('--pos', default=None)
@click.option('-m', is_flag=True, default=False)
@click.pass_context
def ap(ctx, pos, m):
    data = ctx.obj['data']
    plot_args = ctx.obj['plot_args']
    y_true = data[data.columns[0]].values
    y_preds = data[data.columns[1]].values
    precision, recall, thresholds = precision_recall_curve(y_true, y_preds, pos)
    label = None
    if m:
        label = 'mAP: {:0.3f}'.format(np.trapz(recall, precision))
    plt.plot(recall, precision, label=label)
    ctx.obj['xlabel'] = ctx.obj['xlabel'] or 'Recall'
    ctx.obj['ylabel'] = ctx.obj['ylabel'] or 'Precision'
    ctx.obj['title'] = ctx.obj['title'] or 'Precision/Recall curve'

    on_end(ctx)


def fscore():
    pass


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
