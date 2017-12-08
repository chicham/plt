# -*- coding: utf-8 -*-

"""Console script for plot."""

import click
from odo import odo
import seaborn as sns
import pylab as P
import pandas as pd
from pathlib import Path
from .plot import *
import json
import ipdb


@click.group(invoke_without_command=True)
@click.argument('src')
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
@click.option('--legend', default=True)
@click.option('--logx', is_flag=True, default=False)
@click.option('--logy', is_flag=True, default=False)
@click.option('--loglog', is_flag=True, default=False)
@click.option('--xlim', default=None)
@click.option('--ylim', default=None)
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
         ylabel):
    src = Path(src).absolute()
    data = odo(str(src), pd.DataFrame)
    sns.set(context=context,
            font_scale=font_scale,
            style=style,
            palette=palette,
            font=font,
            color_codes=color_codes,
            rc=rc)

    if rc is not None:
        rc = json.load(open(Path(rc).absolute(), 'r'))

    plot_args = {
        'title': title,
        'subplots': subplots,
        'sharex': sharex,
        'sharey': sharey,
        'use_index': use_index,
        'legend': legend,
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

    if ctx.invoked_subcommand is None:
        data.plot(**plot_args)
        if xlabel is not None:
            P.xlabel(xlabel)
        if ylabel is not None:
            P.ylabel(ylabel)
        P.show()
    else:
        ctx.obj = {}
        ctx.obj['data'] = data
        ctx.obj['plot_args'] = plot_args
        ctx.obj['xlabel'] = xlabel
        ctx.obj['ylabel'] = ylabel


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

    xlabel = ctx.obj['xlabel']
    ylabel = ctx.obj['ylabel']

    data.plot(**ctx.obj['plot_args'])
    if xlabel is not None:
        P.xlabel(xlabel)
    if ylabel is not None:
        P.ylabel(ylabel)
    P.show()


if __name__ == "__main__":
    plot()
