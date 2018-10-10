# -*- coding: utf-8 -*-

"""Main module."""

from absl import app, flags
import seaborn as sns
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def run():
    app.run(main)


def plot():
    sns.set(context=FLAGS.context,
            style=FLAGS.style,
            palette=FLAGS.palette,
            font=FLAGS.font,
            color_codes=True)
    names = FLAGS.names
    if names is not None:
        header = 0
    else:
        header = 'infer'

    fig, ax = plt.subplots()
    src = FLAGS.src
    if len(src) > 1:
        for s in src:
            s = Path(s).expanduser().absolute()
            df = pd.read_csv(s,
                             sep=FLAGS.sep,
                             header=header,
                             names=names or [s.stem])
            df.plot(ax=ax)
    plt.title(FLAGS.title)
    plt.xlabel(FLAGS.xlabel)
    plt.ylabel(FLAGS.ylabel)
    plt.show()


PLOTS = {
    'plot': plot,
    'image': ''
}

CONTEXTS = ['paper', 'notebook', 'talk', 'poster']
STYLES = ['whitegrid', 'darkgrid', 'ticks', 'white', 'dark']
PALETTES = ['deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind']

FLAGS = flags.FLAGS
flags.DEFINE_enum('plot', 'plot', PLOTS.keys(), 'Type of plot', short_name='p')
flags.DEFINE_string('xlabel', 'x', 'Name of the x-label', short_name='x')
flags.DEFINE_string('ylabel', 'y', 'Name of the y-label', short_name='y')
flags.DEFINE_string('title', '', 'Plot title', short_name='t')
flags.DEFINE_multi_string('src', None, 'Sources file', short_name='s')
flags.DEFINE_string('sep', ',', 'Elements separator in src')
flags.DEFINE_spaceseplist('names', None, 'Name of the columns', short_name='n', comma_compat=True)
flags.DEFINE_enum('context', CONTEXTS[0], CONTEXTS, 'Context of the plot')
flags.DEFINE_enum('style', STYLES[0], STYLES, 'Style of the plot')
flags.DEFINE_string('font', 'sans-sherif', 'Font of the plot')
flags.DEFINE_enum('palette', PALETTES[0], PALETTES, 'Color palette')
flags.DEFINE_string('rc', None, 'file with parameter mappings of seaborn')


def main(argv):
    PLOTS[FLAGS.plot]()
