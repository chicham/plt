# -*- coding: utf-8 -*-

"""Main module."""

from absl import app, flags
import seaborn as sns
import pandas as pd
from pathlib import Path


def run():
    app.run(main)


def plot():
    src = Path(FLAGS.src).expanduser().absolute()
    names = FLAGS.names
    __import__('ipdb').set_trace()
    if names is not None:
        header = 0
    else:
        header = 'infer'
    df = pd.read_csv(sep=FLAGS.sep, header=header)


PLOTS = {
    'plot': plot,
    'image': ''
}

FLAGS = flags.FLAGS
flags.DEFINE_enum('plot', 'plot', PLOTS.keys(), 'Type of plot', short_name='p')
flags.DEFINE_string('xlabel', 'x', 'Name of the x-label', short_name='x')
flags.DEFINE_string('ylabel', 'y', 'Name of the y-label', short_name='y')
flags.DEFINE_string('title', '', 'Plot title', short_name='t')
flags.DEFINE_string('src', None, 'Source file title', short_name='s')
flags.DEFINE_string('sep', ',', 'Elements separator in src')
# flags.DEFINE_list('names', None, ',', 'Names of the columns in src ')


def main(argv):
    PLOTS[FLAGS.plot]()
