#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('./requirements.txt', 'r') as reqs:
    requirements = reqs.read()


setup(
    name='plt',
    version='0.1.0',
    description="Fast plot from command line",
    long_description=readme + '\n\n' + history,
    author="Hicham Randrianarivo",
    author_email='h.randrianarivo@qwant.com',
    url='https://github.com/chicham/plt',
    packages=find_packages(include=['plt']),
    entry_points={
        'console_scripts': [
            'plt=plt.cli:plot'
        ]
    },
    include_package_data=True,
    license="MIT license",
    zip_safe=False,
    keywords='plt',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    install_requires=requirements
)
