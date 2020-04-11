#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

# TODO - convert md to RST before submitting to PyPI
with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
                    'Click>=7.0',
                    'gym>=0.17.1',
                    'numpy>=1.18',
                    'mesa>=0.8.6'
                ]

setup_requirements = []

test_requirements = []

setup(
    author="Sharada Mohanty",
    author_email='spmohanty91@gmail.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Reinforcement Learning on Disease Transmission",
    entry_points={
        'console_scripts': [
            'rogi-rl-demo=rogi_rl.cli:demo',
            'rogi-rl-profile_perf=rogi_rl.cli:profile_perf',
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='rogi_rl',
    name='rogi_rl',
    packages=find_packages(include=['rogi_rl', 'rogi_rl.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/spMohanty/rogi_rl',
    version='0.1.0',
    zip_safe=False,
)
