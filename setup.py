import os

from setuptools import find_packages, setup


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as opened:
        return opened.read()


def main():
    with open('requirements.txt') as f:
        required = f.read().splitlines()

    setup(
        name='tffm',
        version='1.0.0',
        author=None,
        author_email=None,
        description=('TensforFlow implementation of arbitrary order '
                     'Factorization Machine'),
        packages=find_packages(),
        long_description=read('README.md'),
        install_requires=required
       )


if __name__ == "__main__":
    main()
