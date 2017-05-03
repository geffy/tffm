import os

from setuptools import find_packages, setup


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as opened:
        return opened.read()


def main():
    setup(
        name='tffm',
        version='1.0.0a1',
        author="Mikhail Trofimov",
        author_email="mikhail.trofimov@phystech.edu",
        url='https://github.com/geffy/tffm',
        description=('TensforFlow implementation of arbitrary order '
                     'Factorization Machine'),
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
        ],
        license='MIT',
        install_requires=[
            'scikit-learn',
            'numpy',
            'tqdm'
        ],
        packages=find_packages()
    )


if __name__ == "__main__":
    main()
