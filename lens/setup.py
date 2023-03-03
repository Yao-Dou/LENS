from setuptools import setup

setup(
    name='lens',
    version='0.1.0',
    description='A new metric for text simplification',
    url='',
    author='Mounica Maddela',
    author_email='mmaddela3@gatech.edu',
    license='Apache license',
    packages=['lens'],
    install_requires=[
        "sentencepiece==0.1.96",
        "pandas==1.1.5",
        "transformers>=4.8",
        "pytorch-lightning==1.6.0",
        "jsonargparse==3.13.1",
        "torch >=1.6.0,<2",
        "numpy >= 1.20.0",
        "torchmetrics >= 0.7",
        "sacrebleu >= 2.0.0",
        "scipy >=1.5.4" ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: Apache License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)