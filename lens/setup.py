from setuptools import setup, find_packages

setup(
    name='lens_metric',
    version='0.2.0',
    description='A learnable evaluation metric for text simplification',
    author='Mounica Maddela',
    author_email='mmaddela3@gatech.edu',
    url='https://github.com/Yao-Dou/LENS',
    download_url='https://github.com/Yao-Dou/LENS/archive/refs/tags/v0.2.0.tar.gz',
    license='Apache license',
    packages=find_packages(),
    install_requires=[
        "sentencepiece >= 0.1.96",
        "pandas == 1.1.5",
        "transformers >= 4.8",
        "pytorch-lightning == 2.0.9",
        "jsonargparse == 3.13.1",
        "torch >= 2.0.1",
        "numpy >= 1.20.0",
        "torchmetrics >= 0.7",
        "sacrebleu >= 2.0.0",
        "scipy >=1.5.4" 
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable ',
        'Intended Audience :: Science/Research',
    ],
)
