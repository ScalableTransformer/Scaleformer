# -*- coding: utf-8 -*-
from pathlib import Path
from setuptools import setup
from setuptools import find_packages

pkg = {}
with open('scaleformer/version.py') as fp:
    exec(fp.read(), pkg)

root = Path(__file__).resolve().parent
with open(root / 'README.md', encoding='utf-8') as f:
    long_description = f.read()

repo = 'https://github.com/ScalableTransformer/Scaleformer'
version = pkg['version']

setup(
    name=pkg['name'],
    packages=find_packages(),
    author=pkg['author'],
    author_email=pkg['author_email'],
    description=pkg['description'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=pkg['keywords'],
    version=version,
    install_requires=pkg['install_requires'],
    package_data=pkg['package_data'],
    license='MIT',
    url=repo,
    download_url=F'{repo}/archive/{version}.tar.gz',
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10'
    ]
)
