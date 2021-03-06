import os
import sys
import shutil
from setuptools import setup


def clean_repo():
    repo_folder = os.path.realpath(os.path.dirname(__file__))
    dist_folder = os.path.join(repo_folder, 'dist')
    build_folder = os.path.join(repo_folder, 'build')
    if os.path.isdir(dist_folder):
        shutil.rmtree(dist_folder, ignore_errors=True)
    if os.path.isdir(build_folder):
        shutil.rmtree(build_folder, ignore_errors=True)


# Read version string
_version = None
script_folder = os.path.realpath(os.path.dirname(__file__))
with open(os.path.join(script_folder, 'ibug', 'au_detection', '__init__.py')) as init:
    for line in init.read().splitlines():
        fields = line.replace('=', ' ').replace('\'', ' ').replace('\"', ' ').replace('\t', ' ').split()
        if len(fields) >= 2 and fields[0] == '__version__':
            _version = fields[1]
            break
if _version is None:
    sys.exit('Sorry, cannot find version information.')

# Installation
config = {
    'name': 'ibug_au_detection',
    'version': _version,
    'description': 'Facial action unit (AU) detection using EmoNet.',
    'author': 'Jing Yang & Jie Shen',
    'author_email': 'js1907@imperial.ac.uk',
    'packages': ['ibug.au_detection'],
    'install_requires': ['numpy>=1.16.0', 'torch>=1.1.0'],
    'zip_safe': False
}
clean_repo()
setup(**config)
clean_repo()
