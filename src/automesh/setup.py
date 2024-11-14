try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Automesh generates meshes from scan',
    'author': 'Duane Malcolm',
    'url': 'https://bitbucket.org/abi_breast_research/automesh',
    'download_url': 'https://bitbucket.org/abi_breast_research/automesh.git',
    'author_email': 'd.malcolm@auckland.ac.nz',
    'version': '0.1',
    'install_requires': ['nose', 'numpy'],
    'packages': ['automesh'],
    'scripts': [],
    'name': 'automesh'
}

setup(**config, requires=['numpy'])
