from setuptools import setup

setup_args = dict(
    name='beetles',
    version='0.0.1',
    packages=['beetles', 'beetles.models'],
    package_dir={'': '.'},
    url='https://github.com/TravisWheelerLab/beetles-cnn/',
    license='MIT',
    author='thomas colligan',
    author_email='thomas.colligan@umontana.edu',
    description='Tool for labeling and training DNNs on spectrogram data',
    entry_points="""
        [console_scripts]
        beetles=beetles.cli:main
    """,
    install_requires=['matplotlib',
                      'pomegranate',
                      'torchaudio',
                      'pytorch_lightning',
                      'tqdm',
                      'numpy',
                      'pandas',
                      'torch',
                      'requests',
                      'torchmetrics',
                      'scikit_learn']
)

setup(**setup_args)
