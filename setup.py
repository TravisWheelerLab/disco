from setuptools import setup, find_packages

setup_args = dict(
    name="beetles",
    version="1.0",
    packages=find_packages(),
    package_data={
        "beetles": ["resources/annotations_key.json", "resources/hmm_weights.json"]
    },
    url="https://github.com/TravisWheelerLab/beetles-cnn/archive/refs/tags/v0.0.1-alpha.tar.gz",
    license="MIT",
    author="thomas colligan",
    author_email="thomas.colligan@umontana.edu",
    description="Tool for labeling and training DNNs on spectrogram data",
    entry_points="""
        [console_scripts]
        beetles=beetles:main
    """,
    install_requires=[
        "torch",
        "numpy",
        "torchaudio",
        "pytorch_lightning",
        "matplotlib<3.5",
        "tqdm",
        "pandas",
        "requests",
        "torchmetrics",
        "scikit_learn",
        "pomegranate",
    ],
)

setup(**setup_args)
