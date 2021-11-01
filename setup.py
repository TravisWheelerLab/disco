from setuptools import setup

setup_args = dict(
    name="beetles",
    version="0.0.2",
    packages=["beetles", "beetles.models"],
    package_dir={"": "."},
    url="https://github.com/TravisWheelerLab/beetles-cnn/archive/refs/tags/v0.0.1-alpha.tar.gz",
    license="MIT",
    author="thomas colligan",
    author_email="thomas.colligan@umontana.edu",
    description="Tool for labeling and training DNNs on spectrogram data",
    include_package_data=True,
    entry_points="""
        [console_scripts]
        beetles=beetles:main
    """,
    install_requires=[
        "torch",
        "numpy",
        "torchaudio",
        "pytorch_lightning",
        "matplotlib",
        "tqdm",
        "pandas",
        "requests",
        "torchmetrics",
        "scikit_learn",
        "pomegranate"
    ],
)

setup(**setup_args)
