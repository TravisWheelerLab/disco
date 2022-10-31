import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="disco",
    version="0.0.1",
    author="Wheeler Lab",
    author_email="colligan@arizona.edu",
    description="Analyze spectrograms with deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TravisWheelerLab/disco",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "pyyaml",
        "pytorch_lightning",
        "numpy",
        "matplotlib",
        "torchaudio",
        "pomegranate",
    ],
    setup_requires=[
        "pyyaml",
        "pytorch_lightning",
        "numpy",
        "matplotlib",
        "torchaudio",
        "pomegranate",
    ],
)
