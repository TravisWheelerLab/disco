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
        "pyyaml>=6.0",
        "pytorch_lightning>1.7",
        "numpy>=1.2",
        "matplotlib>3",
        "torchaudio>=0.12.1",
        "pomegranate>=0.14.8",
    ],
    setup_requires=[
        "pyyaml>=6.0",
        "pytorch_lightning>1.7",
        "numpy>=1.2",
        "matplotlib>3",
        "torchaudio>=0.12.1",
        "pomegranate>=0.14.8",
    ],
)
