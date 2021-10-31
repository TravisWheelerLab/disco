This tool is available as a pip package!
We recommend creating a virtual environment with
[conda](https://docs.conda.io/en/latest/) and then installing.

```
conda create -n <env_name> python=3.6
pip install beetles
```
If pip fails because of conflicting dependencies (which it probably will due to pomegranate), install the package without dependencies and install from requirements.txt
```
pip install --no-dependencies beetles
pip install -r requirements.txt
```
If this fails, install each package by hand 😢, or delete pomegranate in requirements.txt. Pomegranate pins some package version numbers that "conflict" with our software.

### Developing
Clone the repo locally and install via setup.py.

```
git clone git@github.com:TravisWheelerLab/beetles-cnn.git
cd beetles-cnn
pip install -r requirements.txt
python setup.py develop
```
