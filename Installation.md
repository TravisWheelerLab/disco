This tool is available as a pip package!
We recommend creating a virtual environment with
[conda](https://docs.conda.io/en/latest/) and then installing.

```
conda create -n <env_name> python=3.8
conda activate <env_name>
pip install beetles
```
If pip fails because of conflicting dependencies, install the package without dependencies and install from requirements.txt
```
pip install --no-dependencies beetles
pip install -r requirements.txt
```
If this fails, install each package by hand 😢, or delete the offending package in requirements.txt and try `pip install -r requirements.txt` again, then install the offending package by hand.

### Developing
Clone the repo locally and install via setup.py.

```
git clone git@github.com:TravisWheelerLab/beetles-cnn.git
cd beetles-cnn
pip install -r requirements.txt
python setup.py develop
```
