## Using:
`beetles train <OPTS>`.
Neural networks are fickle things that require extensive hyperparameter tuning to do well and as such there are many tunable arguments to pass into this utility. Sensible defaults have been provided for most of the hyperparameters. You only have to worry about providing a path to the data, a place to save logs, the number of GPUs and nodes you're running the code on, and whether you want to bootstrap sample your training set.

Example (with bootstrapping enable)
```
beetles train --data_path <output of extract> --log_dir ~/my_logs/ --gpus 1 --bootstrap
```
`--data_path` should have three subdirectories: `train`, `test`, and `validation`, containing the train files, test files, and validation files, respectively.