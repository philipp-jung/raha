# Data Cleaning
Based on `raha`, we set out to improve data cleaning.

## Installation
- Install python version 3.7.10. Consider using pyenv to manage python versions.
- Install dependencies via `python -m pip install -r requirements.txt`.
- In the root directory, clone my fork of [datawig](https://github.com/philipp-jung/datawig),
which is a dependency for the imputer feature generator.
- To install datawig, run `python -m pip install -e datawig/.`. You may need to
  install dependencies of datawig for this via `python -m pip install -r datawig/requirements.txt`.

## How to use it
- Run `python raha/correction.py` to clean data with the configuration specified
  at the bottom of `correction.py`.
- For more fancy experimental setups, consider installing `ruska` and running
  `raha/run_experiments.py`.
