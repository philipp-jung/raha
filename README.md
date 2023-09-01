# Mirmir: A Holistic Value Imputation System
Mirmir is a state-of-the-art value cleaning system.

## Installation
Mirmir can be executed on any platform using `conda`.
To install Mirmir on your machine, follow these steps:

1) Install Miniforge3 on you machine.\
Follow the [official installation instructions](https://github.com/conda-forge/miniforge#download).
1) Clone this repository via `git clone https://github.com/philipp-jung/mirmir.git`.
1) In the folder into which you cloned the repository, run `conda env create -n mirmir -f environment.yml` to create a new conda environment called `mirmir`.

## How to use it
Follow these instructions to clean data with `mirmir`:

1) Run `conda activate mirmir` to activate the `mirmir` environment.
1) Navigate into the `src/` folder in the directory into which you cloned `mirmir`.
1) Run `python correction.py` to correct sample data errors.


## Experiments
For a fancy experimental setups, consider installing [`ruska`](https://github.com/philipp-jung/ruska) and running `raha/run_experiments.py`.
