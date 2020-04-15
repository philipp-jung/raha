# Raha: A Configuration-Free Error Detection System
Detecting erroneous values is a key step in data cleaning.
Error detection algorithms usually require a user to provide input configurations in the form of integrity constraints or statistical parameters. However, providing a complete, yet correct, set of configurations for each new dataset is tedious and error-prone, as the user has to know about both the dataset and the error detection algorithms upfront.

Raha is a new configuration-free error detection system. The basic idea is that by generating a reasonably limited number of configurations for a set of error detection algorithms covering different types of data errors, we can generate an expressive feature vector for each entry in the dataset. Leveraging these feature vectors, we propose a novel sampling and classification approach that effectively chooses the most representative values for training. Furthermore, Raha can exploit  historical data, if available, to filter out irrelevant error detection algorithms and configurations.


## Installation
To install Raha, you can run:
```console
sudo python3 setup.py install
```
To uninstall Raha, you can run:
```console
sudo pip3 uninstall raha
```

## Usage
Running Raha is simple!
   - **Benchmarking Raha**: If you have a dirty dataset and its corresponding clean dataset and you want to benchmark Raha, please check the sample codes in `raha/benchmark.py` or `raha/detection.py`.
   - **Interactive error detection with Raha**: If you have a dirty dataset and you want to interatively detect data errors, please check the `raha/raha.ipynb` notebook. A user interface is also `coming soon`.
   ![Data Annotation](pictures/ui.png)   
   ![Promising Strategies](pictures/ui_strategies.png)   
   ![Drill Down](pictures/ui_clusters.png)   
   ![Dashboard](pictures/ui_dashboard.png) 

## Reference
You can find more information about this project and the authors [here](https://dl.acm.org/doi/abs/10.1145/3299869.3324956).
You can also use the following bib entry to cite this project/paper.
```
@inproceedings{mahdavi2019raha,
  title={Raha: A configuration-free error detection system},
  author={Mahdavi, Mohammad and Abedjan, Ziawasch and Castro Fernandez, Raul and Madden, Samuel and Ouzzani, Mourad and Stonebraker, Michael and Tang, Nan},
  booktitle={Proceedings of the 2019 International Conference on Management of Data (SIGMOD)},
  pages={865--882},
  year={2019},
  organization={ACM}
}
```

## Related Projects
Check out [Baran](https://github.com/bigdama/baran).