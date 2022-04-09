import os
from raha.correction import Correction
from raha.dataset import Dataset

def validate_raha():
    for dataset_name in ['flights', 'beers', 'rayyan', 'hospital']:
        dataset_dictionary = {
            "name": dataset_name,
            "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "dirty.csv")),
            "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "clean.csv"))
        }
        data = Dataset(dataset_dictionary)
        data.detected_cells = dict(data.get_actual_errors_dictionary())
        app = Correction()
        correction_dictionary = app.run(data)
        p, r, f = data.get_data_cleaning_evaluation(correction_dictionary)[-3:]
        print("Baran's performance on {}:\nPrecision = {:.2f}\nRecall = {:.2f}\nF1 = {:.2f}".format(data.name, p, r, f))


if __name__ == '__main__':
    for run in range(3):
        print(f'starting run {run}')
        validate_raha()
