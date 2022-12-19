import helpers, dataset, correction
from ruska import Ruska


def run_baran(c: dict):
    """
    Wrapper for Ruska to run parameterized Cleaning Experiments.
    @param c: a dictionary that contains all parameters the Cleaning Experiment requires.
    @return: A dictionary containing measurements and the configuration of the measurement.
    """
    version = c['run'] + 1  # dataset versions are 1-indexed, Ruska runs are 0-indexed.
    data_dict = helpers.get_data_dict(c['dataset'], c['error_fraction'], version, c['error_class'])

    data = dataset.Dataset(data_dict)
    data.detected_cells = dict(data.get_actual_errors_dictionary())

    app = correction.Correction()
    app.VERBOSE = False
    correction_dictionary = app.run(data)
    p, r, f = data.get_data_cleaning_evaluation(correction_dictionary)[-3:]
    return {"result": {"precision": p, "recall": r, "f1": f}, "config": c}


if __name__ == "__main__":
    rsk_openml = Ruska(
        name="2022-12-13-openml",
        description="Measure all openml datasets to benchmark with Baran.",
        commit="",
        config={
            "dataset": "letter",
            "error_class": "simple_mcar",
            "error_fraction": 1,
        },
        ranges={
            "dataset": [137, 1481, 184, 41027, 4135, 42493, 6],
            "error_fraction": [1, 5, 10]
        },
        runs=3,
        save_path="/root/measurements/",
    )

    rsk_openml.run(experiment=run_baran, parallel=False)
