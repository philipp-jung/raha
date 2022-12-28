import logging
from ruska import Ruska
import raha
from pathlib import Path


def run_baran(c: dict):
    """
    Wrapper for Ruska to run parameterized Cleaning Experiments.
    @param c: a dictionary that contains all parameters the Cleaning Experiment requires.
    @return: A dictionary containing measurements and the configuration of the measurement.
    """
    version = c['run'] + 1  # dataset versions are 1-indexed, Ruska runs are 0-indexed.
    data_dict = raha.helpers.get_data_dict(c['dataset'], c['error_fraction'], version, c['error_class'])

    try:
        version = (
            c["run"] + 1
        )  # dataset versions are 1-indexed, Ruska runs are 0-indexed.
        data_dict = raha.helpers.get_data_dict(
            c["dataset"], c["error_fraction"], version, c["error_class"]
        )

        data = raha.Dataset(data_dict, n_rows=c["n_rows"])
        data.detected_cells = dict(data.get_actual_errors_dictionary())

        app = raha.Correction(
            c["labeling_budget"],
            c["classification_model"],
            c["feature_generators"],
            c["vicinity_orders"],
            c["vicinity_feature_generator"],
            c["imputer_cache_model"],
            c["n_best_pdeps"],
            c["training_time_limit"],
            c["rule_based_value_cleaning"],
            c["synth_tuples"],
        )
        app.VERBOSE = False
        seed = None
        correction_dictionary = app.run(data, seed)
        p, r, f = data.get_data_cleaning_evaluation(correction_dictionary)[-3:]
        return {"result": {"precision": p, "recall": r, "f1": f}, "config": c}
    except Exception as e:
        return {"result": e, "config": c}


if __name__ == "__main__":
    experiment_name = "2022-12-27-openml-imputer"
    save_path = "/Users/philipp/code/raha/raha"

    logging.root.handlers = []  # deletes the default StreamHandler to stderr.
    logging.getLogger("ruska").setLevel(logging.DEBUG)

    # create formatter to use with the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # create console handler with a higher log level to reduce noise
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # ch.setLevel(logging.DEBUG)

    ch.setFormatter(formatter)
    logging.root.addHandler(ch)

    fh = logging.FileHandler(str(Path(save_path)/Path(experiment_name))+'.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logging.root.addHandler(fh)

    rsk_openml = Ruska(
        name=experiment_name,
        description="Misst den Effekt des imputer feature generators auf den generierten openml Datensätzen.",
        commit="",
        config={
            "dataset": "debug",
            "error_class": "imputer_simple_mcar",
            "error_fraction": 1,
            "labeling_budget": 3,
            "synth_tuples": 20,
            "imputer_cache_model": False,
            "training_time_limit": 30,
            "feature_generators": ["domain", "vicinity", "value"],
            "classification_model": "ABC",
            "vicinity_orders": [1, 2],
            "vicinity_feature_generator": "pdep",
            "n_rows": 2500,
            "n_best_pdeps": 3,
            "rule_based_value_cleaning": "V4",
        },
        ranges={
            "dataset": ['rayyan', 'beers'],
            # "dataset": [137, 1481, 184, 41027, 4135, 42493, 6],
            # "feature_generators": [
            #     ["domain", "vicinity", "value"],
            #     ["domain", "vicinity", "value", "imputer"],
            # ],
            # "error_class": ["imputer_simple_mcar", "simple_mcar"]
        },
        runs=1,
        save_path=save_path
        # save_path="/root/measurements/",
    )

    rsk_openml.run(experiment=run_baran, parallel=True)
