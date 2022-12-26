import raha
from ruska import Ruska


def run_baran(c: dict):
    """
    Wrapper for Ruska to run parameterized Cleaning Experiments.
    @param c: a dictionary that contains all parameters the Cleaning Experiment requires.
    @return: A dictionary containing measurements and the configuration of the measurement.
    """
    version = c['run'] + 1  # dataset versions are 1-indexed, Ruska runs are 0-indexed.
    data_dict = raha.helpers.get_data_dict(c['dataset'], c['error_fraction'], version, c['error_class'])

    data = raha.Dataset(data_dict, n_rows=c["n_rows"])
    data.detected_cells = dict(data.get_actual_errors_dictionary())

    app = raha.Correction(c['labeling_budget'], c["classification_model"], c['feature_generators'],
                          c['vicinity_orders'], c["vicinity_feature_generator"], c["imputer_cache_model"],
                          c["n_best_pdeps"], c['training_time_limit'], c['rule_based_value_cleaning'],
                          c['synth_tuples'])
    app.VERBOSE = False
    seed = None
    correction_dictionary = app.run(data, seed)
    p, r, f = data.get_data_cleaning_evaluation(correction_dictionary)[-3:]
    return {"result": {"precision": p, "recall": r, "f1": f}, "config": c}


if __name__ == "__main__":
    rsk_openml = Ruska(
        name="2022-12-25-openml",
        description="Misst den Effekt des imputer feature generators auf den generierten openml Datensätzen.",
        commit="",
        config={
            "dataset": "letter",
            "error_class": "simple_mcar_imputer",
            "error_fraction": 1,
            "labeling_budget": 20,
            "synth_tuples": 20,
            "imputer_cache_model": False,
            "training_time_limit": 30,
            "feature_generators": ["domain", "vicinity", "value"],
            "classification_model": "ABC",
            "vicinity_orders": [1, 2],
            "vicinity_feature_generator": "pdep",
            "n_rows": None,
            "n_best_pdeps": 3,
            "rule_based_value_cleaning": "V4",
        },
        ranges={
            "dataset": [137, 1481, 184, 41027, 4135, 42493, 6],
            "error_fraction": [1, 5, 10],
            "error_class": ["imputer_simple_mcar", "simple_mcar"],
            "feature_generators": [["domain", "vicinity", "value"], ["domain", "vicinity", "value", "imputer"]]
        },
        runs=3,
        save_path="/root/measurements/",
    )

    rsk_openml.run(experiment=run_baran, parallel=True)
