import raha
from ruska import Ruska


def run_baran(c: dict):
    """
    Wrapper for Ruska to run parameterized Cleaning Experiments.
    @param c: a dictionary that contains all parameters the Cleaning Experiment requires.
    @return: A dictionary containing measurements and the configuration of the measurement.
    """
    version = c['run'] + 1  # dataset versions are 1-indexed, Ruska runs are 0-indexed.
    data_dict = raha.helpers.get_data_dict(c['dataset'], c['error_fraction'], version)

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
        name="2022-12-13-openml",
        description="Measure all openml datasets to benchmark with Baran.",
        commit="",
        config={
            "dataset": "letter",
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
            "dataset": [1046, 137, 1459, 1481, 151, 184, 251, 30, 310, 32, 375, 40498, 40685, 40922, 41027, 4135, 42493, 6, 725, 823],
            "error_fraction": [1, 5, 10, 30, 50]
        },
        runs=3,
        save_path="/root/measurements/",
    )

    rsk_openml.run(experiment=run_baran, parallel=True)
