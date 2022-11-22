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
    rsk_renuver = Ruska(
        name="2022-11-22-augment-working-renover",
        description="I finally got the augmented data approach to work.",
        commit="",
        config={
            "dataset": "bridges",
            "error_fraction": 1,
            "labeling_budget": 20,
            "imputer_cache_model": True,
            "training_time_limit": 30,
            "feature_generators": ["domain", "vicinity", "value"],
            "classification_model": "CV",
            "vicinity_orders": [1, 2],
            "vicinity_feature_generator": "pdep",
            "n_rows": None,
            "n_best_pdeps": 3,
            "rule_based_value_cleaning": "V3",
            "synth_tuples": 0
        },
        ranges={
            "dataset": ["bridges", "cars", "glass", "restaurant"],
            "error_fraction": [1, 2, 3, 4, 5],
            "synth_tuples": [0, 10, 20],
        },
        runs=3,
        save_path="/root/measurements/",
    )

    rsk_baran = Ruska(
        name="2022-11-22-augment-working-baran",
        description="I finally got the augmented data approach to work.",
        commit="",
        config={
            "dataset": "breast-cancer",
            "error_fraction": 1,
            "labeling_budget": 20,
            "imputer_cache_model": True,
            "training_time_limit": 30,
            "feature_generators": ["domain", "vicinity", "value"],
            "classification_model": "CV",
            "vicinity_orders": [1, 2],
            "vicinity_feature_generator": "pdep",
            "n_rows": None,
            "n_best_pdeps": 3,
            "rule_based_value_cleaning": "V3",
            "synth_tuples": 0
        },
        ranges={
            "dataset": ["beers", "flights", "hospital", "rayyan"],
            "error_fraction": [1, 5, 10, 30, 50],
            "synth_tuples": [0, 10, 20]
            },
        runs=3,
        save_path="/root/measurements/",
    )

    rsk_openml = Ruska(
        name="2022-11-13-openml-datasets",
        description="Try the new datasets",
        commit="",
        config={
            "dataset": "breast-cancer",
            "error_fraction": 1,
            "labeling_budget": 20,
            "imputer_cache_model": True,
            "training_time_limit": 30,
            "feature_generators": ["domain", "vicinity", "value"],
            "classification_model": "CV",
            "vicinity_orders": [1, 2],
            "vicinity_feature_generator": "pdep",
            "n_rows": None,
            "n_best_pdeps": 3,
            "rule_based_value_cleaning": "V3",
        },
        ranges={
            "dataset": [725, 310, 1046, 823, 137, 42493, 4135, 251, 151, 40922, 40498, 30, 1459, 1481, 184, 375, 32, 41027, 6, 40685],
            "feature_generators": [['domain', 'vicinity', 'value'], ['domain', 'vicinity', 'value', 'imputer']]
            },
        runs=3,
        save_path="/root/measurements/",
    )

    rsk_renuver.run(experiment=run_baran, parallel=False)
    rsk_baran.run(experiment=run_baran, parallel=False)
    # rsk_openml.run(experiment=run_baran, parallel=False)
