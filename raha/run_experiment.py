import os
import raha
import logging
from ruska import Ruska
from pathlib import Path

import dotenv

dotenv.load_dotenv()


def run_baran(i: int, c: dict):
    """
    Wrapper for Ruska to run parameterized Cleaning Experiments.
    @param c: a dictionary that contains all parameters the Cleaning Experiment requires.
    @return: A dictionary containing measurements and the configuration of the measurement.
    """
    logger = logging.getLogger("ruska")
    logger.info(f"Started experiment {i}.")
    logger.debug(f"Started experiment {i} with config {c}.")
    version = c["run"] + 1  # dataset versions are 1-indexed, Ruska runs are 0-indexed.
    data_dict = raha.helpers.get_data_dict(
        c["dataset"], c["error_fraction"], version, c["error_class"]
    )

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
            c["synth_tuples_error_threshold"]
        )
        app.VERBOSE = False
        seed = None
        correction_dictionary = app.run(data, seed)
        p, r, f = data.get_data_cleaning_evaluation(correction_dictionary)[-3:]
        logger.info(f"Finished experiment {i}.")
        return {"result": {"precision": p, "recall": r, "f1": f}, "config": c}
    except Exception as e:
        logger.error(f"Finished experiment {i} with an error: {e}.")
        return {"result": e, "config": c}


if __name__ == "__main__":
    experiment_name = "2023-02-22-autogluon-meta-learner"
    save_path = "/root/measurements"

    logging.root.handlers = []  # deletes the default StreamHandler to stderr.
    logging.getLogger("ruska").setLevel(logging.DEBUG)

    # create formatter to use with the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # create console handler with a higher log level to reduce noise
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    ch.setFormatter(formatter)
    logging.root.addHandler(ch)

    fh = logging.FileHandler(str(Path(save_path) / Path(experiment_name)) + ".log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logging.root.addHandler(fh)

    rsk_openml = Ruska(
        name=f"{experiment_name}-openml",
        description="Binary Voting on OpenML Datasets.",
        commit="",
        config={
            "dataset": "1481",
            "error_class": "simple_mcar",
            "error_fraction": 1,
            "labeling_budget": 20,
            "synth_tuples": 100,
            "synth_tuples_error_threshold": 0,
            "imputer_cache_model": False,
            "training_time_limit": 30,
            "feature_generators": ["domain", "vicinity", "value"],
            "classification_model": "AG",
            "vicinity_orders": [1, 2],
            "vicinity_feature_generator": "pdep",
            "n_rows": None,
            "n_best_pdeps": 3,
            "rule_based_value_cleaning": "V5",
        },
        ranges={
            "dataset": [137, 1481, 184, 41027],
            "error_fraction": [1, 5, 10],
        },
        runs=3,
        save_path=save_path,
        chat_id=os.environ["TELEGRAM_CHAT_ID"],
        token=os.environ["TELEGRAM_BOT_TOKEN"],
    )

    rsk_baran = Ruska(
        name=f"{experiment_name}-baran",
        description="Binary Voting on Baran Datasets",
        commit="",
        config={
            "dataset": "1481",
            "error_class": "simple_mcar",
            "error_fraction": 1,
            "labeling_budget": 20,
            "synth_tuples": 100,
            "synth_tuples_error_threshold": 0,
            "imputer_cache_model": False,
            "training_time_limit": 30,
            "feature_generators": ["domain", "vicinity", "value"],
            "classification_model": "AG",
            "vicinity_orders": [1, 2],
            "vicinity_feature_generator": "pdep",
            "n_rows": None,
            "n_best_pdeps": 3,
            "rule_based_value_cleaning": "V5",
        },
        ranges={
            "dataset": ["beers", "flights", "hospital", "rayyan"],
        },
        runs=3,
        save_path=save_path,
        chat_id=os.environ["TELEGRAM_CHAT_ID"],
        token=os.environ["TELEGRAM_BOT_TOKEN"],
    )

    rsk_renuver = Ruska(
        name=f"{experiment_name}-renuver",
        description="Binary Voting on Renuver Datasets.",
        commit="",
        config={
            "dataset": "1481",
            "error_class": "simple_mcar",
            "error_fraction": 1,
            "labeling_budget": 20,
            "synth_tuples": 100,
            "synth_tuples_error_threshold": 0,
            "imputer_cache_model": False,
            "training_time_limit": 30,
            "feature_generators": ["domain", "vicinity", "value"],
            "classification_model": "AG",
            "vicinity_orders": [1, 2],
            "vicinity_feature_generator": "pdep",
            "n_rows": None,
            "n_best_pdeps": 3,
            "rule_based_value_cleaning": "V5",
        },
        ranges={
            "dataset": ["bridges", "cars", "glass", "restaurant"],
            "error_fraction": [1, 2, 3, 4, 5]
        },
        runs=3,
        save_path=save_path,
        chat_id=os.environ["TELEGRAM_CHAT_ID"],
        token=os.environ["TELEGRAM_BOT_TOKEN"],
    )

    rsk_openml.run(experiment=run_baran, parallel=True)
    rsk_baran.run(experiment=run_baran, parallel=True)
    rsk_renuver.run(experiment=run_baran, parallel=True)
