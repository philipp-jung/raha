import os
from pathlib import Path
import logging
import helpers, dataset, correction
from ruska import Ruska

import dotenv
dotenv.load_dotenv()


def run_baran(i: int, c: dict):
    """
    Wrapper for Ruska to run parameterized Cleaning Experiments.
    @param c: a dictionary that contains all parameters the Cleaning Experiment requires.
    @return: A dictionary containing measurements and the configuration of the measurement.
    """
    logger = logging.getLogger('ruska')
    logger.info(f'Started experiment {i}.')
    logger.debug(f'Started experiment {i} with config {c}.')

    version = c['run'] + 1  # dataset versions are 1-indexed, Ruska runs are 0-indexed.
    data_dict = helpers.get_data_dict(c['dataset'], c['error_fraction'], version, c['error_class'])
    data_dict['n_rows'] = c.get('n_rows')

    data = dataset.Dataset(data_dict)
    data.detected_cells = dict(data.get_actual_errors_dictionary())

    app = correction.Correction(cripple=c['cripple'])
    app.VERBOSE = False
    p, r, f = app.run(data)
    logger.info(f'Finished experiment {i}.')
    return {"result": {"precision": p, "recall": r, "f1": f}, "config": c}


if __name__ == "__main__":
    experiment_name = "2023-11-10-baseline-baran-vs-mirmir"
    save_path = "/root/measurements/"

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

    fh = logging.FileHandler(str(Path(save_path)/Path(experiment_name))+'.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logging.root.addHandler(fh)

    rsk_baran = Ruska(
        name=experiment_name+'-baran',
        save_path=save_path,
        description="Make a baseline for baran vs mirmir",
        commit="",
        config={
            "dataset": "beers",
            "error_class": "simple_mcar",
            "error_fraction": 1,
            "n_rows": None,
            "cripple": None,
        },
        ranges={
            "dataset": ['beers', 'hospital', 'flights', 'rayyan'],
        },
        runs=3,
        chat_id=os.environ['TELEGRAM_CHAT_ID'],
        token=os.environ['TELEGRAM_BOT_TOKEN'],
    )

    rsk_renuver = Ruska(
        name=experiment_name+'-renuver',
        save_path=save_path,
        description="Make a baseline for baran vs mirmir",
        commit="",
        config={
            "dataset": "beers",
            "error_class": "simple_mcar",
            "error_fraction": 1,
            "n_rows": None,
            "cripple": None,
        },
        ranges={
            "dataset": ['cars', 'glass', 'restaurant', 'bridges'],
            "error_fraction": [1, 3],
        },
        runs=3,
        chat_id=os.environ['TELEGRAM_CHAT_ID'],
        token=os.environ['TELEGRAM_BOT_TOKEN'],
    )

    rsk_openml = Ruska(
        name=experiment_name+'-openml',
        save_path=save_path,
        description="Make a baseline for baran vs mirmir",
        commit="",
        config={
            "dataset": "beers",
            "error_class": "simple_mcar",
            "error_fraction": 5,
            "n_rows": 1000,
            "cripple": None,
        },
        ranges={
            "dataset": [6, 137, 1481, 184, 41027, 43572],
            "error_class": ['imputer_simple_mcar', 'simple_mcar'],
        },
        runs=3,
        chat_id=os.environ['TELEGRAM_CHAT_ID'],
        token=os.environ['TELEGRAM_BOT_TOKEN'],
    )

    rsk_baran.run(experiment=run_baran, parallel=False)
    rsk_renuver.run(experiment=run_baran, parallel=False)
    rsk_openml.run(experiment=run_baran, parallel=False)
