import raha
from ruska import Ruska


def run_baran(c: dict):
    """
    Use run in [1,5] to not get an error.
    """

    # an das format von renuver angepasst.
    if c["dataset"] in ["bridges", "cars", "glass", "restaurant"]:  # renuver dataset
        rate_formatted = int(str(c["error_fraction"]).split(".")[1])
        run = c["run"] + 1
        data_dict = {
            "name": c["dataset"],
            "path": f"../datasets/renuver/{c['dataset']}/{c['dataset']}_{rate_formatted}_{run}.csv",
            "clean_path": f"../datasets/renuver/{c['dataset']}/clean.csv",
        }
    elif c["dataset"] in ["beers", "flights", "hospital", "tax",  "rayyan", "toy"]:
        data_dict = {
            "name": c["dataset"],
            "path": f"../datasets/{c['dataset']}/dirty.csv",
            "clean_path": f"../datasets/{c['dataset']}/clean.csv",
        }
    elif c["dataset"] in ["adult", "breast-cancer", "letter", "nursery"]:
        rate_formatted = int(c["error_fraction"] * 10)
        data_dict = {
            "name": c["dataset"],
            "path": f"../datasets/{c['dataset']}/MCAR/dirty_{rate_formatted}.csv",
            "clean_path": f"../datasets/{c['dataset']}/clean.csv",
        }
    else:
        raise ValueError("Unknown Dataset.")

    data = raha.Dataset(data_dict, n_rows=c["n_rows"])
    data.detected_cells = dict(data.get_actual_errors_dictionary())
    app = raha.Correction()
    app.LABELING_BUDGET = c["labeling_budget"]
    app.VERBOSE = False
    app.FEATURE_GENERATORS = c["feature_generators"]
    app.CLASSIFICATION_MODEL = c["classification_model"]
    app.VICINITY_ORDERS = c["vicinity_orders"]
    app.VICINITY_FEATURE_GENERATOR = c["vicinity_feature_generator"]
    app.N_BEST_PDEPS = c["n_best_pdeps"]
    app.USE_PDEP_FEATURE = c["use_pdep_feature"]

    d = app.initialize_dataset(data)
    app.initialize_models(d)
    while len(d.labeled_tuples) < app.LABELING_BUDGET:
        app.sample_tuple(d, random_seed=None)
        app.label_with_ground_truth(d)
        app.update_models(d)
        app.generate_features(d, synchronous=True)
        app.predict_corrections(d)

        p, r, f = d.get_data_cleaning_evaluation(d.corrected_cells)[-3:]
    return {"result": {"precision": p, "recall": r, "f1": f}, "config": c}


if __name__ == "__main__":
    rsk_renuver = Ruska(
        name="2022-08-07-evaluate-pdep-features-renuver-updated-all-features",
        description="""Untersuche, wie sich n_best_pdeps auf die Reinigung
        auswirkt.""",
        commit="",
        config={
            "dataset": "breast-cancer",
            "sampling": "MCAR",
            "error_fraction": 0.1,
            "labeling_budget": 20,
            "feature_generators": ["vicinity", "domain", "value"],
            "classification_model": "LOGR",
            "vicinity_orders": [1, 2],
            "vicinity_feature_generator": "pdep",
            "n_best_pdeps": 5,
            "n_rows": None,
            "use_pdep_feature": True,
        },
        ranges={
            "classification_model": ["LOGR", "ABC", "DTC"],
            "dataset": ["bridges", "cars", "glass", "restaurant"],
            "error_fraction": [0.01, 0.02, 0.03, 0.04, 0.05],
            "use_pdep_feature": [True, False],
        },
        runs=5,
        save_path="/root/measurements/",
    )

    rsk_baran = Ruska(
        name="2022-08-07-evaluate-pdep-features-baran-updated-all-features",
        description="""Untersuche, wie sich n_best_pdeps auf die Reinigung
        auswirkt.""",
        commit="",
        config={
            "dataset": "breast-cancer",
            "sampling": "MCAR",
            "error_fraction": 0.1,
            "labeling_budget": 20,
            "feature_generators": ["vicinity", "domain", "value"],
            "classification_model": "LOGR",
            "vicinity_orders": [1, 2],
            "vicinity_feature_generator": "pdep",
            "n_best_pdeps": 5,
            "n_rows": None,
            "use_pdep_feature": True,
        },
        ranges={
            "classification_model": ["LOGR", "ABC", "DTC"],
            "dataset": ["beers", "flights", "hospital", "rayyan"],
            "use_pdep_feature": [True, False],
        },
        runs=5,
        save_path="/root/measurements/",
    )

    rsk_renuver.run(experiment=run_baran, parallel=True)
    rsk_baran.run(experiment=run_baran, parallel=True)
