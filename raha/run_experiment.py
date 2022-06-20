import raha
from ruska import Ruska


def run_baran(c: dict):
    rate_formatted = str(c['error_fraction']).split(".")[1]
    data_dict = {
        "name": c["dataset"],
        "path": f"../datasets/{c['dataset']}/{c['sampling']}/dirty_{rate_formatted}.csv",
        "clean_path": f"../datasets/{c['dataset']}/clean.csv",
    }

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
    app.PDEP_SCORE_STRATEGY = c["score_strategy"]

    d = app.initialize_dataset(data)
    app.initialize_models(d)
    while len(d.labeled_tuples) < app.LABELING_BUDGET:
        app.sample_tuple(d, random_seed=None)
        app.label_with_ground_truth(d)
        app.update_models(d)
        app.generate_features(d, synchronous=True)
        app.predict_corrections(d, random_seed=None)

        p, r, f = d.get_data_cleaning_evaluation(d.corrected_cells)[-3:]
    return {"result": {"precision": p, "recall": r, "f1": f}, "config": c}


def run_baran_renuver(c: dict):
    rate_formatted = str(c['error_fraction']).split(".")[1]
    data_dict = {
        "name": c["dataset"],
        "path": f"../datasets/renuver/{c['dataset']}/{c['sampling']}/dirty_{rate_formatted}.csv",
        "clean_path": f"../datasets/{c['dataset']}/clean.csv",
    }

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
    app.PDEP_SCORE_STRATEGY = c["score_strategy"]

    d = app.initialize_dataset(data)
    app.initialize_models(d)
    while len(d.labeled_tuples) < app.LABELING_BUDGET:
        app.sample_tuple(d, random_seed=None)
        app.label_with_ground_truth(d)
        app.update_models(d)
        app.generate_features(d, synchronous=True)
        app.predict_corrections(d, random_seed=None)

        p, r, f = d.get_data_cleaning_evaluation(d.corrected_cells)[-3:]
    return {"result": {"precision": p, "recall": r, "f1": f}, "config": c}

if __name__ == "__main__":
    rsk = Ruska(
        name="2022-06-10-validate-simple-mcar",
        description="""Wiederholung aller Messungen auf den Datensätzen
                 von denen ich weiß, dass pdep besser ist als Baran. Die Fehler
                 erzeuge ich mit Ruska's simple-mcar""",
        commit="füge ich später ein :)",
        config={
            "dataset": "breast-cancer",
            "sampling": "MCAR",
            "error_fraction": 0.1,
            "labeling_budget": 20,
            "feature_generators": ["value", "domain", "vicinity"],
            "classification_model": "ABC",
            "vicinity_orders": [1, 2],
            "vicinity_feature_generator": "pdep",
            "n_best_pdeps": 5,
            "score_strategy": "multiply",
            "n_rows": None,
        },
        ranges={
            "error_fraction": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
            "dataset": ["breast-cancer", "letter"],
        },
        runs=3,
        save_path="/root/measurements/",
    )

    rsk.run(experiment=run_baran, parallel=True)

    rsk_naive = Ruska(
        name="2022-06-10-validate-simple-mcar-naive",
        description="""Wiederholung aller Messungen auf den Datensätzen
             von denen ich weiß, dass pdep besser ist als Baran. Die Fehler
             erzeuge ich mit Ruska's simple-mcar""",
        commit="füge ich später ein :)",
        config={
            "dataset": "breast-cancer",
            "sampling": "MCAR",
            "error_fraction": 0.1,
            "labeling_budget": 20,
            "feature_generators": ["value", "domain", "vicinity"],
            "classification_model": "ABC",
            "vicinity_orders": [1],
            "vicinity_feature_generator": "naive",
            "n_best_pdeps": 5,
            "score_strategy": "multiply",
            "n_rows": None,
        },
        ranges={
            "error_fraction": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
            "dataset": ["breast-cancer", "letter"],
        },
        runs=3,
        save_path="/root/measurements/",
    )

    rsk_naive.run(experiment=run_baran, parallel=True)
