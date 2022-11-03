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
    app.RULE_BASED_VALUE_CLEANING = c['rule_based_value_cleaning']
    app.SYNTHESIZE_TRAIN_DATA = c['synthesize_train_data']

    d = app.initialize_dataset(data)
    app.initialize_models(d)
    # TODO make this use app.run()
    precisions, recalls, f1s = [], [], []
    while len(d.labeled_tuples) < app.LABELING_BUDGET:
        app.sample_tuple(d, random_seed=None)
        app.label_with_ground_truth(d)
        app.update_models(d)
        app.generate_features(d, synchronous=True)
        if app.RULE_BASED_VALUE_CLEANING:
            app.rule_based_value_cleaning(d)
        app.predict_corrections(d)
        if app.RULE_BASED_VALUE_CLEANING:
            # write the rule-based value corrections into the corrections dictionary. This overwrites
            # meta-learning result for domain & vicinity features. The idea is that the rule-based value
            # corrections are super precise and thus should be used if possible.
            for cell, correction in d.rule_based_value_corrections.items():
                row, col = cell
                if row not in d.labeled_tuples:  # don't overwrite user's corrections
                    d.corrected_cells[cell] = correction

        p, r, f = data.get_data_cleaning_evaluation(d.corrected_cells)[-3:]
        precisions.append(p), recalls.append(r), f1s.append(f)
    return {"result": {"precision": precisions, "recall": recalls, "f1": f1s}, "config": c}


if __name__ == "__main__":
    rsk_renuver = Ruska(
        name="2022-11-03-synthesize-train-data-renuver",
        description="See if synthesizing train data improves cleaning.",
        commit="",
        config={
            "dataset": "bridges",
            "sampling": "MCAR",
            "error_fraction": 0.01,
            "labeling_budget": 20,
            "feature_generators": ["domain", "vicinity", "value"],
            "classification_model": "CV",
            "vicinity_orders": [1, 2],
            "vicinity_feature_generator": "pdep",
            "n_rows": None,
            "n_best_pdeps": 3,
            "rule_based_value_cleaning": "V3",
            "synthesize_train_data": False,
        },
        ranges={
            "dataset": ["bridges", "cars", "glass", "restaurant"],
            "error_fraction": [0.01, 0.02, 0.03, 0.04, 0.05],
            "synthesize_train_data": [True, False],
        },
        runs=3,
        save_path="/root/measurements/",
    )

    rsk_baran = Ruska(
        name="2022-11-03-synthesize-train-data-baran",
        description="See if synthesizing train data improves cleaning.",
        commit="",
        config={
            "dataset": "breast-cancer",
            "sampling": "MCAR",
            "error_fraction": 0.1,
            "labeling_budget": 20,
            "feature_generators": ["domain", "vicinity", "value"],
            "classification_model": "CV",
            "vicinity_orders": [1, 2],
            "vicinity_feature_generator": "pdep",
            "n_rows": None,
            "n_best_pdeps": 3,
            "rule_based_value_cleaning": "V3",
            "synthesize_train_data": False,
        },
        ranges={
            "dataset": ["beers", "flights", "hospital", "rayyan"],
            "synthesize_train_data": [True, False],
            },
        runs=3,
        save_path="/root/measurements/",
    )

    rsk_renuver.run(experiment=run_baran, parallel=True)
    rsk_baran.run(experiment=run_baran, parallel=True)
