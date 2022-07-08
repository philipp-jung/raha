import raha
from ruska import Ruska

def run_baran_renuver(c: dict):
    """
    Use run in [1,5] to not get an error.
    """

    # an das format von renuver angepasst.
    rate_formatted = int(str(c['error_fraction']).split(".")[1])
    run = c['run'] + 1
    data_dict = {
        "name": c["dataset"],
        "path": f"../datasets/renuver/{c['dataset']}/{c['dataset']}_{rate_formatted}_{run}.csv",
        "clean_path": f"../datasets/renuver/{c['dataset']}/clean.csv",
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
    app.TRAINING_TIME_LIMIT = c['training_time_limit']
    app.EXCLUDE_VALUE_SPECIAL_CASE = c["exclude_value_special_case"]
    app.AG_PRESETS = c["ag_presets"]


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
    rsk = Ruska(
        name="2022-07-08-tune-autogluon",
        description="""Autogluon hat nicht den gewünschten Effekt. Ich erhoehe
        das time_limit und die presets, um doch noch gute Ergebnisse zu erhalten.""",
        commit="",
        config={
            "dataset": "breast-cancer",
            "sampling": "MCAR",
            "error_fraction": 0.1,
            "labeling_budget": 20,
            "feature_generators": ["value", "domain", "vicinity"],
            "classification_model": "AG",
            "vicinity_orders": [1, 2],
            "vicinity_feature_generator": "pdep",
            "n_best_pdeps": 5,
            "score_strategy": "multiply",
            "n_rows": None,
            "exclude_value_special_case": True,
            "training_time_limit": 45,
            "ag_presets": "good_quality_faster_inference_only_refit"
        },
        ranges={
            "dataset": ['bridges', 'cars', 'glass', 'restaurant'],
            "score_strategy": ['ensemble_new_feature', 'ensemble', 'multiply'],
            "error_fraction": [0.01, 0.02, 0.03, 0.04, 0.05]
        },
        runs=3,
        save_path="/root/measurements/",
    )

    rsk.run(experiment=run_baran_renuver, parallel=False)
