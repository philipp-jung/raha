########################################
# Baran: The Error Correction System
# Mohammad Mahdavi
# moh.mahdavi.l@gmail.com
# April 2019
# Big Data Management Group
# TU Berlin
# All Rights Reserved
########################################


########################################
import os
import io
import sys
import math
import json
import pickle
import random
import difflib
import unicodedata
import multiprocessing
from collections import Counter
import json

import numpy as np
import sklearn.svm
import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.linear_model
import sklearn.tree
from typing import Dict, List, Union
import pandas as pd
from sentence_transformers import SentenceTransformer, util

import raha
from raha import imputer
from raha import pdep
from raha import hpo
from raha import helpers
from raha import ml_helpers
from raha import value_cleaning
from raha import value_helpers


########################################


########################################

class Correction:
    """
    The main class.
    """

    def __init__(self, labeling_budget: int, classification_model: str, clean_with_user_input: bool, feature_generators: List[str],
                 vicinity_orders: List[int], vicinity_feature_generator: str, imputer_cache_model: bool,
                 n_best_pdeps: int, training_time_limit: int, rule_based_value_cleaning: Union[str, bool],
                 synth_tuples: int, synth_tuples_error_threshold: int):
        """
        Parameters of the cleaning experiment.
        @param labeling_budget: How many tuples are labeled by the user. Baran default is 20.
        @param classification_model: "ABC" for sklearn's AdaBoostClassifier with n_estimators=100, "CV" for
        cross-validation. Baran default is "ABC".
        @param clean_with_user_input: Take user input to clean data with. This will always improve cleaning performance,
        and is recommended to set to True as the default value. Handy to disable when debugging models.
        @param feature_generators: Four feature generators are available: 'imputer', 'domain', 'vicinity' and 'value'.
        Pass them as strings in a list to make Baran use them, e.g. ['domain', 'vicinity', 'imputer']. The Baran
        default is ['domain', 'vicinity', 'value'].
        @param vicinity_orders: The pdep approach enables the usage of higher-order dependencies to clean data. Each
        order that Baran shall use is passed as an integer, e.g. [1, 2]. The Baran default is [1].
        @param vicinity_feature_generator: How vicinity features are generated. Either 'pdep' or 'naive'. The Baran
        default is 'naive'.
        @param imputer_cache_model: Whether or not the AutoGluon model used in the imputer model is stored after
        training and used from cache. Otherwise, a model is retrained with every user-provided tuple. If true, reduces
        cleaning time significantly.
        @param n_best_pdeps: When using vicinity_feature_generator = 'pdep', dependencies are ranked via the pdep
        measure. After ranking, the n_best_pdeps dependencies are used to provide cleaning suggestions. A good heuristic
        is to set this to 3.
        @param training_time_limit: Limit in seconds of how long the AutoGluon imputer model is trained.
        @param rule_based_value_cleaning: Use rule based cleaning approach if 'V1' or 'V3'. Set to False to have value
        cleaning be part of the meta learning, which is the Baran default. V1 uses rules from 2022W38, V3 from 2022W40.
        @param synth_tuples: maximum number of tuples to synthesize training data with.
        @param synth_tuples_error_threshold: maximum number of errors in a row that is used to synthesize tuples.
        """
        # Philipps changes
        self.SYNTH_TUPLES = synth_tuples
        self.SYNTH_TUPLES_ERROR_THRESHOLD = synth_tuples_error_threshold
        self.CLEAN_WITH_USER_INPUT = clean_with_user_input

        self.FEATURE_GENERATORS = feature_generators
        self.VICINITY_ORDERS = vicinity_orders
        self.VICINITY_FEATURE_GENERATOR = vicinity_feature_generator
        self.IMPUTER_CACHE_MODEL = imputer_cache_model
        self.N_BEST_PDEPS = n_best_pdeps
        self.TRAINING_TIME_LIMIT = training_time_limit
        self.RULE_BASED_VALUE_CLEANING = rule_based_value_cleaning
        self.CLASSIFICATION_MODEL = classification_model

        # original Baran
        self.PRETRAINED_VALUE_BASED_MODELS_PATH = ""
        self.VALUE_ENCODINGS = ["identity", "unicode"]
        self.IGNORE_SIGN = "<<<IGNORE_THIS_VALUE>>>"
        self.VERBOSE = False
        self.SAVE_RESULTS = False
        self.ONLINE_PHASE = False
        self.LABELING_BUDGET = labeling_budget
        self.MIN_CORRECTION_CANDIDATE_PROBABILITY = 0.00
        self.MIN_CORRECTION_OCCURRENCE = 2
        self.MAX_VALUE_LENGTH = 50
        self.REVISION_WINDOW_SIZE = 5

        # variable for debugging CV
        self.n_true_classes = {}
        self.sampled_tuples = 0

    @staticmethod
    def _value_encoder(value, encoding):
        """
        This method represents a value with a specified value abstraction encoding method.
        """
        if encoding == "identity":
            return json.dumps(list(value))
        if encoding == "unicode":
            return json.dumps([unicodedata.category(c) for c in value])

    @staticmethod
    def _to_model_adder(model, key, value):
        """
        This method incrementally adds a key-value into a dictionary-implemented model.
        """
        if key not in model:
            model[key] = {}
        if value not in model[key]:
            model[key][value] = 0.0
        model[key][value] += 1.0

    @staticmethod
    def _to_value_model_adder(value_model, enc_error_value, transformation, error_cell):
        """
        This method updates the value_model. It is different from _to_model_adder, since value_model contains
        coordinates of the error_cell from which the transformation originates.
        """
        if enc_error_value not in value_model:
            value_model[enc_error_value] = {}
        if transformation not in value_model[enc_error_value]:
            value_model[enc_error_value][transformation] = []
        value_model[enc_error_value][transformation].append(error_cell)

    def _value_based_models_updater(self, models, ud, cell):
        """
        This method updates the value-based error corrector models with a given update dictionary.
        """
        if (ud["new_value"] and len(ud["new_value"]) <= self.MAX_VALUE_LENGTH and
                ud["old_value"] and len(ud["old_value"]) <= self.MAX_VALUE_LENGTH and
                ud["old_value"] != ud["new_value"] and ud["old_value"].lower() != "n/a"):
            models.update_rules(ud['old_value'], ud['new_value'], cell)

    def _domain_based_model_updater(self, model, ud):
        """
        This method updates the domain-based error corrector model with a given update dictionary.
        """
        self._to_model_adder(model, ud["column"], ud["new_value"])

    def _imputer_based_corrector(self, model: Dict[int, pd.DataFrame], ed: dict) -> list:
        """
        Use class probabilities generated by an Datawig model to make corrections.

        @param model: Dictionary with an AutoGluonImputer per column. Will an empty dict if no models have been trained.
        @param ed: Error Dictionary with information on the error and its vicinity.
        @return: list of corrections.
        """
        df_probas = model.get(ed['column'])
        if df_probas is None:
            return []
        probas = df_probas.iloc[ed['row']]

        prob_d = {key: probas.to_dict()[key] for key in probas.to_dict()}
        prob_d_sorted = {key: value for key, value in sorted(prob_d.items(), key=lambda x: x[1])}

        result = {}
        for correction, probability in prob_d_sorted.items():
            # make sure that suggested correction is likely and isn't the error.
            if probability > 0 and correction != ed['old_value']:
                result[correction] = probability
        return [result]

    def _domain_based_corrector(self, model, ed):
        """
        This method takes a domain-based model and an error dictionary to generate potential domain-based corrections.
        """
        results_dictionary = {}
        value_counts = model.get(ed["column"])
        if value_counts is not None:
            sum_scores = sum(model[ed["column"]].values())
            for new_value in model[ed["column"]]:
                pr = model[ed["column"]][new_value] / sum_scores
                results_dictionary[new_value] = pr
        return [results_dictionary]

    def initialize_dataset(self, d):
        """
        This method initializes the dataset.
        """
        self.ONLINE_PHASE = True
        d.results_folder = os.path.join(os.path.dirname(d.path), "raha-baran-results-" + d.name)
        if self.SAVE_RESULTS and not os.path.exists(d.results_folder):
            os.mkdir(d.results_folder)
        d.labeled_tuples = {} if not hasattr(d, "labeled_tuples") else d.labeled_tuples
        d.labeled_cells = {} if not hasattr(d, "labeled_cells") else d.labeled_cells
        d.corrected_cells = {} if not hasattr(d, "corrected_cells") else d.corrected_cells
        return d

    def initialize_models(self, d):
        """
        This method initializes the error corrector models.
        """
        d.value_models = value_cleaning.ValueCleaning()
        d.pdeps = {c: {cc: {} for cc in range(d.dataframe.shape[1])}
                   for c in range(d.dataframe.shape[1])}

        d.domain_models = {}

        for row in d.dataframe.itertuples():
            i, row = row[0], row[1:]
            # Das ist richtig cool: Jeder Wert des Tupels wird untersucht und
            # es wird überprüft, ob dieser Wert ein aus Error Detection bekannter
            # Fehler ist. Wenn dem so ist, wird der Wert durch das IGNORE_SIGN
            # ersetzt.
            vicinity_list = [cv if (i, cj) not in d.detected_cells else self.IGNORE_SIGN for cj, cv in enumerate(row)]
            for j, value in enumerate(row):
                # if rhs_value's position is not a known error
                if (i, j) not in d.detected_cells:
                    temp_vicinity_list = list(vicinity_list)
                    temp_vicinity_list[j] = self.IGNORE_SIGN
                    update_dictionary = {
                        "column": j,
                        "new_value": value,
                        "vicinity": temp_vicinity_list
                    }
                    self._domain_based_model_updater(d.domain_models, update_dictionary)

        # BEGIN Philipp's changes
        d.value_corrections = {}
        d.sbert_cos_sim = {}

        # for debugging purposes only
        d.domain_corrections = {}
        d.naive_vicinity_corrections = {}
        d.pdep_vicinity_corrections = {}
        d.imputer_corrections = {}

        d.vicinity_models = {}
        if 'vicinity' in self.FEATURE_GENERATORS:
            for o in self.VICINITY_ORDERS:
                d.vicinity_models[o] = pdep.calculate_counts_dict(
                    df=d.dataframe,
                    detected_cells=d.detected_cells,
                    order=o,
                    ignore_sign=self.IGNORE_SIGN)
        d.imputer_models = {}

        if self.VERBOSE:
            print("The error corrector models are initialized.")

    def sample_tuple(self, d, random_seed):
        """
        This method samples a tuple.
        Philipp extended this with a random_seed in an effort to make runs reproducible.

        I also added two tiers of columns to choose samples from. Either, there are error cells
        for which no correction suggestion has been made yet. In that case, we sample from these error cells.
        If correction suggestions have been made for all cells, we sample from error cells that have not been
        sampled before.
        """
        rng = np.random.default_rng(seed=random_seed)
        remaining_column_unlabeled_cells = {}
        remaining_column_unlabeled_cells_error_values = {}
        remaining_column_uncorrected_cells = {}
        remaining_column_uncorrected_cells_error_values = {}

        error_positions = helpers.ErrorPositions(d.detected_cells, d.dataframe.shape, d.corrected_cells)

        column_errors = error_positions.original_column_errors

        for j in column_errors:
            for error_cell in column_errors[j]:
                if error_cell not in d.corrected_cells:  # no correction suggestion has been found yet.
                    self._to_model_adder(remaining_column_uncorrected_cells, j, error_cell)
                    self._to_model_adder(remaining_column_uncorrected_cells_error_values, j,
                                         d.dataframe.iloc[error_cell])
                if error_cell not in d.labeled_cells:
                    # the cell has not been labeled by the user yet. this is stricter than the above condition.
                    self._to_model_adder(remaining_column_unlabeled_cells, j, error_cell)
                    self._to_model_adder(remaining_column_unlabeled_cells_error_values, j, d.dataframe.iloc[error_cell])
        tuple_score = np.ones(d.dataframe.shape[0])
        tuple_score[list(d.labeled_tuples.keys())] = 0.0

        if len(remaining_column_uncorrected_cells) > 0:
            remaining_columns_to_choose_from = remaining_column_uncorrected_cells
            remaining_cell_error_values = remaining_column_uncorrected_cells_error_values
        else:
            remaining_columns_to_choose_from = remaining_column_unlabeled_cells
            remaining_cell_error_values = remaining_column_unlabeled_cells_error_values

        for j in remaining_columns_to_choose_from:
            for cell in remaining_columns_to_choose_from[j]:
                value = d.dataframe.iloc[cell]
                column_score = math.exp(len(remaining_columns_to_choose_from[j]) / len(column_errors[j]))
                cell_score = math.exp(remaining_cell_error_values[j][value] / len(remaining_columns_to_choose_from[j]))
                tuple_score[cell[0]] *= column_score * cell_score

        # Nützlich, um tuple-sampling zu debuggen: Zeigt die Tupel, aus denen
        # zufällig gewählt wird.
        # print(np.argwhere(tuple_score == np.amax(tuple_score)).flatten())
        d.sampled_tuple = rng.choice(np.argwhere(tuple_score == np.amax(tuple_score)).flatten())
        if self.VERBOSE:
            print("Tuple {} is sampled.".format(d.sampled_tuple))
        self.sampled_tuples += 1

    def label_with_ground_truth(self, d):
        """
        This method labels a tuple with ground truth.
        Takes the sampled row from d.sampled_tuple, iterates over each cell
        in that row taken from the clean data, and then adds
        d.labeled_cells[(row, col)] = [is_error, clean_value_from_clean_dataframe]
        to d.labeled_cells.
        """
        d.labeled_tuples[d.sampled_tuple] = 1
        for col in range(d.dataframe.shape[1]):
            cell = (d.sampled_tuple, col)
            error_label = 0
            if d.dataframe.iloc[cell] != d.clean_dataframe.iloc[cell]:
                error_label = 1
            d.labeled_cells[cell] = [error_label, d.clean_dataframe.iloc[cell]]
        if self.VERBOSE:
            print("Tuple {} is labeled.".format(d.sampled_tuple))

    def update_models(self, d):
        """
        This method updates the error corrector models with a new labeled tuple.
        """
        cleaned_sampled_tuple = []
        for column in range(d.dataframe.shape[1]):
            clean_cell = d.labeled_cells[(d.sampled_tuple, column)][1]
            cleaned_sampled_tuple.append(clean_cell)

        for column in range(d.dataframe.shape[1]):
            cell = (d.sampled_tuple, column)
            update_dictionary = {
                "column": column,
                "old_value": d.dataframe.iloc[cell],
                "new_value": cleaned_sampled_tuple[column],
            }

            # if the value in that cell has been labelled an error
            if d.labeled_cells[cell][0] == 1:
                # update value and vicinity models.
                self._value_based_models_updater(d.value_models, update_dictionary, cell)
                self._domain_based_model_updater(d.domain_models, update_dictionary)
                update_dictionary["vicinity"] = [cv if column != cj else self.IGNORE_SIGN
                                                 for cj, cv in enumerate(cleaned_sampled_tuple)]

                # if the cell hadn't been detected as an error
                if cell not in d.detected_cells:
                    # add that cell to detected_cells and assign it IGNORE_SIGN
                    # --> das passiert, wenn die Error Detection nicht perfekt
                    # war, dass man einen Fehler labelt, der vorher noch nicht
                    # gelabelt war.
                    d.detected_cells[cell] = self.IGNORE_SIGN

            else:
                update_dictionary["vicinity"] = [cv if column != cj and d.labeled_cells[(d.sampled_tuple, cj)][0] == 1
                                                 else self.IGNORE_SIGN for cj, cv in enumerate(cleaned_sampled_tuple)]

        # BEGIN Philipp's changes
        if 'vicinity' in self.FEATURE_GENERATORS:
            for o in self.VICINITY_ORDERS:
                pdep.update_counts_dict(d.dataframe,
                                        d.vicinity_models[o],
                                        o,
                                        cleaned_sampled_tuple)

        # Todo update sbert_models here instead of recalculating the whole thing every time a tuple is sampled.
        # END Philipp's changes

        if self.VERBOSE:
            print("The error corrector models are updated with new labeled tuple {}.".format(d.sampled_tuple))

    def _feature_generator_process(self, args):
        """
        This method generates cleaning suggestions for one error in one cell. The suggestion
        gets turned into features for the classifier in predict_corrections(). It gets called
        once for each each error cell.

        Depending on the value of `synchronous` in `generate_features()`, the method will
        be executed in parallel or not.
        """
        d, cell, is_synth = args

        # vicinity ist die Zeile, column ist die Zeilennummer, old_value ist der Fehler
        error_dictionary = {"column": cell[1],
                            "old_value": d.dataframe.iloc[cell],
                            "vicinity": list(d.dataframe.iloc[cell[0], :]),
                            "row": cell[0]}
        naive_vicinity_corrections = []
        pdep_vicinity_corrections = []
        value_corrections = []
        domain_corrections = []
        imputer_corrections = []
        sbert_corrections = []

        # Begin Philipps Changes
        if "vicinity" in self.FEATURE_GENERATORS:
            if self.VICINITY_FEATURE_GENERATOR == 'naive':
                for o in self.VICINITY_ORDERS:
                    naive_corrections = pdep.vicinity_based_corrector_order_n(
                        counts_dict=d.vicinity_models[o],
                        ed=error_dictionary,
                        probability_threshold=self.MIN_CORRECTION_CANDIDATE_PROBABILITY)
                    naive_vicinity_corrections.append(naive_corrections)

            elif self.VICINITY_FEATURE_GENERATOR == 'pdep':
                for o in self.VICINITY_ORDERS:
                    pdep_corrections = pdep.pdep_vicinity_based_corrector(
                        inverse_sorted_gpdeps=d.inv_vicinity_gpdeps[o],
                        counts_dict=d.vicinity_models[o],
                        ed=error_dictionary,
                        n_best_pdeps=self.N_BEST_PDEPS)
                    pdep_vicinity_corrections.append(pdep_corrections)
            else:
                raise ValueError(f'Unknown VICINITY_FEATURE_GENERATOR '
                                 f'{self.VICINITY_FEATURE_GENERATOR}')

        if "value" in self.FEATURE_GENERATORS and not is_synth:
            error = error_dictionary['old_value']
            value_corrections = d.value_models.cleaning_features(error, 'both')
            if not is_synth:
                d.value_corrections[cell] = value_corrections

        # TODO implement sbert for synth
        if "sbert" in self.FEATURE_GENERATORS and not is_synth:
            row, column = error_dictionary['row'], error_dictionary['column']
            sbert_corrections = d.sbert_cos_sim[(row, column)]

        if "domain" in self.FEATURE_GENERATORS:
            domain_corrections = self._domain_based_corrector(d.domain_models, error_dictionary)
        if "imputer" in self.FEATURE_GENERATORS:
            imputer_corrections = self._imputer_based_corrector(d.imputer_models, error_dictionary)

            # Sometimes training an imputer model fails for a column, while it succeeds on other columns.
            # If training failed for a column, imputer_corrections will be an empty list. Which will lead
            # to one less feature being added to values in that column. Which in turn is bad news.
            # To prevent this, I have imputer_corrections refer to [{}], which has length 1 and will create
            # a feature with value 0.
            if len(d.labeled_tuples) == self.LABELING_BUDGET and len(imputer_corrections) == 0:
                imputer_corrections = [{}]

        # below block is for debugging purposes only.
        d.domain_corrections[cell] = domain_corrections
        d.naive_vicinity_corrections[cell] = naive_vicinity_corrections
        d.pdep_vicinity_corrections[cell] = pdep_vicinity_corrections
        d.imputer_corrections[cell] = imputer_corrections

        if not self.RULE_BASED_VALUE_CLEANING and not is_synth:
            # construct value corrections as in original Baran
            # TODO I think this is broken rn. I changed the data structure value_corrections refers to.
            value_corrections = [{correction: d[correction]['encoded_string_frequency']} for d in value_corrections for
                                 correction in d]
            models_corrections = value_corrections \
                                 + domain_corrections \
                                 + [corrections for order in naive_vicinity_corrections for corrections in order] \
                                 + [corrections for order in pdep_vicinity_corrections for corrections in order] \
                                 + imputer_corrections
        elif not self.RULE_BASED_VALUE_CLEANING and is_synth:
            raise ValueError('It is impossible to synthesize tuples and use the meta-learner to apply value-corrections'
                             ' at the same time. Instead, perform RULE_BASED_VALUE_CLEANING, or set n_synth_tuples=0.')
        else:  # do rule based value cleaning.
            models_corrections = domain_corrections \
                                 + [corrections for order in naive_vicinity_corrections for corrections in order] \
                                 + [corrections for order in pdep_vicinity_corrections for corrections in order] \
                                 + imputer_corrections \
                                 + [sbert_corrections]
        # End Philipps Changes

        corrections_features = {}
        for mi, model in enumerate(models_corrections):
            for correction in model:
                if correction not in corrections_features:
                    corrections_features[correction] = np.zeros(len(models_corrections))
                corrections_features[correction][mi] = model[correction]
        return corrections_features

    def prepare_augmented_models(self, d):
        """
        Prepare augmented models that Philipp added:
        1) Calculate gpdeps and append them to d.
        2) Train imputer model for each column.
        """
        if self.VICINITY_FEATURE_GENERATOR == 'pdep' and 'vicinity' in self.FEATURE_GENERATORS:
            d.inv_vicinity_gpdeps = {}
            for order in self.VICINITY_ORDERS:
                vicinity_gpdeps = pdep.calc_all_gpdeps(d.vicinity_models,
                                                       d.dataframe, d.detected_cells, order)
                d.inv_vicinity_gpdeps[order] = pdep.invert_and_sort_gpdeps(vicinity_gpdeps)

        if 'imputer' in self.FEATURE_GENERATORS and len(d.labeled_tuples) == self.LABELING_BUDGET:
            # simulate user input by reading labeled data from the typed dataframe
            inputted_rows = list(d.labeled_tuples.keys())
            typed_user_input = d.typed_clean_dataframe.iloc[inputted_rows, :]
            df_clean_subset = imputer.get_clean_table(d.typed_dataframe, d.detected_cells, typed_user_input)
            for i_col, col in enumerate(df_clean_subset.columns):
                imp = imputer.train_cleaning_model(df_clean_subset,
                                                   d.name,
                                                   label=i_col,
                                                   time_limit=self.TRAINING_TIME_LIMIT,
                                                   use_cache=self.IMPUTER_CACHE_MODEL)
                if imp is not None:
                    d.imputer_models[i_col] = imp.predict_proba(d.typed_dataframe)
                else:
                    d.imputer_models[i_col] = None

        if 'sbert' in self.FEATURE_GENERATORS:
            sbert_model = SentenceTransformer('all-MiniLM-L6-v2')  # keep model loaded
            error_positions = helpers.ErrorPositions(d.detected_cells, d.dataframe.shape, d.corrected_cells)
            column_errors = error_positions.original_column_errors
            values_in_cols = {k: [val for val, _ in v.items()] for k, v in d.domain_models.items()}
            d.sbert_models = {}
            for col in column_errors:
                emb_values = sbert_model.encode(values_in_cols[col])
                for error_cell in column_errors[col]:
                    error_value = error_positions.detected_cells[error_cell]
                    emb_error = sbert_model.encode(error_value)
                    d.sbert_cos_sim[error_cell] = {values_in_cols[col][i]: float(util.cos_sim(emb_error, emb_values[i])) for i, _ in enumerate(values_in_cols[col])}


    def generate_features(self, d, synchronous):
        """
        This method generates a feature vector for each pair of a data error
        and a potential correction.
        Philipp added a `synchronous` parameter to make debugging easier.
        """

        d.pair_features = {}
        d.synth_pair_features = {}
        pairs_counter = 0
        process_args_list = [[d, cell, False] for cell in d.detected_cells]
        if not synchronous:
            pool = multiprocessing.Pool()
            feature_generation_results = pool.map(self._feature_generator_process, process_args_list)
            pool.close()
        else:
            feature_generation_results = []
            for args in process_args_list:
                result = self._feature_generator_process(args)
                feature_generation_results.append(result)

        for ci, corrections_features in enumerate(feature_generation_results):
            cell = process_args_list[ci][1]
            d.pair_features[cell] = {}
            for correction in corrections_features:
                d.pair_features[cell][correction] = corrections_features[correction]
                pairs_counter += 1

        if self.VERBOSE:
            print("{} pairs of (a data error, a potential correction) are featurized.".format(pairs_counter))

        error_positions = helpers.ErrorPositions(d.detected_cells, d.dataframe.shape, d.corrected_cells)
        row_errors = error_positions.updated_row_errors

        # determine rows with least amount of erronous values to sample from.
        candidate_rows = [(row, len(cells)) for row, cells in row_errors.items() if
                                   len(cells) <= self.SYNTH_TUPLES_ERROR_THRESHOLD]
        ranked_candidate_rows = sorted(candidate_rows, key=lambda x: x[1])
        if self.SYNTH_TUPLES > 0 and len(ranked_candidate_rows) > 0:
            # sample randomly to prevent sorted data to mess with synth tuple sampling.
            random.seed(0)
            if len(ranked_candidate_rows) <= self.SYNTH_TUPLES:
                synthetic_error_rows = random.sample([x[0] for x in ranked_candidate_rows], len(ranked_candidate_rows))
            else:
                synthetic_error_rows = random.sample([x[0] for x in ranked_candidate_rows[:self.SYNTH_TUPLES]], self.SYNTH_TUPLES)
            synthetic_error_cells = [(i, j) for i in synthetic_error_rows for j in range(d.dataframe.shape[1])]
            synth_args_list = [[d, cell, True] for cell in synthetic_error_cells]

            if not synchronous:
                pool = multiprocessing.Pool()
                synth_feature_generation_results = pool.map(self._feature_generator_process, synth_args_list)
                pool.close()
            else:
                synth_feature_generation_results = []
                for args in synth_args_list:
                    result = self._feature_generator_process(args)
                    synth_feature_generation_results.append(result)

            synth_pairs_counter = 0
            for ci, corrections_features in enumerate(synth_feature_generation_results):
                cell = synth_args_list[ci][1]
                d.synth_pair_features[cell] = {}
                for correction in corrections_features:
                    d.synth_pair_features[cell][correction] = corrections_features[correction]
                    synth_pairs_counter += 1

            if self.VERBOSE:
                print(f"{synth_pairs_counter} pairs of synthetic (error, potential correction) are featurized.")

    def rule_based_value_cleaning(self, d):
        """ Find value corrections with a conditional probability of 1.0 and use them as corrections."""
        d.rule_based_value_corrections = {}

        for cell, value_corrections in d.value_corrections.items():
            rule_based_suggestion = None
            if self.RULE_BASED_VALUE_CLEANING == 'V4' and len(value_corrections) > 0:
                atomic_corrections, complex_corrections = value_corrections[:8], value_corrections[8]
                value_suggestions = value_helpers.BothValueSuggestions(cell, atomic_corrections, complex_corrections)
                rule_based_suggestion = value_suggestions.rule_based_suggestion_v4()

            if self.RULE_BASED_VALUE_CLEANING == 'V5' and len(value_corrections) > 0:
                atomic_corrections, complex_corrections = value_corrections[:8], value_corrections[8]
                value_suggestions = value_helpers.BothValueSuggestions(cell, atomic_corrections, complex_corrections)
                rule_based_suggestion = value_suggestions.rule_based_suggestion_v4()

            if rule_based_suggestion is not None:
                d.rule_based_value_corrections[cell] = rule_based_suggestion

    def multi_predict_corrections(self, d):
        """
        In an effort to improve cleaning performance and adapt to non-categorical problems, we model the selection
        of cleaning suggestions as a multi-category classification task.
        """
        error_positions = helpers.ErrorPositions(d.detected_cells, d.dataframe.shape, d.corrected_cells)
        column_errors = error_positions.original_column_errors
        for col in column_errors:
            x_train, y_train, x_test, user_corrected_cells = ml_helpers.multi_generate_train_test_data(column_errors,
                                                                                                       d.labeled_cells,
                                                                                                       d.pair_features,
                                                                                                       d.synth_pair_features,
                                                                                                       d.dataframe,
                                                                                                       col)
            for error_cell in user_corrected_cells:
                d.corrected_cells[error_cell] = user_corrected_cells[error_cell]

            if len(x_train) > 0 and len(x_test) > 0:
                # TODO refactor so that AG blends in better.
                if self.CLASSIFICATION_MODEL == "AG":
                    gs_clf = hpo.ag_predictor(np.ndarray(x_train), y_train, self.TRAINING_TIME_LIMIT)
                    if len(set(y_train)) == 1:  # only one class in the training data, so cast all results to that class.
                        predicted_labels = [y_train[0] for _ in range(len(x_test))]
                    elif len(set(y_train)) > 1:
                        df_test = pd.DataFrame(x_test)
                        predicted_labels = gs_clf.predict(df_test)
                else:
                    if self.CLASSIFICATION_MODEL == "ABC":
                        gs_clf = sklearn.ensemble.AdaBoostClassifier(n_estimators=100)
                    elif self.CLASSIFICATION_MODEL == "CV":
                        gs_clf = hpo.cross_validated_estimator(x_train, y_train)
                    elif self.CLASSIFICATION_MODEL == "DTC":
                        gs_clf = sklearn.tree.DecisionTreeClassifier()
                    elif self.CLASSIFICATION_MODEL == "LOGR":
                        gs_clf = sklearn.linear_model.LogisticRegression(multi_class="multinomial")
                    else:
                        raise ValueError('Unknown model.')

                    if len(set(y_train)) == 1:  # only one class in the training data, so cast all results to that class.
                        predicted_labels = [y_train[0] for _ in range(len(x_test))]
                    elif len(set(y_train)) > 1:
                        gs_clf.fit(x_train, y_train)
                        predicted_labels = gs_clf.predict(x_test)

                # set final cleaning suggestion from meta-learning result. User corrected cells are not overwritten!
                for cell, label in zip(column_errors[col], predicted_labels):
                    d.corrected_cells[cell] = label

            elif len(d.labeled_tuples) == 0:  # no training data is available because no user labels have been set.
                for cell in d.pair_features:
                    correction_dict = d.pair_features[cell]
                    if len(correction_dict) > 0:
                        # select the correction with the highest sum of features.
                        max_proba_feature = \
                            sorted([v for v in correction_dict.items()], key=lambda x: sum(x[1]), reverse=True)[0]
                        d.corrected_cells[cell] = max_proba_feature[0]

            elif len(x_train) > 0 and len(x_test) == 0:  # len(x_test) == 0 because all rows have been labeled.
                pass  # nothing to do here -- just use the manually corrected cells to correct all errors.

            elif len(x_train) == 0:
                pass  # nothing to learn because x_train is empty.
            else:
                raise ValueError('Invalid state')

        if self.VERBOSE:
            print("{:.0f}% ({} / {}) of data errors are corrected.".format(
                100 * len(d.corrected_cells) / len(d.detected_cells),
                len(d.corrected_cells), len(d.detected_cells)))

    def binary_predict_corrections(self, d):
        """
        The ML problem as formulated in the Baran paper.
        """
        error_positions = helpers.ErrorPositions(d.detected_cells, d.dataframe.shape, d.corrected_cells)
        column_errors = error_positions.original_column_errors
        for j in column_errors:
            x_train, y_train, x_test, user_corrected_cells, all_error_correction_suggestions = ml_helpers.generate_train_test_data(
                column_errors,
                d.labeled_cells,
                d.pair_features,
                d.dataframe,
                d.synth_pair_features,
                j)

            is_valid_problem, predicted_labels = ml_helpers.handle_edge_cases(x_train, x_test, y_train, d)
            if is_valid_problem:
                # TODO refactor so that AG blends in better.
                if self.CLASSIFICATION_MODEL == "AG":
                    x_train = np.stack(x_train)
                    x_test = np.stack(x_test)
                    gs_clf = hpo.ag_predictor(x_train, y_train, self.TRAINING_TIME_LIMIT)
                    if len(set(y_train)) == 1:  # only one class in the training data, so cast all results to that class.
                        predicted_labels = [y_train[0] for _ in range(len(x_test))]
                    elif len(set(y_train)) > 1:
                        df_test = pd.DataFrame(x_test)
                        predicted_labels = gs_clf.predict(df_test)
                else:
                    if self.CLASSIFICATION_MODEL == "ABC" or sum(y_train) <= 2:
                        gs_clf = sklearn.ensemble.AdaBoostClassifier(n_estimators=100)
                        gs_clf.fit(x_train, y_train)
                    elif self.CLASSIFICATION_MODEL == "CV":
                        gs_clf = hpo.cross_validated_estimator(x_train, y_train)
                    else:
                        raise ValueError('Unknown model.')
                    predicted_labels = gs_clf.predict(x_test)

            ml_helpers.set_binary_cleaning_suggestions(predicted_labels, all_error_correction_suggestions,
                                                       user_corrected_cells, d)

        if self.VERBOSE:
            print("{:.0f}% ({} / {}) of data errors are corrected.".format(
                100 * len(d.corrected_cells) / len(d.detected_cells),
                len(d.corrected_cells), len(d.detected_cells)))

    def voting_binary_predict_corrections(self, d):
        """
        Train two binary classifiers. One with synth_tuples, the other without. Combine them via a voting ensemble
        that sums up the class probabilities of both models.
        """
        error_positions = helpers.ErrorPositions(d.detected_cells, d.dataframe.shape, d.corrected_cells)
        column_errors = error_positions.original_column_errors
        for j in column_errors:
            classifiers = []
            train_test_data = [
                ml_helpers.generate_train_test_data(column_errors, d.labeled_cells, d.pair_features, d.dataframe, synth,
                                                    j) for synth in ([], d.synth_pair_features)]

            valid_problems = 0
            for (x_train, y_train, x_test, user_corrected_cells, all_error_correction_suggestions) in train_test_data:
                is_valid_problem, predicted_labels = ml_helpers.handle_edge_cases(x_train, x_test, y_train, d)
                if is_valid_problem:
                    valid_problems += 1
                    clf = sklearn.ensemble.AdaBoostClassifier(n_estimators=100)
                    clf.fit(x_train, y_train)
                    classifiers.append(clf)

            if valid_problems == 2:  # both problems need to be valid
                est = [('no synth', classifiers[0]), ['synth', classifiers[1]]]
                ensemble = ml_helpers.VotingClassifier(est)
                predicted_labels = ensemble.predict(np.stack(x_test, axis=0))

            ml_helpers.set_binary_cleaning_suggestions(predicted_labels, all_error_correction_suggestions,
                                                       user_corrected_cells, d)

        if self.VERBOSE:
            print("{:.0f}% ({} / {}) of data errors are corrected.".format(
                100 * len(d.corrected_cells) / len(d.detected_cells),
                len(d.corrected_cells), len(d.detected_cells)))

    def clean_with_user_input(self, d):
        """
        The user input ideally contains completely correct data. It should be leveraged for an ideal cleaning
        performance.
        """
        if not self.CLEAN_WITH_USER_INPUT:
            return None
        for error_cell in d.detected_cells:
            if error_cell in d.labeled_cells:
                d.corrected_cells[error_cell] = d.labeled_cells[error_cell][1]

    def store_results(self, d):
        """
        This method stores the results.
        """
        ec_folder_path = os.path.join(d.results_folder, "error-correction")
        if not os.path.exists(ec_folder_path):
            os.mkdir(ec_folder_path)
        pickle.dump(d, open(os.path.join(ec_folder_path, "correction.dataset"), "wb"))
        if self.VERBOSE:
            print("The results are stored in {}.".format(os.path.join(ec_folder_path, "correction.dataset")))

    def run(self, d, random_seed):
        """
        This method runs Baran on an input dataset to correct data errors.
        """
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "---------------------Initialize the Dataset Object----------------------\n"
                  "------------------------------------------------------------------------")
        d = self.initialize_dataset(d)
        if len(d.detected_cells) == 0:
            raise ValueError('There are no errors in the data to correct.')
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "--------------------Initialize Error Corrector Models-------------------\n"
                  "------------------------------------------------------------------------")
        self.initialize_models(d)

        while len(d.labeled_tuples) < self.LABELING_BUDGET:
            self.sample_tuple(d, random_seed=random_seed)
            self.label_with_ground_truth(d)
            # self.update_models(d)
            self.prepare_augmented_models(d)
            self.generate_features(d, synchronous=True)
            if self.RULE_BASED_VALUE_CLEANING:
                self.rule_based_value_cleaning(d)
            # self.voting_binary_predict_corrections(d)
            self.binary_predict_corrections(d)
            # self.multi_predict_corrections(d)
            if self.RULE_BASED_VALUE_CLEANING:
                # write the rule-based value corrections into the corrections dictionary. This overwrites
                # results for domain & vicinity features. The idea is that the rule-based value
                # corrections are super precise and thus should be used if possible.

                for cell, correction in d.rule_based_value_corrections.items():
                    d.corrected_cells[cell] = correction

            self.clean_with_user_input(d)
            if self.VERBOSE:
                p, r, f = d.get_data_cleaning_evaluation(d.corrected_cells)[-3:]
                print(
                    "Cleaning performance on {}:\nPrecision = {:.2f}\nRecall = {:.2f}\nF1 = {:.2f}".format(d.name, p, r,
                                                                                                           f))
        if self.VERBOSE:
            print("Number of true classes: {}".format(self.n_true_classes))
        return d.corrected_cells


if __name__ == "__main__":
    # configure Cleaning object
    classification_model = "ABC"

    dataset_name = "hospital"
    version = 1
    error_fraction = 4
    error_class = 'simple_mcar'

    feature_generators = ['domain', 'vicinity', ]
    imputer_cache_model = False
    clean_with_user_input = False
    labeling_budget = 20
    n_best_pdeps = 3
    n_rows = None
    rule_based_value_cleaning = 'V5'
    synth_tuples = 0
    synth_tuples_error_threshold = 0
    training_time_limit = 30
    vicinity_feature_generator = "pdep"
    vicinity_orders = [1, 2]

    # Load Dataset object
    data_dict = helpers.get_data_dict(dataset_name, error_fraction, version, error_class)

    # Set this parameter to keep runtimes low when debugging
    data = raha.dataset.Dataset(data_dict, n_rows=n_rows)
    data.detected_cells = data.get_errors_dictionary()

    app = Correction(labeling_budget, classification_model, clean_with_user_input, feature_generators, vicinity_orders,
                     vicinity_feature_generator, imputer_cache_model, n_best_pdeps, training_time_limit,
                     rule_based_value_cleaning, synth_tuples, synth_tuples_error_threshold)
    app.VERBOSE = True
    seed = 0
    correction_dictionary = app.run(data, seed)
    p, r, f = data.get_data_cleaning_evaluation(correction_dictionary)[-3:]
    print("Cleaning performance on {}:\nPrecision = {:.2f}\nRecall = {:.2f}\nF1 = {:.2f}".format(data.name, p, r, f))
