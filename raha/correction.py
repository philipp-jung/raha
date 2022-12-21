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

import bs4
import bz2
import numpy
import py7zr
import mwparserfromhell
import sklearn.svm
import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.linear_model
import sklearn.tree
from typing import Dict, List, Union
import pandas as pd

import raha
from raha import imputer
from raha import pdep
from raha import hpo
from raha import helpers
from raha import ml_helpers
from raha import value_helpers

########################################


########################################

class Correction:
    """
    The main class.
    """

    def __init__(self, labeling_budget: int, classification_model: str, feature_generators: List[str],
                 vicinity_orders: List[int], vicinity_feature_generator: str, imputer_cache_model: bool,
                 n_best_pdeps: int, training_time_limit: int, rule_based_value_cleaning: Union[str, bool],
                 synth_tuples: int):
        """
        Parameters of the cleaning experiment.
        @param labeling_budget: How many tuples are labeled by the user. Baran default is 20.
        @param classification_model: "ABC" for sklearn's AdaBoostClassifier with n_estimators=100, "CV" for
        cross-validation. Baran default is "ABC".
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
        """
        # Philipps changes
        self.SYNTH_TUPLES = synth_tuples

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

    @staticmethod
    def _wikitext_segmenter(wikitext):
        """
        This method takes a Wikipedia page revision text in wikitext and segments it recursively.
        """
        def recursive_segmenter(node):
            if isinstance(node, str):
                segments_list.append(node)
            elif isinstance(node, mwparserfromhell.nodes.text.Text):
                segments_list.append(node.value)
            elif not node:
                pass
            elif isinstance(node, mwparserfromhell.wikicode.Wikicode):
                for n in node.nodes:
                    if isinstance(n, str):
                        recursive_segmenter(n)
                    elif isinstance(n, mwparserfromhell.nodes.text.Text):
                        recursive_segmenter(n.value)
                    elif isinstance(n, mwparserfromhell.nodes.heading.Heading):
                        recursive_segmenter(n.title)
                    elif isinstance(n, mwparserfromhell.nodes.tag.Tag):
                        recursive_segmenter(n.contents)
                    elif isinstance(n, mwparserfromhell.nodes.wikilink.Wikilink):
                        if n.text:
                            recursive_segmenter(n.text)
                        else:
                            recursive_segmenter(n.title)
                    elif isinstance(n, mwparserfromhell.nodes.external_link.ExternalLink):
                        # recursive_parser(n.url)
                        recursive_segmenter(n.title)
                    elif isinstance(n, mwparserfromhell.nodes.template.Template):
                        recursive_segmenter(n.name)
                        for p in n.params:
                            # recursive_parser(p.name)
                            recursive_segmenter(p.value)
                    elif isinstance(n, mwparserfromhell.nodes.html_entity.HTMLEntity):
                        segments_list.append(n.normalize())
                    elif not n or isinstance(n, mwparserfromhell.nodes.comment.Comment) or \
                            isinstance(n, mwparserfromhell.nodes.argument.Argument):
                        pass
                    else:
                        sys.stderr.write("Inner layer unknown node found: {}, {}\n".format(type(n), n))
            else:
                sys.stderr.write("Outer layer unknown node found: {}, {}\n".format(type(node), node))

        try:
            parsed_wikitext = mwparserfromhell.parse(wikitext)
        except:
            parsed_wikitext = ""
        segments_list = []
        recursive_segmenter(parsed_wikitext)
        return segments_list

    def extract_revisions(self, wikipedia_dumps_folder):
        """
        This method takes the folder path of Wikipedia page revision history dumps and extracts the value-based corrections.
        """
        rd_folder_path = os.path.join(wikipedia_dumps_folder, "revision-data")
        if not os.path.exists(rd_folder_path):
            os.mkdir(rd_folder_path)
        compressed_dumps_list = [df for df in os.listdir(wikipedia_dumps_folder) if df.endswith(".7z")]
        page_counter = 0
        for file_name in compressed_dumps_list:
            compressed_dump_file_path = os.path.join(wikipedia_dumps_folder, file_name)
            dump_file_name, _ = os.path.splitext(os.path.basename(compressed_dump_file_path))
            rdd_folder_path = os.path.join(rd_folder_path, dump_file_name)
            if not os.path.exists(rdd_folder_path):
                os.mkdir(rdd_folder_path)
            else:
                continue
            archive = py7zr.SevenZipFile(compressed_dump_file_path, mode="r")
            archive.extractall(path=wikipedia_dumps_folder)
            archive.close()
            decompressed_dump_file_path = os.path.join(wikipedia_dumps_folder, dump_file_name)
            decompressed_dump_file = io.open(decompressed_dump_file_path, "r", encoding="utf-8")
            page_text = ""
            for i, line in enumerate(decompressed_dump_file):
                line = line.strip()
                if line == "<page>":
                    page_text = ""
                page_text += "\n" + line
                if line == "</page>":
                    revisions_list = []
                    page_tree = bs4.BeautifulSoup(page_text, "html.parser")
                    previous_text = ""
                    for revision_tag in page_tree.find_all("revision"):
                        revision_text = revision_tag.find("text").text
                        if previous_text:
                            a = [t for t in self._wikitext_segmenter(previous_text) if t]
                            b = [t for t in self._wikitext_segmenter(revision_text) if t]
                            s = difflib.SequenceMatcher(None, a, b)
                            for tag, i1, i2, j1, j2 in s.get_opcodes():
                                if tag == "equal":
                                    continue
                                revisions_list.append({
                                    "old_value": a[i1:i2],
                                    "new_value": b[j1:j2],
                                    "left_context": a[i1 - self.REVISION_WINDOW_SIZE:i1],
                                    "right_context": a[i2:i2 + self.REVISION_WINDOW_SIZE]
                                })
                        previous_text = revision_text
                    if revisions_list:
                        page_counter += 1
                        if self.VERBOSE and page_counter % 100 == 0:
                            for entry in revisions_list:
                                print("----------Page Counter:---------\n", page_counter,
                                      "\n----------Old Value:---------\n", entry["old_value"],
                                      "\n----------New Value:---------\n", entry["new_value"],
                                      "\n----------Left Context:---------\n", entry["left_context"],
                                      "\n----------Right Context:---------\n", entry["right_context"],
                                      "\n==============================")
                        json.dump(revisions_list, open(os.path.join(rdd_folder_path, page_tree.id.text + ".json"), "w"))
            decompressed_dump_file.close()
            os.remove(decompressed_dump_file_path)
            if self.VERBOSE:
                print("{} ({} / {}) is processed.".format(file_name, len(os.listdir(rd_folder_path)), len(compressed_dumps_list)))

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
        if self.ONLINE_PHASE or (ud["new_value"] and len(ud["new_value"]) <= self.MAX_VALUE_LENGTH and
                                 ud["old_value"] and len(ud["old_value"]) <= self.MAX_VALUE_LENGTH and
                                 ud["old_value"] != ud["new_value"] and ud["old_value"].lower() != "n/a" and
                                 not ud["old_value"][0].isdigit()):
            remover_transformation = {}
            adder_transformation = {}
            replacer_transformation = {}
            s = difflib.SequenceMatcher(None, ud["old_value"], ud["new_value"])
            for tag, i1, i2, j1, j2 in s.get_opcodes():
                index_range = json.dumps([i1, i2])
                if tag == "delete":
                    remover_transformation[index_range] = ""
                if tag == "insert":
                    adder_transformation[index_range] = ud["new_value"][j1:j2]
                if tag == "replace":
                    replacer_transformation[index_range] = ud["new_value"][j1:j2]
            for encoding in self.VALUE_ENCODINGS:
                encoded_old_value = self._value_encoder(ud["old_value"], encoding)
                if remover_transformation:
                    self._to_value_model_adder(models[0], encoded_old_value, json.dumps(remover_transformation), cell)
                if adder_transformation:
                    self._to_value_model_adder(models[1], encoded_old_value, json.dumps(adder_transformation), cell)
                if replacer_transformation:
                    self._to_value_model_adder(models[2], encoded_old_value, json.dumps(replacer_transformation), cell)
                self._to_value_model_adder(models[3], encoded_old_value, ud["new_value"], cell)

    def pretrain_value_based_models(self, revision_data_folder):
        """
        This method pretrains value-based error corrector models.
        """
        def _models_pruner():
            for mi, model in enumerate(models):
                for k in list(model.keys()):
                    for v in list(model[k].keys()):
                        if model[k][v] < self.MIN_CORRECTION_OCCURRENCE:
                            models[mi][k].pop(v)
                    if not models[mi][k]:
                        models[mi].pop(k)

        models = [{}, {}, {}, {}]
        rd_folder_path = revision_data_folder
        page_counter = 0
        for folder in os.listdir(rd_folder_path):
            if os.path.isdir(os.path.join(rd_folder_path, folder)):
                for rf in os.listdir(os.path.join(rd_folder_path, folder)):
                    if rf.endswith(".json"):
                        page_counter += 1
                        if page_counter % 100000 == 0:
                            _models_pruner()
                            if self.VERBOSE:
                                print(page_counter, "pages are processed.")
                        try:
                            revision_list = json.load(io.open(os.path.join(rd_folder_path, folder, rf), encoding="utf-8"))
                        except:
                            continue
                        for rd in revision_list:
                            update_dictionary = {
                                "old_value": raha.dataset.Dataset.value_normalizer("".join(rd["old_value"])),
                                "new_value": raha.dataset.Dataset.value_normalizer("".join(rd["new_value"]))
                            }
                            self._value_based_models_updater(models, update_dictionary, (0, 0))
        _models_pruner()
        pretrained_models_path = os.path.join(revision_data_folder, "pretrained_value_based_models.dictionary")
        if self.PRETRAINED_VALUE_BASED_MODELS_PATH:
            pretrained_models_path = self.PRETRAINED_VALUE_BASED_MODELS_PATH
        pickle.dump(models, bz2.BZ2File(pretrained_models_path, "wb"))
        if self.VERBOSE:
            print("The pretrained value-based models are stored in {}.".format(pretrained_models_path))

    def _domain_based_model_updater(self, model, ud):
        """
        This method updates the domain-based error corrector model with a given update dictionary.
        """
        self._to_model_adder(model, ud["column"], ud["new_value"])

    def _value_based_corrector(self, models, ed, d):
        """
        This method takes the value-based models and an error dictionary to generate potential value-based corrections.
        """
        results_list = []
        for m, model_name in enumerate(["remover", "adder", "replacer", "swapper"]):
            model = models[m]
            for encoding in self.VALUE_ENCODINGS:
                results_dictionary = {}
                encoded_value_string = self._value_encoder(ed["old_value"], encoding)
                if encoded_value_string in model:
                    # Aus V1 urspruenglich.
                    sum_scores = sum([len(x) for x in model[encoded_value_string].values()])
                    if model_name in ["remover", "adder", "replacer"]:
                        for transformation_string in model[encoded_value_string]:
                            features = {}
                            a = 1
                            new_value = helpers.assemble_cleaning_suggestion(transformation_string,
                                                                             model_name,
                                                                             ed['old_value'])

                            # Aus V1 und ursprünglich Baran.
                            pr_baran = len(model[encoded_value_string][transformation_string]) / sum_scores

                            features['transformation_string'] = transformation_string  # for debugging
                            features['encoded_string_frequency'] = pr_baran

                            # Aus V2.
                            error_cells = model[encoded_value_string][transformation_string]
                            features['error_cells'] = error_cells

                            # für V5: Wende eine Regel auf alle korrekten Werte der selben Spalte an. Überspringe Werte,
                            # die die Regel nicht matchen. Matched die Regel ein Mal und führt zu einer falschen Reini-
                            # gung, überspringe die Regel.
                            # Die Ausführung ist ziemlich teuer. Deshalb verstecke ich sie hinter dieser Abfrage.
                            if self.RULE_BASED_VALUE_CLEANING == 'V5':
                                n_rows = d.dataframe.shape[0]
                                error_free_mask = [True if (x, ed['column']) not in d.detected_cells else False for x in range(n_rows)]
                                se_error_free = d.dataframe.iloc[error_free_mask, ed['column']]
                                unique_values_in_column = se_error_free.unique()
                                is_correct = True
                                for correct_value in unique_values_in_column:
                                    unicode_correct_value = self._value_encoder(correct_value, 'unicode')
                                    identity_correct_value = self._value_encoder(correct_value, 'identity')
                                    if unicode_correct_value in model or identity_correct_value in model:
                                        correction = helpers.assemble_cleaning_suggestion(transformation_string,
                                                                                          model_name,
                                                                                          correct_value)
                                        if correction != correct_value:
                                            is_correct = False
                                            break
                                features['is_correct_on_column'] = is_correct
                            results_dictionary[new_value] = features

                    if model_name == "swapper":
                        for new_value in model[encoded_value_string]:
                            features = {}

                            # Aus V1.
                            pr_baran = len(model[encoded_value_string][new_value]) / sum_scores
                            features['encoded_string_frequency'] = pr_baran

                            # Aus V2. Wird zzt. nicht benutzt und kann eigentlich weg.
                            error_cells = model[encoded_value_string][new_value]
                            features['error_cells'] = error_cells

                            if self.RULE_BASED_VALUE_CLEANING == 'V5':
                                n_rows = d.dataframe.shape[0]
                                error_free_mask = [True if (x, ed['column']) not in d.detected_cells else False for x in
                                                   range(n_rows)]
                                se_error_free = d.dataframe.iloc[error_free_mask, ed['column']]
                                unique_values_in_column = se_error_free.unique()
                                is_correct = True
                                features['transformation_string'] = transformation_string  # for debugging
                                for correct_value in unique_values_in_column:
                                    unicode_correct_value = self._value_encoder(correct_value, 'unicode')
                                    identity_correct_value = self._value_encoder(correct_value, 'identity')
                                    if unicode_correct_value in model or identity_correct_value in model:
                                        correction = helpers.assemble_cleaning_suggestion(transformation_string,
                                                                                          model_name,
                                                                                          correct_value)
                                        if correction != correct_value:
                                            is_correct = False
                                            break
                                features['is_correct_on_column'] = is_correct
                            results_dictionary[new_value] = features

                results_list.append(results_dictionary)
        return results_list

    def _imputer_based_corrector(self, model: Dict[int, pd.DataFrame], ed: dict) -> list:
        """
        Use an AutoGluon imputer to generate corrections.

        @param model: Dictionary with an AutoGluonImputer per column.
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
        # TODO normalize probabilities when old error gets deleted
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
        d.column_errors = {i: {} for i in range(d.dataframe.shape[1])}
        d.row_errors = {i: 0 for i in range(d.dataframe.shape[0])}
        for cell in d.detected_cells:
            self._to_model_adder(d.column_errors, cell[1], cell)
            d.row_errors[cell[0]] = 1
        d.labeled_tuples = {} if not hasattr(d, "labeled_tuples") else d.labeled_tuples
        d.labeled_cells = {} if not hasattr(d, "labeled_cells") else d.labeled_cells
        d.corrected_cells = {} if not hasattr(d, "corrected_cells") else d.corrected_cells
        return d

    def initialize_models(self, d):
        """
        This method initializes the error corrector models.
        """
        d.value_models = [{}, {}, {}, {}]
        d.pdeps = {c: {cc: {} for cc in range(d.dataframe.shape[1])}
                   for c in range(d.dataframe.shape[1])}
        if os.path.exists(self.PRETRAINED_VALUE_BASED_MODELS_PATH):
            d.value_models = pickle.load(bz2.BZ2File(self.PRETRAINED_VALUE_BASED_MODELS_PATH, "rb"))
            if self.VERBOSE:
                print("The pretrained value-based models are loaded.")

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

        # for debugging purposes only
        d.domain_corrections = {}
        d.naive_vicinity_corrections = {}
        d.pdep_vicinity_corrections = {}
        d.imputer_corrections = {}

        d.vicinity_models = {}
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
        Philipp extended this with a random_seed in an effort to make runs
        reproducible.

        I also added two tiers of columns to choose samples from. Either, there are error cells
        for which no correction suggestion has been made yet. In that case, we sample from these error cells.
        If correction suggestions have been made for all cells, we sample from error cells that have not been
        sampled before.
        """
        rng = numpy.random.default_rng(seed=random_seed)
        remaining_column_unlabeled_cells = {}
        remaining_column_unlabeled_cells_error_values = {}
        remaining_column_uncorrected_cells = {}
        remaining_column_uncorrected_cells_error_values = {}
        for j in d.column_errors:
            for error_cell in d.column_errors[j]:
                if error_cell not in d.corrected_cells:  # no correction suggestion has been found yet.
                    self._to_model_adder(remaining_column_uncorrected_cells, j, error_cell)
                    self._to_model_adder(remaining_column_uncorrected_cells_error_values, j, d.dataframe.iloc[error_cell])
                if error_cell not in d.labeled_cells:
                    # the cell has not been labeled by the user yet. this is stricter than the above condition.
                    self._to_model_adder(remaining_column_unlabeled_cells, j, error_cell)
                    self._to_model_adder(remaining_column_unlabeled_cells_error_values, j, d.dataframe.iloc[error_cell])
        tuple_score = numpy.ones(d.dataframe.shape[0])
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
                column_score = math.exp(len(remaining_columns_to_choose_from[j]) / len(d.column_errors[j]))
                cell_score = math.exp(remaining_cell_error_values[j][value] / len(remaining_columns_to_choose_from[j]))
                tuple_score[cell[0]] *= column_score * cell_score


        # Nützlich, um tuple-sampling zu debuggen: Zeigt die Tupel, aus denen
        # zufällig gewählt wird.
        # print(numpy.argwhere(tuple_score == numpy.amax(tuple_score)).flatten())
        d.sampled_tuple = rng.choice(numpy.argwhere(tuple_score == numpy.amax(tuple_score)).flatten())
        if self.VERBOSE:
            print("Tuple {} is sampled.".format(d.sampled_tuple))

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

                # and the cell hasn't been detected as an error
                if cell not in d.detected_cells:
                    # add that cell to detected_cells and assign it IGNORE_SIGN
                    # --> das passiert, wenn die Error Detection nicht perfekt
                    # war, dass man einen Fehler labelt, der vorher noch nicht
                    # gelabelt war.
                    d.detected_cells[cell] = self.IGNORE_SIGN
                    self._to_model_adder(d.column_errors, cell[1], cell)

            else:
                update_dictionary["vicinity"] = [cv if column != cj and d.labeled_cells[(d.sampled_tuple, cj)][0] == 1
                                                 else self.IGNORE_SIGN for cj, cv in enumerate(cleaned_sampled_tuple)]

        # BEGIN Philipp's changes
        for o in self.VICINITY_ORDERS:
            pdep.update_counts_dict(d.dataframe,
                                    d.vicinity_models[o],
                                    o,
                                    cleaned_sampled_tuple)
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
            column_corrections = []
            column_errors = []
            for labeled_cell in d.labeled_cells:
                if labeled_cell in d.column_errors[cell[1]]:
                    column_errors.append(d.dataframe.iloc[labeled_cell])
                    column_corrections.append(d.labeled_cells[labeled_cell][1])
            value_corrections = self._value_based_corrector(d.value_models,
                                                            error_dictionary,
                                                            d=d)
        if "domain" in self.FEATURE_GENERATORS:
            domain_corrections = self._domain_based_corrector(d.domain_models, error_dictionary)
        if "imputer" in self.FEATURE_GENERATORS and len(d.labeled_tuples) == (self.LABELING_BUDGET - 1):
            imputer_corrections = self._imputer_based_corrector(d.imputer_models, error_dictionary)

        if not is_synth:
            d.value_corrections[cell] = value_corrections
        # below block is for debugging purposes only.
        d.domain_corrections[cell] = domain_corrections
        d.naive_vicinity_corrections[cell] = naive_vicinity_corrections
        d.pdep_vicinity_corrections[cell] = pdep_vicinity_corrections
        d.imputer_corrections[cell] = imputer_corrections

        if not self.RULE_BASED_VALUE_CLEANING and not is_synth:
            # construct value corrections as in original Baran
            value_corrections = [{correction: d[correction]['encoded_string_frequency']} for d in value_corrections for correction in d]
            models_corrections = value_corrections \
            + domain_corrections \
            + [corrections for order in naive_vicinity_corrections for corrections in order] \
            + [corrections for order in pdep_vicinity_corrections for corrections in order] \
            + imputer_corrections
        elif not self.RULE_BASED_VALUE_CLEANING and is_synth:
            raise ValueError('It is impossible to synthesize tuples and use the meta-learner to apply value-corrections'
                             ' at the same time. Either perform RULE_BASED_VALUE_CLEANING, or set n_synth_tuples=0.')
        else:
            models_corrections = domain_corrections \
            + [corrections for order in naive_vicinity_corrections for corrections in order] \
            + [corrections for order in pdep_vicinity_corrections for corrections in order] \
            + imputer_corrections
        # End Philipps Changes

        corrections_features = {}
        for mi, model in enumerate(models_corrections):
            for correction in model:
                if correction not in corrections_features:
                    corrections_features[correction] = numpy.zeros(len(models_corrections))
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

        if 'imputer' in self.FEATURE_GENERATORS and len(d.labeled_tuples) == (self.LABELING_BUDGET - 1):
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
                    d.imputer_models[i_col] = imp.predict_proba(d.dataframe)
                else:
                    d.imputer_models[i_col] = None

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
            for args_list in process_args_list:
                result = self._feature_generator_process(args_list)
                feature_generation_results.append(result)

        for ci, corrections_features in enumerate(feature_generation_results):
            cell = process_args_list[ci][1]
            d.pair_features[cell] = {}
            for correction in corrections_features:
                d.pair_features[cell][correction] = corrections_features[correction]
                pairs_counter += 1

        if self.VERBOSE:
            print("{} pairs of (a data error, a potential correction) are featurized.".format(pairs_counter))

        synth_tuples_choice = [row for row, is_error in d.row_errors.items() if is_error == 0]
        if self.SYNTH_TUPLES > 0 and len(synth_tuples_choice) > 0:
            try:
                synthetic_error_rows = random.sample(synth_tuples_choice, k=self.SYNTH_TUPLES)
            except ValueError:  # not enough synthetic data to sample from.
                synthetic_error_rows = synth_tuples_choice
            synthetic_error_cells = [(i, j) for i in synthetic_error_rows for j in range(d.dataframe.shape[1])]
            synth_args_list = [[d, cell, True] for cell in synthetic_error_cells]

            if not synchronous:
                pool = multiprocessing.Pool()
                synth_feature_generation_results = pool.map(self._feature_generator_process, synth_args_list)
                pool.close()
            else:
                synth_feature_generation_results = []
                for args_list in synth_args_list:
                    result = self._feature_generator_process(args_list)
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

        # for debugging synth tuples sampling
        # elif len(synth_tuples_choice) == 0:
        #     print('No error-free rows to sample data from.')

    def rule_based_value_cleaning(self, d):
        """ Find value corrections with a conditional probability of 1.0 and use them as corrections."""
        d.rule_based_value_corrections = {}

        for cell, value_corrections in d.value_corrections.items():
            rule_based_suggestion = None
            if self.RULE_BASED_VALUE_CLEANING == 'V1':
                value_suggestions = value_helpers.ValueSuggestions(cell, value_corrections)
                rule_based_suggestion = value_suggestions.rule_based_suggestion_v1(d)
            elif self.RULE_BASED_VALUE_CLEANING == 'V3':
                value_suggestions = value_helpers.ValueSuggestions(cell, value_corrections)
                rule_based_suggestion = value_suggestions.rule_based_suggestion_v3(d)
            elif self.RULE_BASED_VALUE_CLEANING == 'V4':
                value_suggestions = value_helpers.ValueSuggestions(cell, value_corrections)
                rule_based_suggestion = value_suggestions.rule_based_suggestion_v4()
            elif self.RULE_BASED_VALUE_CLEANING == 'V5':
                value_suggestions = value_helpers.ValueSuggestions(cell, value_corrections)
                rule_based_suggestion = value_suggestions.rule_based_suggestion_v5()
            if rule_based_suggestion is not None:
                d.rule_based_value_corrections[cell] = rule_based_suggestion

    def predict_corrections(self, d):
        for j in d.column_errors:
            x_train, y_train, x_test, user_corrected_cells, all_error_correction_suggestions = ml_helpers.generate_train_test_data(d.column_errors[j],
                                                                                                                                   d.labeled_cells,
                                                                                                                                   d.pair_features,
                                                                                                                                   d.dataframe,
                                                                                                                                   d.synth_pair_features)
            for error_cell in user_corrected_cells:
                d.corrected_cells[error_cell] = user_corrected_cells[error_cell]

            if len(x_train) > 0 and len(x_test) > 0:
                if sum(y_train) == 0:  # no correct suggestion was created for any of the error cells.
                    predicted_labels = numpy.zeros(len(x_test))
                elif sum(y_train) == len(y_train):  # no incorrect suggestion was created for any of the error cells.
                    predicted_labels = numpy.ones(len(x_test))
                else:
                    if self.n_true_classes.get(len(d.labeled_tuples)) is None:
                        self.n_true_classes[len(d.labeled_tuples)] = []
                    if self.CLASSIFICATION_MODEL == "ABC" or sum(y_train) <= 2:
                        gs_clf = sklearn.ensemble.AdaBoostClassifier(n_estimators=100)
                        gs_clf.fit(x_train, y_train)
                        self.n_true_classes[len(d.labeled_tuples)].append({'clf': str(gs_clf),
                                                                           'clf__n_estimators': 100,
                                                                           'n_y_true': sum(y_train)})
                    elif self.CLASSIFICATION_MODEL == "CV":
                        gs_clf = hpo.cross_validated_estimator(x_train, y_train)
                        self.n_true_classes[len(d.labeled_tuples)].append({'clf': gs_clf.best_estimator_,
                                                                           'score': gs_clf.best_score_,
                                                                           **gs_clf.best_params_,
                                                                           'n_y_true': sum(y_train)})
                    else:
                        raise ValueError('Unknown model.')
                    predicted_labels = gs_clf.predict(x_test)

                # set final cleaning suggestion from meta-learning result. User corrected cells are not overwritten!
                for index, predicted_label in enumerate(predicted_labels):
                    if predicted_label:
                        cell, predicted_correction = all_error_correction_suggestions[index]
                        d.corrected_cells[cell] = predicted_correction

            elif len(d.labeled_tuples) == 0:  # no training data is available because no user labels have been set.
                for cell in d.pair_features:
                    correction_dict = d.pair_features[cell]
                    if len(correction_dict) > 0:
                        # select the correction with the highest sum of features.
                        max_proba_feature = sorted([v for v in correction_dict.items()], key=lambda x: sum(x[1]), reverse=True)[0]
                        d.corrected_cells[cell] = max_proba_feature[0]

            elif len(x_train) > 0 and len(x_test) == 0:  # len(x_test) == 0 because all rows have been labeled.
                pass  # nothing to do here -- just use the manually corrected cells to correct all errors.

            elif len(x_train) == 0:
                pass  # nothing to learn because x_train is empty.
            else:
                raise ValueError('Invalid state')

        if self.VERBOSE:
            print("{:.0f}% ({} / {}) of data errors are corrected.".format(100 * len(d.corrected_cells) / len(d.detected_cells),
                                                                           len(d.corrected_cells), len(d.detected_cells)))

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
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "--------------Iterative Tuple Sampling, Labeling, and Learning----------\n"
                  "------------------------------------------------------------------------")
        while len(d.labeled_tuples) < self.LABELING_BUDGET:
            self.sample_tuple(d, random_seed=random_seed)
            self.label_with_ground_truth(d)
            self.update_models(d)
            self.prepare_augmented_models(d)
            self.generate_features(d, synchronous=True)
            if self.RULE_BASED_VALUE_CLEANING:
                self.rule_based_value_cleaning(d)
            self.predict_corrections(d)
            if self.RULE_BASED_VALUE_CLEANING:
                # write the rule-based value corrections into the corrections dictionary. This overwrites
                # results for domain & vicinity features. The idea is that the rule-based value
                # corrections are super precise and thus should be used if possible.

                # for value-performance debugging
                # print(f'{len(d.rule_based_value_corrections)} value corrections')
                for cell, correction in d.rule_based_value_corrections.items():
                    row, col = cell
                    if row not in d.labeled_tuples:  # don't overwrite user-given corrections
                        d.corrected_cells[cell] = correction

            if self.VERBOSE:
                p, r, f = d.get_data_cleaning_evaluation(d.corrected_cells)[-3:]
                print("Baran's performance on {}:\nPrecision = {:.2f}\nRecall = {:.2f}\nF1 = {:.2f}".format(d.name, p, r, f))

        if self.VERBOSE:
            print("Number of true classes: {}".format(self.n_true_classes))
        return d.corrected_cells


if __name__ == "__main__":

    # configure Cleaning object
    classification_model = "ABC"

    dataset_name = "beers"
    version = 2
    error_fraction = 10
    error_class = 'simple_mcar'

    # feature_generators = ['value', 'vicinity', 'domain']
    feature_generators = ['value']
    imputer_cache_model = False
    labeling_budget = 20
    n_best_pdeps = 3
    n_rows = None
    rule_based_value_cleaning = 'V4'
    synth_tuples = 20
    training_time_limit = 30
    vicinity_feature_generator = "pdep"
    vicinity_orders = [1, 2]

    # Load Dataset object
    data_dict = helpers.get_data_dict(dataset_name, error_fraction, version, error_class)

    # Set this parameter to keep runtimes low when debugging
    data = raha.dataset.Dataset(data_dict, n_rows=n_rows)
    data.detected_cells = dict(data.get_actual_errors_dictionary())

    app = Correction(labeling_budget, classification_model,  feature_generators, vicinity_orders,
                     vicinity_feature_generator, imputer_cache_model, n_best_pdeps, training_time_limit,
                     rule_based_value_cleaning, synth_tuples)
    app.VERBOSE = True
    seed = 0
    correction_dictionary = app.run(data, seed)
    p, r, f = data.get_data_cleaning_evaluation(correction_dictionary)[-3:]
    print("Baran's performance on {}:\nPrecision = {:.2f}\nRecall = {:.2f}\nF1 = {:.2f}".format(data.name, p, r, f))
