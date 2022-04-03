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

import raha
from raha import imputer
from raha import pdep
########################################


########################################

from IPython.core.debugger import set_trace

class Correction:
    """
    The main class.
    """

    def __init__(self):
        """
        The constructor.
        """
        self.PRETRAINED_VALUE_BASED_MODELS_PATH = ""
        self.VALUE_ENCODINGS = ["identity", "unicode"]
        self.CLASSIFICATION_MODEL = "ABC"   # ["ABC", "DTC", "GBC", "GNB", "KNC" ,"SGDC", "SVC"]
        self.IGNORE_SIGN = "<<<IGNORE_THIS_VALUE>>>"
        self.VERBOSE = False
        self.SAVE_RESULTS = True
        self.ONLINE_PHASE = False
        self.LABELING_BUDGET = 20
        self.MIN_CORRECTION_CANDIDATE_PROBABILITY = 0.0
        self.MIN_CORRECTION_OCCURRENCE = 2
        self.MAX_VALUE_LENGTH = 50
        self.REVISION_WINDOW_SIZE = 5

        # Philipps changes
        self.VICINITY_ONLY = False
        self.IMPUTER_FEATURE_GENERATOR = False
        self.VICINITY_ORDERS = [1]  # Baran default
        self.VICINITY_FEATURE_GENERATOR = "naive"  # "naive" or "pdep"
        self.N_BEST_PDEPS = None  # recommend up to 10. Ignored when using
        # naive feature generator.

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
    def _to_model_constant(model, key, value):
        """
        This methods incrementally adds a key-value into a dictionary-implemented model.
        Philipp hat es kaputt gemacht um zu testen, was passiert, wenn man die
        Vicinity models statisch auf 1 setzt.
        """
        if key not in model:
            model[key] = {}
        if value not in model[key]:
            model[key][value] = 0.0
        model[key][value] = 1.0

    @staticmethod
    def _to_model_ente(model, key, value):
        """
        This methods incrementally adds a key-value into a dictionary-implemented model.
        Philipp hat es kaputt gemacht um zu testen, was passiert, wenn man die
        Vicinity models komplett auslässt.
        """
        value = 'ente'
        if key not in model:
            model[key] = {}
        if value not in model[key]:
            model[key][value] = 0.0
        model[key][value] = 1.0

    @staticmethod
    def _to_model_adder(model, key, value):
        """
        This methods incrementally adds a key-value into a dictionary-implemented model.
        """
        if key not in model:
            model[key] = {}
        if value not in model[key]:
            model[key][value] = 0.0
        model[key][value] += 1.0

    def _value_based_models_updater(self, models, ud):
        """
        This method updates the value-based error corrector models with a given update dictionary.
        """
        # TODO: adding jabeja konannde bakhshahye substring
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
                    self._to_model_adder(models[0], encoded_old_value, json.dumps(remover_transformation))
                if adder_transformation:
                    self._to_model_adder(models[1], encoded_old_value, json.dumps(adder_transformation))
                if replacer_transformation:
                    self._to_model_adder(models[2], encoded_old_value, json.dumps(replacer_transformation))
                self._to_model_adder(models[3], encoded_old_value, ud["new_value"])

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
                            self._value_based_models_updater(models, update_dictionary)
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

    def _value_based_corrector(self, models, ed):
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
                    sum_scores = sum(model[encoded_value_string].values())
                    if model_name in ["remover", "adder", "replacer"]:
                        for transformation_string in model[encoded_value_string]:
                            index_character_dictionary = {i: c for i, c in enumerate(ed["old_value"])}
                            transformation = json.loads(transformation_string)
                            for change_range_string in transformation:
                                change_range = json.loads(change_range_string)
                                if model_name in ["remover", "replacer"]:
                                    for i in range(change_range[0], change_range[1]):
                                        index_character_dictionary[i] = ""
                                if model_name in ["adder", "replacer"]:
                                    ov = "" if change_range[0] not in index_character_dictionary else \
                                        index_character_dictionary[change_range[0]]
                                    index_character_dictionary[change_range[0]] = transformation[change_range_string] + ov
                            new_value = ""
                            for i in range(len(index_character_dictionary)):
                                new_value += index_character_dictionary[i]
                            pr = model[encoded_value_string][transformation_string] / sum_scores
                            if pr >= self.MIN_CORRECTION_CANDIDATE_PROBABILITY:
                                results_dictionary[new_value] = pr
                    if model_name == "swapper":
                        for new_value in model[encoded_value_string]:
                            pr = model[encoded_value_string][new_value] / sum_scores
                            if pr >= self.MIN_CORRECTION_CANDIDATE_PROBABILITY:
                                results_dictionary[new_value] = pr
                results_list.append(results_dictionary)
        return results_list


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
                if pr >= self.MIN_CORRECTION_CANDIDATE_PROBABILITY:
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
        d.column_errors = {}
        for cell in d.detected_cells:
            self._to_model_adder(d.column_errors, cell[1], cell)
        d.labeled_tuples = {} if not hasattr(d, "labeled_tuples") else d.labeled_tuples
        d.labeled_cells = {} if not hasattr(d, "labeled_cells") else d.labeled_cells
        d.corrected_cells = {} if not hasattr(d, "corrected_cells") else d.corrected_cells
        return d

    def _calc_pdep(self, d: dict, N: int, A: int, B: int):
        counts_dict = d[A][B]
        sum_components = []
        for lhs_val, rhs_dict in counts_dict.items(): # lhs_val same as A_i
            lhs_counts = sum(rhs_dict.values()) # same as a_i
            for rhs_val, rhs_counts in rhs_dict.items(): # rhs_counts same as n_ij
                sum_components.append(rhs_counts**2 / lhs_counts)
        return sum(sum_components) / N


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
        d.vicinity_models = {}
        for o in self.VICINITY_ORDERS:
            d.vicinity_models[o] = pdep.calculate_counts_dict(
                                    df=d.dataframe,
                                    detected_cells=d.detected_cells,
                                    order=o,
                                    ignore_sign=self.IGNORE_SIGN)
        if self.VERBOSE:
            print("The error corrector models are initialized.")

    def sample_tuple(self, d, random_seed):
        """
        This method samples a tuple.
        Philipp extended this with a random_seed to make runs (more?) reprodu-
        cible.
        """
        rng = numpy.random.default_rng(seed=random_seed)
        remaining_column_erroneous_cells = {}
        remaining_column_erroneous_values = {}
        for j in d.column_errors:
            for cell in d.column_errors[j]:
                if cell not in d.corrected_cells:
                    self._to_model_adder(remaining_column_erroneous_cells, j, cell)
                    self._to_model_adder(remaining_column_erroneous_values, j, d.dataframe.iloc[cell])
        tuple_score = numpy.ones(d.dataframe.shape[0])
        tuple_score[list(d.labeled_tuples.keys())] = 0.0
        for j in remaining_column_erroneous_cells:
            for cell in remaining_column_erroneous_cells[j]:
                value = d.dataframe.iloc[cell]
                column_score = math.exp(len(remaining_column_erroneous_cells[j]) / len(d.column_errors[j]))
                cell_score = math.exp(remaining_column_erroneous_values[j][value] / len(remaining_column_erroneous_cells[j]))
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
        """
        d.labeled_tuples[d.sampled_tuple] = 1
        for j in range(d.dataframe.shape[1]):
            cell = (d.sampled_tuple, j)
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
        cleaned_sampled_tuple = [d.labeled_cells[(d.sampled_tuple, j)][1] for j in range(d.dataframe.shape[1])]
        for j in range(d.dataframe.shape[1]):
            cell = (d.sampled_tuple, j)
            update_dictionary = {
                "column": cell[1],
                "old_value": d.dataframe.iloc[cell],
                "new_value": cleaned_sampled_tuple[j],
            }

            # if the value in that cell has been labelled an error...
            if d.labeled_cells[cell][0] == 1:
                # and the cell hasn't been detected as an error...
                if cell not in d.detected_cells:
                    # add that cell to detected_cells and assign it IGNORE_SIGN
                    # komisch wirklich, wann ist das denn bitteschön der Fall?
                    # Dass man eine Zelle labelt, die nicht in detected_cells ist?
                    d.detected_cells[cell] = self.IGNORE_SIGN
                    self._to_model_adder(d.column_errors, cell[1], cell)
                self._value_based_models_updater(d.value_models, update_dictionary)
                self._domain_based_model_updater(d.domain_models, update_dictionary)
                update_dictionary["vicinity"] = [cv if j != cj else self.IGNORE_SIGN
                                                 for cj, cv in enumerate(cleaned_sampled_tuple)]
            else:
                update_dictionary["vicinity"] = [cv if j != cj and d.labeled_cells[(d.sampled_tuple, cj)][0] == 1
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
        This method generates features for each data column. Depending on the
        config of `generate_features()`, it will run in parallel or not. This
        gets called once for each error cell.
        """
        d, cell = args

        # vicinity ist die Zeile, column ist die Zeilennummer, old_value ist der Fehler
        error_dictionary = {"column": cell[1], "old_value": d.dataframe.iloc[cell], "vicinity": list(d.dataframe.iloc[cell[0], :])}
        naive_vicinity_corrections, pdep_vicinity_corrections, value_corrections, domain_corrections = [], [], [], []

        # Begin Philipps Changes
        if self.VICINITY_FEATURE_GENERATOR == 'naive':
            for o in self.VICINITY_ORDERS:
                naive_corrections = pdep.vicinity_based_corrector_order_n(
                    counts_dict = d.vicinity_models[o],
                    ed=error_dictionary,
                    probability_threshold=self.MIN_CORRECTION_CANDIDATE_PROBABILITY)
                naive_vicinity_corrections.append(naive_corrections)

        elif self.VICINITY_FEATURE_GENERATOR == 'pdep':
            for o in self.VICINITY_ORDERS:
                pdep_corrections = pdep.pdep_vicinity_based_corrector(
                        inverse_sorted_gpdeps=d.inv_vicinity_gpdeps[o],
                        counts_dict=d.vicinity_models[o],
                        ed=error_dictionary,
                        probability_threshold=self.MIN_CORRECTION_CANDIDATE_PROBABILITY,
                        n_best_pdeps = self.N_BEST_PDEPS)
                pdep_vicinity_corrections.append(pdep_corrections)
        else:
            raise ValueError(f'Unknown VICINITY_FEATURE_GENERATOR'
                    f'{self.VICINITY_FEATURE_GENERATOR}')

        if not self.VICINITY_ONLY:  # skip if not required for experiment
            value_corrections = self._value_based_corrector(d.value_models, error_dictionary)
            domain_corrections = self._domain_based_corrector(d.domain_models, error_dictionary)

        # for debugging purposes
        d.naive_vicinity_corrections = naive_vicinity_corrections
        d.pdep_vicinity_corrections = pdep_vicinity_corrections

        models_corrections = value_corrections + domain_corrections \
        + [corrections for order in naive_vicinity_corrections for corrections in order] \
        + [corrections for order in pdep_vicinity_corrections for corrections in order]
        # End Philipps Changes

        corrections_features = {}
        for mi, model in enumerate(models_corrections):
            for correction in model:
                if correction not in corrections_features:
                    corrections_features[correction] = numpy.zeros(len(models_corrections))
                corrections_features[correction][mi] = model[correction]
        return corrections_features

    def generate_features(self, d, synchronous=False):
        """
        This method generates a feature vector for each pair of a data error
        and a potential correction.
        Philipp added a `synchronous` parameter to make debugging easier.
        """
        d.create_repaired_dataset(d.corrected_cells)

        # Calculate gpdeps and append them do d
        if self.VICINITY_FEATURE_GENERATOR == 'pdep':
            d.inv_vicinity_gpdeps = {}
            for o in self.VICINITY_ORDERS:
                vicinity_gpdeps = pdep.calc_all_gpdeps(d.vicinity_models[o],
                        d.repaired_dataframe)
                d.inv_vicinity_gpdeps[o] = pdep.invert_and_sort_gpdeps(vicinity_gpdeps)

        d.pair_features = {}
        pairs_counter = 0
        process_args_list = [[d, cell] for cell in d.detected_cells]
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

        if self.IMPUTER_FEATURE_GENERATOR:
            self.imputer_feature_generator(d)  # TODO parallelize this

        if self.VERBOSE:
            print("{} pairs of (a data error, a potential correction) are featurized.".format(pairs_counter))

    def predict_corrections(self, d, random_seed=None):
        """
        This method predicts

        Philipp added a random state for an experiment to all classification
        models.
        """
        for j in d.column_errors:
            x_train = []
            y_train = []
            x_test = []
            test_cell_correction_list = []
            for _, cell in enumerate(d.column_errors[j]):
                if cell in d.pair_features:
                    for correction in d.pair_features[cell]:
                        if cell in d.labeled_cells and d.labeled_cells[cell][0] == 1:
                            x_train.append(d.pair_features[cell][correction])
                            y_train.append(int(correction == d.labeled_cells[cell][1]))
                            d.corrected_cells[cell] = d.labeled_cells[cell][1]
                        else:
                            x_test.append(d.pair_features[cell][correction])
                            test_cell_correction_list.append([cell, correction])
            if self.CLASSIFICATION_MODEL == "ABC":
                classification_model = sklearn.ensemble.AdaBoostClassifier(n_estimators=100, random_state=random_seed)
            if self.CLASSIFICATION_MODEL == "DTC":
                classification_model = sklearn.tree.DecisionTreeClassifier(criterion="gini", random_state=random_seed)
            if self.CLASSIFICATION_MODEL == "GBC":
                classification_model = sklearn.ensemble.GradientBoostingClassifier(n_estimators=100)
            if self.CLASSIFICATION_MODEL == "GNB":
                classification_model = sklearn.naive_bayes.GaussianNB()
            if self.CLASSIFICATION_MODEL == "KNC":
                classification_model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
            if self.CLASSIFICATION_MODEL == "SGDC":
                classification_model = sklearn.linear_model.SGDClassifier(loss="hinge", penalty="l2")
            if self.CLASSIFICATION_MODEL == "SVC":
                classification_model = sklearn.svm.SVC(kernel="sigmoid")
            if x_train and x_test:
                if sum(y_train) == 0:
                    predicted_labels = numpy.zeros(len(x_test))
                elif sum(y_train) == len(y_train):
                    predicted_labels = numpy.ones(len(x_test))
                else:
                    classification_model.fit(x_train, y_train)
                    predicted_labels = classification_model.predict(x_test)
                # predicted_probabilities = classification_model.predict_proba(x_test)
                # correction_confidence = {}
                for index, predicted_label in enumerate(predicted_labels):
                    cell, predicted_correction = test_cell_correction_list[index]
                    # confidence = predicted_probabilities[index][1]
                    if predicted_label:
                        # if cell not in correction_confidence or confidence > correction_confidence[cell]:
                        #     correction_confidence[cell] = confidence
                        d.corrected_cells[cell] = predicted_correction
        if self.VERBOSE:
            print("{:.0f}% ({} / {}) of data errors are corrected.".format(100 * len(d.corrected_cells) / len(d.detected_cells),
                                                                           len(d.corrected_cells), len(d.detected_cells)))

    def imputer_feature_generator(self, d):
        """
        After the Baran-style cleaning has been finished, train an imputer
        model on top of it to try and improve performance a bit.
        """
        d.create_repaired_dataset(d.corrected_cells)
        imputer_corrections = imputer.imputation_based_corrector(d)
        for i_column in imputer_corrections:
            for imputer_correction in imputer_corrections[i_column]:
                for error_cell in d.pair_features:
                    for correction in d.pair_features[error_cell]:
                        if correction == list(imputer_correction.keys())[0]:
                            p = 1
                        else:
                            p = 0
                        d.pair_features[error_cell][correction] = numpy.append(d.pair_features[error_cell][correction], p)

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

    def run(self, d):
        """
        This method runs Baran on an input dataset to correct data errors.
        """
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "---------------------Initialize the Dataset Object----------------------\n"
                  "------------------------------------------------------------------------")
        d = self.initialize_dataset(d)
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
            self.sample_tuple(d, random_seed=0)
            if d.has_ground_truth:
                self.label_with_ground_truth(d)
            # else:
            #   In this case, user should label the tuple interactively as shown in the Jupyter notebook.
            self.update_models(d)
            self.generate_features(d)
            self.predict_corrections(d, random_seed=0)
            if self.VERBOSE:
                print("------------------------------------------------------------------------")
        if self.SAVE_RESULTS:
            if self.VERBOSE:
                print("------------------------------------------------------------------------\n"
                      "---------------------------Storing the Results--------------------------\n"
                      "------------------------------------------------------------------------")
            self.store_results(d)
        return d.corrected_cells
########################################


########################################
if __name__ == "__main__":
    dataset_name = "flights"
    dataset_dictionary = {
        "name": dataset_name,
        "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "dirty.csv")),
        "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "clean.csv"))
    }
    data = raha.dataset.Dataset(dataset_dictionary)
    data.detected_cells = dict(data.get_actual_errors_dictionary())
    app = Correction()
    correction_dictionary = app.run(data)
    p, r, f = data.get_data_cleaning_evaluation(correction_dictionary)[-3:]
    print("Baran's performance on {}:\nPrecision = {:.2f}\nRecall = {:.2f}\nF1 = {:.2f}".format(data.name, p, r, f))
    # --------------------
    # app.extract_revisions(wikipedia_dumps_folder="../wikipedia-data")
    # app.pretrain_value_based_models(revision_data_folder="../wikipedia-data/revision-data")
########################################
