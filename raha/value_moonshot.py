import json
import difflib
import unicodedata
import pandas as pd
from typing import Tuple, List, Dict, Union, Set, NewType, Any

EncodedError = NewType('EncodedError', str)
RuleDumpJSON = NewType('RuleDumpJSON', str)
RuleStatistics = NewType('RuleStatistics', Dict)
CleaningRules = NewType('CleaningRules', Dict[RuleDumpJSON, RuleStatistics])


def encode_error(error: str) -> Tuple[EncodedError, EncodedError]:
    return json.dumps(list(error)), json.dumps([unicodedata.category(c) for c in error])


def assemble_cleaning_suggestion(transformation_string: str, old_value: str) -> Union[str, None]:
    """
    Use the operation encoded in transform_string to transform old_value into a cleaning suggestion
    @param transformation_string: the encoded transformation.
    @param old_value: the erroneous value.
    @return: a cleaning suggestion.
    """
    index_character_dictionary = {i: c for i, c in enumerate(old_value)}
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
    try:
        for i in range(len(index_character_dictionary)):
            new_value += index_character_dictionary[i]
    except KeyError:  # not possible to transform old_value.
        new_value = None
    return new_value


def features_for_rules(rules: Dict[RuleDumpJSON, RuleStatistics], d, error_cell: Tuple[int, int]) -> Dict:
    """

    @param rules:
    @param d:
    @param error_cell: error cell of the cell for that we generate features to correct.
    @return:
    """
    rule_features = {}
    sum_times_created = sum([stat['times_created'] for stat in rules.values()])
    for rule in rules:
        relative_times_created = rules[rule]['times_created'] / sum_times_created
        old_errors = [error_str for cell, error_str in d.detected_cells.items() if cell[1] == error_cell[1]]

        n_rule_worked_on_old_errors = 0
        for error in old_errors:
            errors.append(d.dataframe.iloc[cell])
            corrections.append(d.labeled_cells[cell])
        rule_features[rule] = {'relative_times_created': relative_times_created,
                               }


class ValueCleaning:
    def __init__(self):
        self.map_error_to_rules: Dict[EncodedError, CleaningRules] = {}

    def update_rules(self, error: str, rule: RuleDumpJSON, error_cell: Tuple[int, int]):
        for encoded_error in encode_error(error):
            if self.map_error_to_rules.get(encoded_error) is None:
                self.map_error_to_rules[encoded_error] = {}
            if self.map_error_to_rules[encoded_error][rule] is None:
                self.map_error_to_rules[encoded_error][rule] = {'times_created': 0,
                                                                'error_cells': []}
            self.map_error_to_rules[encoded_error][rule]['times_created'] += 1
            self.map_error_to_rules[encoded_error][rule]['error_cells'].append(error_cell)

    def clean_value(self, error: str, error_cell: Tuple[int, int]):
        identity_encoded, unicode_encoded = encode_error(error)
        identity_rules = self.map_error_to_rules.get(identity_encoded)
        unicode_rules = self.map_error_to_rules.get(unicode_encoded)
        # TODO continue here
