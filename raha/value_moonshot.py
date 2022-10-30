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


def clean_with_rule(error: str, rule: str):
    rule = json.loads(rule)
    new_string = ''

    for instruction in rule:
        if instruction[0] == 'replace':
            new_string = new_string + instruction[3]
        if instruction[0] == 'delete':
            pass
        if instruction[0] == 'insert':
            new_string = new_string + instruction[3]
        if instruction[0] == 'equal':
            new_string = new_string + error[instruction[1]:instruction[2]]
    return new_string


def opcode_to_rule(correction: str, opcodes: list):
    rule = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'replace':
            rule.append(('replace', i1, i2, correction[j1:j2]))
        if tag == 'delete':
            rule.append(('delete', i1, i2))
        if tag == 'insert':
            rule.append(('insert', i1, i2, correction[j1:j2]))
        if tag == 'equal':
            rule.append(('equal', i1, i2))
    return rule


def rule_from_correction(error: str, correction: str):
    s = difflib.SequenceMatcher(None, error, correction, autojunk=False)
    opcodes = s.get_opcodes()
    rule = opcode_to_rule(correction, opcodes)
    return json.dumps(rule)


def transform_insert(o, error):
    insert_str = o[3]
    insert_pos_start = o[1]
    insert_pos_end = insert_pos_start + len(insert_str)

    s = difflib.SequenceMatcher(None, insert_str, error)
    blocks = s.get_matching_blocks()

    i = 0
    operations = []

    while i < insert_pos_end:
        if len(blocks) > 0:
            b = blocks.pop(0)
            if i == b.a:  # insert position is same as block's starting position
                operation = ['equal', b.b, b.b + b.size]
                i = b.a + b.size
            elif i < b.a:
                operation = ['insert', i, i, insert_str[i]]
                i += 1
            else:
                raise ValueError('Illegal state.')
        else:
            operation = ['insert', i, i, insert_str[i:insert_pos_end]]
            i = insert_pos_end
        operations.append(operation)
    return operations


def optimise_rule(rule: str, error: str):
    optimised_rule = []
    rule = json.loads(rule)
    for o in rule:
        if o[0] == 'insert':
            transformed = transform_insert(o, error)
            optimised_rule.extend(transformed)
        elif o[0] == 'replace':
            #ValueError('not implemented yet')
            optimised_rule.append(o)
        else:
            optimised_rule.append(o)
    return json.dumps(optimised_rule)


def features_for_rules(error: str, rules: Dict[RuleDumpJSON, RuleStatistics], column_errors, column_corrections) -> Dict:
    rule_features = {}
    sum_times_created = sum([stat['times_created'] for stat in rules.values()])
    for rule in rules:
        correction = clean_with_rule(error, rule)
        relative_times_created = rules[rule]['times_created'] / sum_times_created

        n_rule_worked_on_old_errors = 0
        for old_error, old_correction in zip(column_errors, column_corrections):
            if clean_with_rule(old_error, rule) == old_correction:
                n_rule_worked_on_old_errors += 1

        relative_success_old_errors = n_rule_worked_on_old_errors / len(column_errors) if len(column_errors) > 0 else None

        rule_features[correction] = {'relative_times_created': relative_times_created,
                                     'relative_success_old_errors': relative_success_old_errors}
    return rule_features


class ValueCleaning:
    def __init__(self):
        self.map_error_to_rules: Dict[EncodedError, CleaningRules] = {}

    def update_rules(self, error: str, correction: str, error_cell: Tuple[int, int]):
        rule = rule_from_correction(error, correction)
        optimised_rule = optimise_rule(rule, error)
        #rules = [rule, optimised_rule]
        rules = [optimised_rule]

        for encoded_error in encode_error(error):
            for rule in rules:
                if self.map_error_to_rules.get(encoded_error) is None:
                    self.map_error_to_rules[encoded_error] = {}
                if self.map_error_to_rules[encoded_error].get(rule) is None:
                    self.map_error_to_rules[encoded_error][rule] = {'times_created': 0,
                                                                    'error_cells': []}
                self.map_error_to_rules[encoded_error][rule]['times_created'] += 1
                self.map_error_to_rules[encoded_error][rule]['error_cells'].append(error_cell)

    def cleaning_features(self, error: str, column_errors: list, column_corrections: list):
        identity_encoded, unicode_encoded = encode_error(error)
        identity_rules = self.map_error_to_rules.get(identity_encoded, {})
        unicode_rules = self.map_error_to_rules.get(unicode_encoded, {})

        rules = {**identity_rules, **unicode_rules}
        if len(rules) == 0:
            return None
        features = features_for_rules(error, rules, column_errors, column_corrections)
        return features
