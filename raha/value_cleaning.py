import json
import difflib
import unicodedata
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
            # ValueError('not implemented yet') TODO implement transform_replace()
            optimised_rule.append(o)
        else:
            optimised_rule.append(o)
    return json.dumps(optimised_rule)


def features_from_rules(error: str, rules: List[Dict]) -> List[Dict]:
    rule_features = []
    total_times_created_id = sum([s['times_created'] for s in rules if s['encoding'] == 'identity'])
    total_times_created_uc = sum([s['times_created'] for s in rules if s['encoding'] == 'unicode'])
    for rule in rules:
        total = total_times_created_id if rule['encoding'] == 'identity' else total_times_created_uc
        correction = clean_with_rule(error, rule['rule'])
        relative_string_frequency = rule['times_created'] / total

        features = {'correction': correction,
                    'rule': rule['rule'],
                    'encoding': rule['encoding'],
                    'relative_string_frequency': relative_string_frequency,
                    'error_cells': rule['error_cells']
                    }
        rule_features.append(features)
    return rule_features


def legacy_to_value_model_adder(value_model, enc_error_value, transformation, error_cell):
    """
        This method updates the value_model. It is different from _to_model_adder, since value_model contains
        coordinates of the error_cell from which the transformation originates.
        """
    if enc_error_value not in value_model:
        value_model[enc_error_value] = {}
    if transformation not in value_model[enc_error_value]:
        value_model[enc_error_value][transformation] = []
    value_model[enc_error_value][transformation].append(error_cell)


def legacy_assemble_cleaning_suggestion(
        transformation_string: str, model_name: str, old_value: str
) -> Union[str, None]:
    index_character_dictionary = {i: c for i, c in enumerate(old_value)}
    transformation = json.loads(transformation_string)
    for change_range_string in transformation:
        change_range = json.loads(change_range_string)
        if model_name in ["remover", "replacer"]:
            for i in range(change_range[0], change_range[1]):
                index_character_dictionary[i] = ""
        if model_name in ["adder", "replacer"]:
            ov = (
                ""
                if change_range[0] not in index_character_dictionary
                else index_character_dictionary[change_range[0]]
            )
            index_character_dictionary[change_range[0]] = (
                    transformation[change_range_string] + ov
            )
    new_value = ""
    # try:
    for i in range(len(index_character_dictionary)):
        new_value += index_character_dictionary[i]
    # except KeyError:  # not possible to transform old_value.
    #     new_value = None
    return new_value


class ValueCleaning:
    def __init__(self):
        self.map_error_to_rules: Dict[EncodedError, CleaningRules] = {}
        self.atomic_rules = ({}, {}, {}, {})

    def _atomic_update_rules(self, error: str, correction: str, error_cell: Tuple[int, int]):
        """
        Atomic Value Cleaning as implemented in Baran.
        """
        remover_transformation = {}
        adder_transformation = {}
        replacer_transformation = {}
        s = difflib.SequenceMatcher(None, error, correction)
        for tag, i1, i2, j1, j2 in s.get_opcodes():
            index_range = json.dumps([i1, i2])
            if tag == "delete":
                remover_transformation[index_range] = ""
            if tag == "insert":
                adder_transformation[index_range] = correction[j1:j2]
            if tag == "replace":
                replacer_transformation[index_range] = correction[j1:j2]
        for encoded_old_value in encode_error(error):
            if remover_transformation:
                legacy_to_value_model_adder(self.atomic_rules[0], encoded_old_value,
                                            json.dumps(remover_transformation),
                                            error_cell)
            if adder_transformation:
                legacy_to_value_model_adder(self.atomic_rules[1], encoded_old_value,
                                            json.dumps(adder_transformation), error_cell)
            if replacer_transformation:
                legacy_to_value_model_adder(self.atomic_rules[2], encoded_old_value,
                                            json.dumps(replacer_transformation),
                                            error_cell)
            legacy_to_value_model_adder(self.atomic_rules[3], encoded_old_value, correction, error_cell)

    def _atomic_cleaning_features(self, error) -> List[List[Dict]]:
        results_list = []
        for model, model_name in zip(self.atomic_rules, ["remover", "adder", "replacer", "swapper"]):
            for encoding, encoded_value_string in zip(('identity', 'unicode'), encode_error(error)):
                model_results = []  # one list per operation + encoding
                if encoded_value_string in model:
                    # Aus V1 urspruenglich.
                    sum_scores = sum([len(x) for x in model[encoded_value_string].values()])
                    if model_name in ["remover", "adder", "replacer"]:
                        for transformation_string in model[encoded_value_string]:
                            new_value = legacy_assemble_cleaning_suggestion(transformation_string,
                                                                            model_name,
                                                                            error)

                            # Aus V1 und ursprünglich Baran.
                            pr_baran = len(model[encoded_value_string][transformation_string]) / sum_scores

                            # Aus V2. Muss jemals optimiert werden, kann das weg.
                            error_cells = model[encoded_value_string][transformation_string]
                            rule = transformation_string
                            model_results.append({'correction': new_value,
                                                  'rule': rule,
                                                  'encoding': encoding,
                                                  'relative_string_frequency': pr_baran,
                                                  'error_cells': error_cells,
                                                  })

                    elif model_name == "swapper":
                        for new_value in model[encoded_value_string]:
                            # Aus V1.
                            pr_baran = len(model[encoded_value_string][new_value]) / sum_scores

                            # Aus V2.
                            error_cells = model[encoded_value_string][new_value]
                            rule = "swap"
                            model_results.append({'correction': new_value,
                                                  'rule': rule,
                                                  'encoding': encoding,
                                                  'relative_string_frequency': pr_baran,
                                                  'error_cells': error_cells,
                                                  })
                    else:
                        raise ValueError('Undefined model name.')

                results_list.append(model_results)
        return results_list

    def update_rules(self, error: str, correction: str, error_cell: Tuple[int, int]):
        self._atomic_update_rules(error, correction, error_cell)
        rule = rule_from_correction(error, correction)
        optimised_rule = optimise_rule(rule, error)

        for encoded_error in encode_error(error):
            if self.map_error_to_rules.get(encoded_error) is None:
                self.map_error_to_rules[encoded_error] = {}
            if self.map_error_to_rules[encoded_error].get(optimised_rule) is None:
                self.map_error_to_rules[encoded_error][optimised_rule] = {'times_created': 0,
                                                                          'error_cells': []}
            self.map_error_to_rules[encoded_error][optimised_rule]['times_created'] += 1
            self.map_error_to_rules[encoded_error][optimised_rule]['error_cells'].append(error_cell)

    def cleaning_features(self, error: str, features: str = 'both') -> List[Dict]:
        """
        Generate cleaning features.
        @param error: An error to be corrected.
        @param features: Either 'atomic', 'complex' or 'both'. If 'atomic', generate atomic features from the optcodes.
        If 'complex', generate only complex rules. And if 'both', generate both kind of features.
        @return:
        """
        identity_encoded, unicode_encoded = encode_error(error)
        identity_rules = self.map_error_to_rules.get(identity_encoded, {})
        unicode_rules = self.map_error_to_rules.get(unicode_encoded, {})

        rules_dicts = []
        for rule, properties in identity_rules.items():
            rule_features = {'rule': rule,
                             'encoding': 'identity',
                             **properties}
            rules_dicts.append(rule_features)

        for rule, properties in unicode_rules.items():
            rule_features = {'rule': rule,
                             'encoding': 'unicode',
                             **properties}
            rules_dicts.append(rule_features)

        if len(rules_dicts) == 0:
            return []
        if features == 'atomic':
            return self._atomic_cleaning_features(error)
        elif features == 'complex':
            return features_from_rules(error, rules_dicts)
        elif features == 'both':
            return self._atomic_cleaning_features(error) + [features_from_rules(error, rules_dicts)]
        else:
            raise ValueError(f'Invalid parameter features: {features}. Needs to be either "atomic", "complex", or '
                             f'"both".')
