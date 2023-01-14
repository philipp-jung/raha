from dataclasses import dataclass
from typing import Tuple, List, Dict, Union, Set, NewType, Any

CorrectionSuggestion = NewType('CorrectionSuggestion', str)
FeatureType = NewType('FeatureType', str)
Suggestions = NewType('Suggestions', List[Dict[CorrectionSuggestion, Dict[FeatureType, Any]]])


@dataclass
class BothValueSuggestions:
    cell: Tuple[int, int]
    atomic_suggestions: List[List[Dict]]
    complex_suggestions: List[Dict]

    def identity_suggestions(self, model_type: str) -> List[Dict]:
        id_atomic = [x for y in self.atomic_suggestions for x in y if x['encoding'] == 'identity']
        id_complex = [x for x in self.complex_suggestions if x['encoding'] == 'identity']
        if model_type == 'atomic':
            return id_atomic
        if model_type == 'complex':
            return id_complex
        if model_type == 'both':
            return id_atomic + id_complex
        else:
            raise ValueError('Invalid model_type.')

    def unicode_suggestions(self, model_type: str) -> List[Dict]:
        uc_atomic = [x for model in self.atomic_suggestions for x in model if x['encoding'] == 'unicode']
        uc_complex = [x for x in self.complex_suggestions if x['encoding'] == 'unicode']
        if model_type == 'atomic':
            return uc_atomic
        if model_type == 'complex':
            return uc_complex
        if model_type == 'both':
            return uc_atomic + uc_complex
        else:
            raise ValueError('Invalid model_type.')

    def certain_suggestions(self, feature: str, model_type: str, encoding: str) -> List[Dict]:
        """
        Returns suggestions whose feature is 1.0
        @param feature: Feature that is supposed to be 1.0
        @param model_type: atomic, complex, or both
        @param encoding: identity, unicode, or both
        @return:
        """
        if encoding == 'identity':
            certain_atomic = [sug for sug in self.identity_suggestions(model_type) if sug[feature] == 1.0]
            certain_complex = [sug for sug in self.identity_suggestions(model_type) if sug[feature] == 1.0]
        elif encoding == 'unicode':
            certain_atomic = [sug for sug in self.unicode_suggestions(model_type) if sug[feature] == 1.0]
            certain_complex = [sug for sug in self.unicode_suggestions(model_type) if sug[feature] == 1.0]
        elif encoding == 'both':
            certain_atomic = [sug for model in self.atomic_suggestions for sug in model if sug[feature] == 1.0]
            certain_complex = [sug for sug in self.complex_suggestions if sug[feature] == 1.0]
        else:
            raise ValueError('Invalid encoding.')

        if model_type == 'atomic':
            return certain_atomic
        elif model_type == 'complex':
            return certain_complex
        elif model_type == 'both':
            return certain_atomic + certain_complex
        else:
            raise ValueError('Invalid model_type.')

    def n_identity_suggestions(self, model_type: str) -> int:
        return len(self.identity_suggestions(model_type))

    def n_unicode_suggestions(self, model_type: str) -> int:
        return len(self.unicode_suggestions(model_type))

    def n_certain_suggestions(self, feature: str, model_type: str, encoding: str) -> int:
        """
        Number of suggestions with probability 1.0
        @param feature: Feature that's supposed to have probabilty 1.0
        @param model_type: atomic, complex or both.
        @param encoding: identity, unicode, or both.
        """
        return len(self.certain_suggestions(feature, model_type, encoding))

    def n_certain_suggestions_no_swap(self, feature: str, model_type: str, encoding: str) -> int:
        """Number of unicode suggestions with probability 1.0 that are no swap operation."""
        certain_suggestions = self.certain_suggestions(feature, model_type, encoding)
        return len([sug for sug in certain_suggestions if sug['rule'] != 'swap'])

    def rule_based_suggestion_v4(self) -> Union[Dict, None]:
        """
        In V4 baue ich bewusst nur die Atomic Corrections nach.
        """
        choice = None
        threshold = 3

        if len(self.certain_suggestions('relative_string_frequency', 'atomic', 'both')) == 0:  # No certain suggestion at all.
            return choice

        if self.n_certain_suggestions('relative_string_frequency', 'atomic', 'both') == 1:  # Use the one certain suggestion, if it was generated from >= 3 user inputs.
            certains = [sug for sug in self.certain_suggestions('relative_string_frequency', 'atomic', 'both')]
            certain_choice = certains[0]
            if len(certain_choice['error_cells']) >= threshold:
                choice = certain_choice['correction']

            if len(certains) > 1:
                raise ValueError('oh no das sollte nicht sein.')

        elif self.n_certain_suggestions('relative_string_frequency', 'atomic', 'unicode') == 1:  #  Use the one certain unicode-encoded suggestion, if it was generated from >= inputs.
            certains = self.certain_suggestions('relative_string_frequency', 'atomic', 'unicode')
            certain_choice = certains[0]
            if len(certain_choice['error_cells']) >= threshold:
                choice = certain_choice['correction']

            if len(certains) > 1:
                raise ValueError('oh no das sollte nicht sein.')
        return choice

    def rule_based_suggestion_v5(self) -> Union[Dict, None]:
        if len(self.certain_suggestions('relative_string_frequency', 'both', 'both')) == 0:
            return None

        if len(self.certain_suggestions('relative_string_frequency', 'complex', 'both')) > 1:
            if len(self.certain_suggestions('relative_string_frequency', 'complex', 'unicode')) == 1:
                return self.certain_suggestions('relative_string_frequency', 'complex', 'unicode')[0]
            return None  # TODO sollten zwei Vorschläge die selbe Correction erhalten, aggregiere die und gib das zurück.

        if len(self.certain_suggestions('relative_string_frequency', 'complex', 'identity')) == 1:
            return self.certain_suggestions('relative_string_frequency', 'complex', 'identity')[0]

        if len(self.certain_suggestions('relative_string_frequency', 'atomic', 'both')) > 1:
            if len(self.certain_suggestions('relative_string_frequency', 'atomic', 'unicode')) == 1:
                return self.certain_suggestions('relative_string_frequency', 'atomic', 'unicode')[0]
            return None  # TODO sollten zwei Vorschläge die selbe Correction erhalten, aggregiere die und gib das zurück.

        if len(self.certain_suggestions('relative_string_frequency', 'atomic', 'identity')) == 1:
            return self.certain_suggestions('relative_string_frequency', 'atomic', 'identity')[0]

        return None
