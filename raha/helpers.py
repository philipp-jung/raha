from typing import Tuple, List, Dict, Union
from dataclasses import dataclass


@dataclass
class ValueSuggestions:
    """A class to hold the suggestions generated by the Value Feature-Generator."""
    cell: Tuple[int, int]
    suggestions: List[Dict[str, float]]
    model_types = ["identity_remover",
                   "unicode_remover",
                   "identity_adder",
                   "unicode_adder",
                   "identity_replacer",
                   "unicode_replacer",
                   "identity_swapper",
                   "unicode_swapper"]

    @property
    def identity_suggestions(self) -> List[Dict[str, float]]:
        return [self.suggestions[0], self.suggestions[2], self.suggestions[4], self.suggestions[6]]

    @property
    def unicode_suggestions(self) -> List[Dict[str, float]]:
        return [self.suggestions[1], self.suggestions[3], self.suggestions[5], self.suggestions[7]]

    @property
    def certain_model_type_indices_and_suggestions(self) -> Tuple[List[int], List[str]]:
        """Index of model types that make a certain suggestion."""
        indices = []
        certain_suggestions = []
        for i, model_suggestions in enumerate(self.suggestions):
            for s, pr in model_suggestions.items():
                if pr == 1:
                    indices.append(i)
                    certain_suggestions.append(s)
        return indices, certain_suggestions

    @property
    def n_certain_suggestions(self) -> int:
        """Number of suggestions with probability 1.0"""
        return sum([1 for model_suggestions in self.suggestions for pr in model_suggestions.values() if pr == 1])

    def get_certain_suggestions(self) -> List:
        """Return all certain suggestions."""
        return [s for model_suggestions in self.suggestions for s, pr in model_suggestions.items() if pr == 1]

    def n_certain_unicode_suggestions(self) -> int:
        """Number of unicode suggestions with probability 1.0"""
        return sum([1 for model_suggestions in self.unicode_suggestions for pr in model_suggestions.values() if pr == 1])

    def get_most_certain_unicode_suggestions(self, threshold: int) -> List[Tuple[str, float]]:
        pass

    def get_rule_based_suggestion(self, d) -> Union[str, None]:
        """
        Cleaning heuristic to determine the best rule-based suggestion. Documentation on this can be found in
        the experiment from 2022W38.
        """
        if self.n_certain_suggestions == 0:  # No certain suggestion at all.
            return None

        if self.n_certain_suggestions == 1:  # Use the one certain suggestion.
            choice = [suggestion for model_suggestions in self.suggestions for suggestion, pr in model_suggestions.items() if pr == 1]
            if len(choice) > 1:
                raise ValueError(f"More than one certain suggestion: {choice}")
            choice = choice[0]

        elif self.n_certain_unicode_suggestions() == 1:  # Use the one certain unicode-encoded suggestion.
            # Das funktioniert in der Praxis nicht perfekt. Ich bekomme auf rayyan z.B. "1/13" als Vorschlag, obwohl ich
            # einen datestring %m/%d/%y erwarte.
            choice = [suggestion for model_suggestions in self.unicode_suggestions for suggestion, pr in model_suggestions.items() if pr == 1]
            if len(choice) > 1:
                raise ValueError(f"More than one certain suggestion: {choice}")
            choice = choice[0]

        else:  # sum up the probabilities of both encodings of all certain corrections and take the max.
            suggestion_sums = {}
            indices, suggestions = self.certain_model_type_indices_and_suggestions
            for i, s in zip(indices, suggestions):
                if i % 2 == 0:
                    corresponding_index = i + 1
                else:
                    corresponding_index = i - 1
                corresponding_score = self.suggestions[corresponding_index].get(s, 0)
                if suggestion_sums.get(s) is None:
                    suggestion_sums[s] = self.suggestions[i][s] + corresponding_score
                else:
                    suggestion_sums[s] += self.suggestions[i][s] + corresponding_score
            choice = max(suggestion_sums, key=suggestion_sums.get)

        error = d.dataframe.iloc[self.cell]
        true_correction = d.clean_dataframe.iloc[self.cell]

        d.debug_rules.append({'suggestions': self.get_certain_suggestions(),
                              'choice': choice,
                              'true_correction': true_correction,
                              'error': error})
        return choice
