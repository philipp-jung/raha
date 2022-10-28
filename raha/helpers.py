import json
import pandas as pd
from typing import Tuple, List, Dict, Union
from Levenshtein import distance as levenshtein_distance


def lev_to_original_error(error_cell: Tuple[int, int],
                          old_error_cells: List[Tuple[int, int]],
                          df_dirty: pd.DataFrame) -> List[float]:
    """
    Bei mehreren unicode-encodeten Suggestions müssen wir eine Suggestion auswählen. Unser Maß hierzu ist
    lev(ursprünglicher Fehler, Fehler aus dem die Regel erstellt wurde). Allgemein gibt es mehrere Fehler, aus denen
    die Regel stammt.
    @param error_cell: Coordinates of the error currently being corrected.
    @param old_error_cells: List of Coordinates of the errors whose corrections were used to create the correcting rule.
    @param df_dirty: Table that is being cleaned.
    @return: Mean levenshtein distance between the error being corrected and the errors used to create the correcting
    rule.
    """
    distances = []
    error_value = df_dirty.iloc[error_cell]
    for old_error_cell in old_error_cells:
        old_error_value = df_dirty.iloc[old_error_cell]
        distances.append(levenshtein_distance(error_value, old_error_value))
    return distances


def lev_to_original_corrections(correction_suggestion: str,
                                old_error_cells: List[Tuple[int, int]],
                                labeled_cells: Dict[Tuple[int, int], List]) -> List[float]:
    """
    Wenn lev_to_original_error keine minimale Distanz birgt, berechnen wir die Levenshtein-Distanz zwischen der ur-
    sprünglichen Korrektur, der zur Operation geführt hat, und dem Korrekturvorschlag.
    @param correction_suggestion: Suggestion to correct the current error
    @param old_error_cells: List of Coordinates of the errors whose corrections were used to create the correcting rule
    @param labeled_cells: Dictionary containing user input
    @return: List of levenshtein distances.
    """
    distances = []
    old_corrections = [labeled_cells[cell] for cell in old_error_cells]
    for old_correction in old_corrections:
        distances.append(levenshtein_distance(correction_suggestion, old_correction))
    return distances


def assemble_cleaning_suggestion(transformation_string: str, model_name: str, old_value: str) -> Union[str, None]:
    """
    Use the operation encoded in transform_string and the model_name to identify the operation to transform old_value
    into a cleaning suggestion
    @param transformation_string: the encoded transformation.
    @param model_name: operation name, which is adder, remover, replacer or swapper.
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
