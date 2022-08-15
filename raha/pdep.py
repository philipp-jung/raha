import pandas as pd
from typing import Tuple, List, Dict, Union, Any
from itertools import combinations
from collections import Counter, namedtuple

PdepTuple = namedtuple("PdepTuple", ["pdep", "gpdep"])


def calculate_frequency(df: pd.DataFrame, col: int):
    """
    Calculates the frequency of a value to occur in colum
    col on dataframe df.
    """
    counts = df.iloc[:, col].value_counts()
    return counts.to_dict()


def calculate_counts_dict(
    df: pd.DataFrame,
    detected_cells: Dict[Tuple, str],
    order=1,
    ignore_sign="<<<IGNORE_THIS_VALUE>>>",
) -> dict:
    """
    Calculates a dictionary d that contains the absolute counts of how
    often values in the lhs occur with values in the rhs in the table df.

    The dictionary has the structure
    d[lhs_columns][rhs_column][lhs_values][rhs_value],
    where lhs_columns is a tuple of one or more lhs_columns, and
    rhs_column is the columns whose values are determined by lhs_columns.

    Pass an `order` argument to indicate how many columns in the lhs should
    be investigated. If order=1, only unary relationships are taken into account,
    if order=2, only binary relationships are taken ito account, and so on.

    Detected cells is a list of dictionaries whose keys are the coordinates of
    errors in the table. It is used to exclude errors from contributing to the
    value counts.

    ignore_sign is a string that marks a cell's value as being erroneous. This
    way that value is excluded from the value counts, too.
    """
    i_cols = list(range(df.shape[1]))
    d = {comb: {cc: {} for cc in i_cols} for comb in combinations(i_cols, order)}
    for lhs_cols in d:
        d[lhs_cols]["value_counts"] = {}

    for row in df.itertuples(index=True):
        i_row, row = row[0], row[1:]
        for lhs_cols in combinations(i_cols, order):
            lhs_vals = tuple(row[lhs_col] for lhs_col in lhs_cols)

            any_cell_contains_error = any(
                [(i_row, lhs_col) in detected_cells for lhs_col in lhs_cols]
            )
            if ignore_sign not in lhs_vals and not any_cell_contains_error:
                # increase counts of values in the LHS, accessed via d[lhs_columns]['value_counts']
                if d[lhs_cols]["value_counts"].get(lhs_vals) is None:
                    d[lhs_cols]["value_counts"][lhs_vals] = 1.0
                else:
                    d[lhs_cols]["value_counts"][lhs_vals] += 1.0

                # update conditional counts
                for rhs_col in i_cols:
                    if rhs_col not in lhs_cols:
                        rhs_contains_error = (i_row, rhs_col) in detected_cells
                        if ignore_sign not in lhs_vals and not rhs_contains_error:
                            if d[lhs_cols][rhs_col].get(lhs_vals) is None:
                                d[lhs_cols][rhs_col][lhs_vals] = {}
                            rhs_val = row[rhs_col]
                            if d[lhs_cols][rhs_col][lhs_vals].get(rhs_val) is None:
                                d[lhs_cols][rhs_col][lhs_vals][rhs_val] = 1.0
                            else:
                                d[lhs_cols][rhs_col][lhs_vals][rhs_val] += 1.0
    return d


def update_counts_dict(
    df: pd.DataFrame, counts_dict: dict, order: int, cleaned_sampled_tuple: list
) -> None:
    """
    This function is created to fit Baran's style of working. It Updates a
    counts dict in place with a newly labeled row.
    """
    i_cols = list(range(df.shape[1]))

    for lhs_cols in combinations(i_cols, order):
        lhs_vals = tuple(cleaned_sampled_tuple[lhs_col] for lhs_col in lhs_cols)

        # Updates counts of values in the LHS
        if counts_dict[lhs_cols]["value_counts"].get(lhs_vals) is None:
            counts_dict[lhs_cols]["value_counts"][lhs_vals] = 1.0
        else:
            counts_dict[lhs_cols]["value_counts"][lhs_vals] += 1.0

        # Updates conditional counts
        for rhs_col in i_cols:
            if rhs_col not in lhs_cols:
                if counts_dict[lhs_cols][rhs_col].get(lhs_vals) is None:
                    counts_dict[lhs_cols][rhs_col][lhs_vals] = {}
                rhs_val = cleaned_sampled_tuple[rhs_col]
                if counts_dict[lhs_cols][rhs_col][lhs_vals].get(rhs_val) is None:
                    counts_dict[lhs_cols][rhs_col][lhs_vals][rhs_val] = 1.0
                else:
                    counts_dict[lhs_cols][rhs_col][lhs_vals][rhs_val] += 1.0


def expected_pdep(
    n_rows: int,
    counts_dict: dict,
    detected_cells: Dict[Tuple, str],
    order: int,
    A: Tuple[int, ...],
    B: int,
) -> Union[float, None]:
    pdep_B = pdep(n_rows, counts_dict, detected_cells, order, tuple([B]))

    if pdep_B is None:
        return None

    if len(A) == 1:
        n_distinct_values_A = len(counts_dict[1][A]["value_counts"])
    elif len(A) > 1:
        n_distinct_values_A = len(counts_dict[order][A][B])
    else:
        raise ValueError("A needs to contain one or more attribute names")

    return pdep_B + (n_distinct_values_A - 1) / (n_rows - 1) * (1 - pdep_B)


def error_corrected_row_count(
    n_rows: int, detected_cells: Dict[Tuple, str], A: Tuple[int, ...], B: int = None
) -> int:
    """
    Calculate the number of rows N that do not contain an error in the LHS or RHS.
    @param A: LHS column
    @param B: RHS column
    @param detected_cells: Dictionary with key error position, value correct value
    @return: Number of rows without an error
    """
    relevant_cols = list(A) if B is None else list(A) + [B]
    excluded_rows = []
    for row, col in detected_cells:
        if col in relevant_cols and row not in excluded_rows:
            excluded_rows.append(row)
    return n_rows - len(excluded_rows)


def pdep(
    n_rows: int,
    counts_dict: dict,
    detected_cells: Dict[Tuple, str],
    order: int,
    A: Tuple[int, ...],
    B: int = None,
) -> Union[float, None]:
    """
    Calculates the probabilistic dependence (pdep) between a left hand side A,
    which consists of one or more attributes, and an optional right hand side B,
    which consists of one attribute.

    If B is None, calculate pdep(A), that is the probability that two randomly
    selected records from A will have the same value.
    """
    N = error_corrected_row_count(n_rows, detected_cells, A, B)
    if N == 0:  # no tuple exists without an error in lhs and rhs
        return None

    sum_components = []

    if B is not None:  # pdep(A,B)
        counts_dict = counts_dict[order][A][B]
        for lhs_val, rhs_dict in counts_dict.items():  # lhs_val same as A_i
            lhs_counts = sum(rhs_dict.values())  # same as a_i
            for rhs_val, rhs_counts in rhs_dict.items():  # rhs_counts same as n_ij
                sum_components.append(rhs_counts**2 / lhs_counts)
        return sum(sum_components) / N

    elif len(A) == 1:  # pdep(A)
        counts_dict = counts_dict[1]
        counts_dict = counts_dict[A]["value_counts"]
        for lhs_val, lhs_rel_frequency in counts_dict.items():
            sum_components.append(lhs_rel_frequency**2)
        return sum(sum_components) / N**2

    else:
        raise ValueError(
            "Wrong data type for A or B, or wrong order. A "
            "should be a tuple of a list of column names of df, "
            "B should be name of a column or None. If B is None,"
            " order must be 1."
        )


def gpdep(
    n_rows: int,
    counts_dict: dict,
    detected_cells: Dict[Tuple, str],
    A: Tuple[int, ...],
    B: int,
    order: int,
) -> Union[PdepTuple, None]:
    """
    Calculates the *genuine* probabilistic dependence (gpdep) between
    a left hand side A, which consists of one or more attributes, and
    a right hand side B, which consists of exactly one attribute.
    """
    pdep_A_B = pdep(n_rows, counts_dict, detected_cells, order, A, B)
    epdep_A_B = expected_pdep(n_rows, counts_dict, detected_cells, order, A, B)

    if pdep_A_B is not None and epdep_A_B is not None:
        gpdep_A_B = pdep_A_B - epdep_A_B
        return PdepTuple(pdep_A_B, gpdep_A_B)
    return None


def vicinity_based_corrector_order_n(counts_dict, ed, probability_threshold):
    """
    Use Baran's original strategy to suggest corrections based on higher-order
    vicinity.

    Features generated by this function grow with (n-1)^2 / 2, where n is the
    number of columns of the dataframe. This corrector can be considered to be
    the naive approach to suggesting corrections from higher-order vicinity.

    Notes:
    ed: {   'column': column int,
            'old_value': old error value,
            'vicinity': row that contains the error, including the error
    }
    counts_dict: Dict[Dict[Dict[Dict]]] [lhs][rhs][lhs_value][rhs_value]

    """
    rhs_col = ed["column"]
    results_list = []

    for lhs_cols in list(counts_dict.keys()):
        results_dictionary = {}
        for lhs_vals in combinations(ed["vicinity"], len(lhs_cols)):
            if rhs_col not in lhs_cols and lhs_vals in counts_dict[lhs_cols][rhs_col]:
                sum_scores = sum(counts_dict[lhs_cols][rhs_col][lhs_vals].values())
                for rhs_val in counts_dict[lhs_cols][rhs_col][lhs_vals]:
                    pr = counts_dict[lhs_cols][rhs_col][lhs_vals][rhs_val] / sum_scores
                    if pr >= probability_threshold:
                        results_dictionary[rhs_val] = pr
        results_list.append(results_dictionary)
    return results_list


def calc_all_gpdeps(
    counts_dict: dict, df: pd.DataFrame, detected_cells: Dict[Tuple, str], order: int
) -> Dict[Tuple, Dict[int, PdepTuple]]:
    """
    Calculate all gpdeps in dataframe df, with an order implied by the depth
    of counts_dict.
    """
    n_rows, n_cols = df.shape
    lhss = set([x for x in counts_dict[order].keys()])
    rhss = list(range(n_cols))
    gpdeps = {lhs: {} for lhs in lhss}
    for lhs in lhss:
        for rhs in rhss:
            gpdeps[lhs][rhs] = gpdep(
                n_rows, counts_dict, detected_cells, lhs, rhs, order
            )
    return gpdeps


def invert_and_sort_gpdeps(
    gpdeps: Dict[Tuple, Dict[int, PdepTuple]]
) -> Dict[int, Dict[Tuple, PdepTuple]]:
    """
    Invert the gpdeps dict and sort it. Results in a dict whose first key is
    the rhs, second key is the lhs, value of that is the gpdep score. The second
    key is sorted by the descending gpdep score.
    """
    inverse_gpdeps = {rhs: {} for rhs in list(gpdeps.items())[0][1]}

    for lhs in gpdeps:
        for rhs in gpdeps[lhs]:
            inverse_gpdeps[rhs][lhs] = gpdeps[lhs][rhs]

    for rhs in inverse_gpdeps:
        inverse_gpdeps[rhs] = {
            k: v
            for k, v in sorted(
                inverse_gpdeps[rhs].items(),
                key=lambda x: (x[1] is not None, x[1].gpdep if x[1] is not None else None),
                reverse=True,
            )
        }
    return inverse_gpdeps


def pdep_vicinity_based_corrector(
    inverse_sorted_gpdeps: Dict[int, Dict[tuple, PdepTuple]],
    counts_dict: dict,
    ed: dict,
    n_best_pdeps: int = 5,
    use_pdep_feature: bool = True
) -> List[Dict]:
    """
    Leverage gpdep to avoid having correction suggestion feature columns
    grow in number at (n-1)^2 / 2 pace. Only take the `n_best_pdeps`
    highest-scoring dependencies to draw corrections from.
    """
    rhs_col = ed["column"]
    gpdeps = inverse_sorted_gpdeps[rhs_col]

    gpdeps_subset = {
        rhs: gpdeps[rhs] for i, rhs in enumerate(gpdeps) if i < n_best_pdeps
    }
    results_list = []

    for lhs_cols, pdep_tuple in gpdeps_subset.items():
        lhs_vals = tuple([ed["vicinity"][x] for x in lhs_cols])

        if not rhs_col in lhs_cols and lhs_vals in counts_dict[lhs_cols][rhs_col]:
            sum_scores = sum(counts_dict[lhs_cols][rhs_col][lhs_vals].values())
            for rhs_val in counts_dict[lhs_cols][rhs_col][lhs_vals]:
                pr = counts_dict[lhs_cols][rhs_col][lhs_vals][rhs_val] / sum_scores

                results_list.append(
                    {"correction": rhs_val, "pr": pr, "pdep": pdep_tuple.pdep if pdep_tuple is not None else 0}
                )

    sorted_results = sorted(results_list, key=lambda x: x["pr"], reverse=True)

    highest_conditional_probabilities = {}
    highest_pdep_scores = {}
    votes = {}

    # Having a sorted dict allows us to only return the highest conditional
    # probability per correction by iterating over all generated corrections
    # like this.
    # I get the highest gpdep score per correction this way, too.
    for d in sorted_results:
        if highest_conditional_probabilities.get(d["correction"]) is None:
            highest_conditional_probabilities[d["correction"]] = d["pr"]
            highest_pdep_scores[d["correction"]] = d["pdep"]

    # The three elements of our decision are majority vote, highest conditional
    # probability, and highest gpdep score.
    # Here, I calculate the majority vote.
    counts = Counter([x["correction"] for x in sorted_results])
    n_corrections = sum(counts.values())

    for correction in counts:
        votes[correction] = counts[correction] / n_corrections

    if not use_pdep_feature:
        return [highest_conditional_probabilities, votes]
    return [highest_conditional_probabilities, highest_pdep_scores, votes]
