import pandas as pd
from typing import Tuple, List, Dict, Union
from itertools import combinations
from collections import namedtuple, defaultdict

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
) -> Tuple[dict, dict]:
    """
    Calculates a dictionary d that contains the absolute counts of how
    often values in the lhs occur with values in the rhs in the table df.

    The dictionary has the structure
    d[lhs_columns][rhs_column][lhs_values][rhs_value],
    where lhs_columns is a tuple of one or more lhs_columns, and
    rhs_column is the columns whose values are determined by lhs_columns.

    Pass an `order` argument to indicate how many columns in the lhs should
    be investigated. If order=1, only unary relationships are taken into account,
    if order=2, only binary relationships are taken into account, and so on.

    Detected cells is a list of dictionaries whose keys are the coordinates of
    errors in the table. It is used to exclude errors from contributing to the
    value counts. `ignore_sign` is a string that indicates that a cell is erronous,
    too. Cells containing this value are ignored, too.
    """
    i_cols = list(range(df.shape[1]))

    distinct_lhs_values = defaultdict(lambda: defaultdict(set))

    d = {comb: {cc: {} for cc in i_cols} for comb in combinations(i_cols, order)}
    for lhs_cols in d:
        d[lhs_cols]["value_counts"] = defaultdict(int)  # defaultdict with counter starting at 0

    for row in df.itertuples(index=True):
        i_row, row = row[0], row[1:]
        for lhs_cols in combinations(i_cols, order):
            lhs_vals = tuple(row[lhs_col] for lhs_col in lhs_cols)

            lhs_contains_error = any(
                [(i_row, lhs_col) in detected_cells for lhs_col in lhs_cols]
            )
            if ignore_sign not in lhs_vals and not lhs_contains_error:
                # update conditional counts
                for rhs_col in i_cols:
                    if rhs_col not in lhs_cols:
                        rhs_contains_error = (i_row, rhs_col) in detected_cells
                        if ignore_sign not in lhs_vals and not rhs_contains_error:
                            rhs_val = row[rhs_col]
                            distinct_lhs_values[lhs_cols][rhs_col].add(lhs_vals)

                            if d[lhs_cols][rhs_col].get(lhs_vals) is None:
                                d[lhs_cols][rhs_col][lhs_vals] = {}  # dict with counter starting at 0
                            if d[lhs_cols][rhs_col][lhs_vals].get(rhs_val) is None:
                                d[lhs_cols][rhs_col][lhs_vals][rhs_val] = 1.0
                            else:
                                d[lhs_cols][rhs_col][lhs_vals][rhs_val] += 1.0

    return d, distinct_lhs_values


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
    distinct_lhs_values: dict,
    order: int,
    A: Tuple[int, ...],
    B: int,
) -> Union[float, None]:
    pdep_B = pdep_0(n_rows, counts_dict, order, B, A)

    if pdep_B is None:
        return None

    n_distinct_values_A = len(distinct_lhs_values[order][A][B])

    return pdep_B + (n_distinct_values_A - 1) / (n_rows - 1) * (1 - pdep_B)


def error_corrected_row_count(
    n_rows: int,
    detected_cells: Dict[Tuple, str],
    A: Tuple[int, ...],
    B: int = None
) -> int:
    """
    Calculate the number of rows N that do not contain an error in the LHS or RHS.
    @param n_rows: Number of rows in the table including all errors.
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


def pdep_0(
        n_rows: int,
        counts_dict: dict,
        order: int,
        B: int,
        A: Tuple[int, ...]
) -> Union[float, None]:
    """
    Calculate pdep(B), that is the probability that two randomly selected records from B will have the same value.
    Note that in order to calculate pdep(B), you may want to limit the records from B to error-free records in
    context of a left hand side A, for example when calculating pdep(B) to calculate gpdep(A,B).
    """
    if n_rows == 0:  # no tuple exists without an error in lhs and rhs
        return None

    # calculate the frequency of each RHS value, given that the values in all columns covered by LHS and RHS contain
    # no error.
    rhs_abs_frequencies = defaultdict(int)
    for lhs_vals in counts_dict[order][A][B]:
        for rhs_val in counts_dict[order][A][B][lhs_vals]:
            rhs_abs_frequencies[rhs_val] += counts_dict[order][A][B][lhs_vals][rhs_val]

    sum_components = []
    for rhs_frequency in rhs_abs_frequencies.values():
        sum_components.append(rhs_frequency**2)
    return sum(sum_components) / n_rows**2


def pdep(
    n_rows: int,
    counts_dict: dict,
    order: int,
    A: Tuple[int, ...],
    B: int,
) -> Union[float, None]:
    """
    Calculates the probabilistic dependence pdep(A,B) between a left hand side A,
    which consists of one or more attributes, and a right hand side B,
    which consists of one attribute.

    """
    if n_rows == 0:  # no tuple exists without an error in lhs and rhs
        return None

    sum_components = []

    if B in A:  # pdep([A,..],A) = 1
        return 1

    counts_dict = counts_dict[order][A][B]
    for lhs_val, rhs_dict in counts_dict.items():  # lhs_val same as A_i
        lhs_counts = sum(rhs_dict.values())  # same as a_i
        for rhs_val, rhs_counts in rhs_dict.items():  # rhs_counts same as n_ij
            sum_components.append(rhs_counts**2 / lhs_counts)
    return sum(sum_components) / n_rows


def gpdep(
    n_rows: int,
    counts_dict: dict,
    distinct_lhs_values: dict,
    A: Tuple[int, ...],
    B: int,
    order: int,
) -> Union[PdepTuple, None]:
    """
    Calculates the *genuine* probabilistic dependence (gpdep) between
    a left hand side A, which consists of one or more attributes, and
    a right hand side B, which consists of exactly one attribute.
    """
    pdep_A_B = pdep(n_rows, counts_dict, order, A, B)
    epdep_A_B = expected_pdep(n_rows, counts_dict, distinct_lhs_values, order, A, B)

    if pdep_A_B is not None and epdep_A_B is not None:
        gpdep_A_B = pdep_A_B - epdep_A_B
        if gpdep_A_B < 0 or gpdep_A_B > 1:
            a = 1
        return PdepTuple(pdep_A_B, gpdep_A_B)
    return None


def vicinity_based_corrector_order_n(counts_dict, ed, probability_threshold) -> List[Dict[str, int]]:
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
    counts_dict: dict, distinct_lhs_values: dict, df: pd.DataFrame, detected_cells: Dict[Tuple, str], order: int
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
            N = error_corrected_row_count(n_rows, detected_cells, lhs, rhs)
            gpdeps[lhs][rhs] = gpdep(N, counts_dict, distinct_lhs_values, lhs, rhs, order)
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
    n_best_pdeps: int = 3,
    features_selection: tuple = ('pr', 'vote')
) -> Dict:
    """
    Leverage gpdep to avoid having correction suggestion feature columns
    grow in number at (n-1)^2 / 2 pace. Only take the `n_best_pdeps`
    highest-scoring dependencies to draw corrections from.
    """
    if len(features_selection) == 0:  # no features to generate
        return {}

    rhs_col = ed["column"]
    gpdeps = inverse_sorted_gpdeps[rhs_col]

    gpdeps_subset = {rhs: gpdeps[rhs] for i, rhs in enumerate(gpdeps) if i < n_best_pdeps}
    results_list = []

    for lhs_cols, pdep_tuple in gpdeps_subset.items():
        lhs_vals = tuple([ed["vicinity"][x] for x in lhs_cols])

        if rhs_col not in lhs_cols and lhs_vals in counts_dict[lhs_cols][rhs_col]:
            sum_scores = sum(counts_dict[lhs_cols][rhs_col][lhs_vals].values())
            for rhs_val in counts_dict[lhs_cols][rhs_col][lhs_vals]:
                pr = counts_dict[lhs_cols][rhs_col][lhs_vals][rhs_val] / sum_scores

                results_list.append(
                    {"correction": rhs_val,
                     "pr": pr,
                     "pdep": pdep_tuple.pdep if pdep_tuple is not None else 0,
                     "gpdep": pdep_tuple.gpdep if pdep_tuple is not None else 0}
                )

    sorted_results = sorted(results_list, key=lambda x: x["pr"], reverse=True)

    highest_conditional_probabilities = {}

    # Having a sorted dict allows us to only return the highest conditional
    # probability per correction by iterating over all generated corrections
    # like this.
    for d in sorted_results:
        if highest_conditional_probabilities.get(d["correction"]) is None:
            highest_conditional_probabilities[d["correction"]] = d["pr"]

    return highest_conditional_probabilities
