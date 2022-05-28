import pandas as pd
from typing import Tuple, List, Dict, Union, Any
from itertools import combinations
from IPython.core.debugger import set_trace


def calculate_frequency(df:pd.DataFrame, col:int):
    """
    Calculates the frequency of a value to occur in colum
    col on dataframe df.
    """
    counts = df.iloc[:, col].value_counts()
    return counts.to_dict()

def calculate_counts_dict(df: pd.DataFrame,
        detected_cells: List[dict],
        order=1,
        ignore_sign="<<<IGNORE_THIS_VALUE>>>") -> dict:
    """
    Calculates a dictionary d that countains the absolute counts of how
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
    d = {comb: {cc: {} for cc in i_cols}
         for comb in combinations(i_cols, order)}

    for row in df.itertuples(index=True):
        i_row, row = row[0], row[1:]
        for lhs_cols in combinations(i_cols, order):
            for rhs_col in i_cols:
                if rhs_col not in lhs_cols:
                    lhs_vals = tuple(row[lhs_col] for lhs_col in lhs_cols)
                    if ignore_sign not in lhs_vals and (i_row, rhs_col) not in detected_cells:
                        if d[lhs_cols][rhs_col].get(lhs_vals) is None:
                            d[lhs_cols][rhs_col][lhs_vals] = {}
                        rhs_val = row[rhs_col]
                        if d[lhs_cols][rhs_col][lhs_vals].get(rhs_val) is None:
                            d[lhs_cols][rhs_col][lhs_vals][rhs_val] = 1.0
                        else:
                            d[lhs_cols][rhs_col][lhs_vals][rhs_val] += 1.0
    return d

def update_counts_dict(df: pd.DataFrame,
        counts_dict: dict,
        order: int,
        cleaned_sampled_tuple: list) -> None:
    """
    This function is created to fit Baran's style of working. It Updates a
    counts dict in place with a newly labeled row.
    """
    i_cols = list(range(df.shape[1]))
    for lhs_cols in combinations(i_cols, order):
        for rhs_col in i_cols:
            lhs_vals = tuple(cleaned_sampled_tuple[lhs_col] for lhs_col
                    in lhs_cols)
            if rhs_col not in lhs_cols:
                if counts_dict[lhs_cols][rhs_col].get(lhs_vals) is None:
                    counts_dict[lhs_cols][rhs_col][lhs_vals] = {}
                rhs_val = cleaned_sampled_tuple[rhs_col]
                if counts_dict[lhs_cols][rhs_col][lhs_vals].get(rhs_val) is None:
                    counts_dict[lhs_cols][rhs_col][lhs_vals][rhs_val] = 1.0
                else:
                    counts_dict[lhs_cols][rhs_col][lhs_vals][rhs_val] += 1.0

def expected_pdep(df, counts_dict: dict, A: Tuple[List[int]], B: int):
    pdep_B = calc_pdep(df, counts_dict, tuple([B]))

    if len(A) == 1:
        n_distinct_values_A = df.iloc[:, A[0]].nunique(dropna=False)
    elif len(A) > 1:
        n_distinct_values_A = len(counts_dict[A][B])
    else:
        raise ValueError('A needs to contain one or more attribute names')

    return pdep_B + (n_distinct_values_A - 1) / (df.shape[0] - 1) * (1 - pdep_B)

def calc_pdep(df: pd.DataFrame, counts_dict: dict, A: Tuple[List[int]], B: int = None):
    """
    Calculates the probabilistic dependence (pdep) between a left hand side A,
    which consists of one or more attributes, and a right hand side B, which
    consists of one or zero attributes.

    If B is None, calculate pdep(A), which is the probability that two randomly
    selected records will have the same value for A.
    """
    N = df.shape[0]
    sum_components = []

    if B is not None:  # pdep(A,B)
        counts_dict = counts_dict[A][B]
        for lhs_val, rhs_dict in counts_dict.items(): # lhs_val same as A_i
            lhs_counts = sum(rhs_dict.values()) # same as a_i
            for rhs_val, rhs_counts in rhs_dict.items(): # rhs_counts same as n_ij
                sum_components.append(rhs_counts**2 / lhs_counts)
        return sum(sum_components) / N

    elif len(A) == 1:  # pdep(A)
        counts_dict = calculate_frequency(df, A[0])
        for lhs_val, lhs_rel_frequency in counts_dict.items():
            sum_components.append(lhs_rel_frequency**2)
        return sum(sum_components) / N**2

    else:
        raise ValueError('Wrong data type for A or B, nor wrong order. A '
                         'should be a tuple of a list of column names of df, '
                         'B should be name of a column or None. If B is none,'
                         ' order must be 1.')

def calc_gpdep(df: pd.DataFrame, counts_dict: dict, A: Tuple[List[int]], B: int):
    """
    Calculates the *genuine* probabilistic dependence (gpdep) between
    a left hand side A, which consists of one or more attributes, and
    a right hand side B, which consists of exactly one attribute.
    """
    return calc_pdep(df, counts_dict, A, B) - expected_pdep(df, counts_dict, A, B)


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


def calc_all_gpdeps(counts_dict: dict, df: pd.DataFrame) -> Dict[Tuple, Dict[int, float]]:
    """
    Calculate all gpdeps in dataframe df, with an order implied by the depth
    of counts_dict.
    """
    lhss = set([x for x in counts_dict.keys()])
    rhss = list(range(df.shape[1]))
    gpdeps = {lhs: {} for lhs in lhss}
    for lhs in lhss:
        for rhs in rhss:
            gpdeps[lhs][rhs] = calc_gpdep(df, counts_dict, lhs, rhs)
    return gpdeps


def invert_and_sort_gpdeps(gpdeps: Dict[Tuple, Dict[int, float]]
        ) -> Dict[int, Dict[Tuple, float]]:
    """
    Invert the gpdeps dict and sort it. Results in a dict whose first key is
    the rhs, second key is the lhs, value of that is the gpdep score. The second
    key is sorted in descendingly by the gpdep score.
    """
    inverse_gpdeps = {rhs: {} for rhs in list(gpdeps.items())[0][1]}

    for lhs in gpdeps:
        for rhs in gpdeps[lhs]:
            inverse_gpdeps[rhs][lhs] = gpdeps[lhs][rhs]

    for rhs in inverse_gpdeps:
        inverse_gpdeps[rhs] = {k: v for k, v in sorted(inverse_gpdeps[rhs].items(),
            key=lambda item: item[1],
            reverse=True)}
    return inverse_gpdeps

def pdep_vicinity_based_corrector(
        inverse_sorted_gpdeps: Dict[int, Dict[tuple, float]],
        counts_dict: dict,
        ed: dict,
        n_best_pdeps: int = 5,
        scoring_strategy: str = 'penalty') -> List[Dict[str, float]]:
    """
    Leverage gpdep to avoid having correction suggestion feature columns
    grow in number at (n-1)^2 / 2 pace. Only take the `n_best_pdeps`
    highest-scoring dependencies to draw corrections from.
    """
    rhs_col = ed["column"]
    gpdeps = inverse_sorted_gpdeps[rhs_col]
    gpdeps_subset = {lhs: gpdeps[lhs] for i, lhs in enumerate(gpdeps) if i < n_best_pdeps}
    max_gpdep_score = list(gpdeps_subset.items())[0][1]

    results_list = []

    for lhs_cols, gpdep_score in gpdeps_subset.items():
        penalty = max_gpdep_score - gpdep_score
        lhs_vals = tuple([ed['vicinity'][x] for x in lhs_cols])

        if not rhs_col in lhs_cols and \
        lhs_vals in counts_dict[lhs_cols][rhs_col]:
            sum_scores = sum(counts_dict[lhs_cols][rhs_col][lhs_vals].values())
            for rhs_val in counts_dict[lhs_cols][rhs_col][lhs_vals]:
                pr = counts_dict[lhs_cols][rhs_col][lhs_vals][rhs_val] / sum_scores
                if scoring_strategy == 'pr_only' or scoring_strategy == 'new_feature':
                    score = pr
                elif scoring_strategy == 'multiply':
                    score = pr * gpdep_score
                elif scoring_strategy == 'penalty':
                    score = pr * (1 - penalty)
                else:
                    raise ValueError('No valid feature strategy specified')

                results_list.append({'correction': rhs_val,
                                     'score': score})
                if scoring_strategy == 'new_feature':  # create new feature for gpdep score
                    results_list.append({'correction': rhs_val,
                                         'score': gpdep_score})

    sorted_results = sorted(results_list, key=lambda x: x['score'], reverse=True)
    correction_suggestions = {}

    # Having a sorted dict allows us to only return the highest score per
    # correction by iterating over all generated corrections like this.
    for d in sorted_results:
        if correction_suggestions.get(d['correction']) is None:
            correction_suggestions[d['correction']] = d['score']

    return [correction_suggestions]
