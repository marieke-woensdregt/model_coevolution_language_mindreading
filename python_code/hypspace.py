__author__ = 'Marieke Woensdregt'


import itertools
import numpy as np
from lex import Lexicon


def calc_no_lex_hyps(n_meanings, n_signals):
    # The 2 below is hardcoded because we only work with binary association weights: 0 or 1. The -1 below is because a row of all 0s and a row of all 1s are functionally equivalent.
    n_possible_rows = (2**n_signals)-1
    n_possible_lexicons = n_possible_rows**n_meanings
    return n_possible_lexicons




def create_all_lexicons(n_meanings, n_signals):
    """
    :param n_meanings: The number of meanings
    :param n_signals: The number of signals
    :return: A 3D numpy matrix containing all possible lexicons, ranging from each meaning mapping to only one signal to each meaning mapping to each signal, and all possible combinations in between. Lexicons consist of binary mappings between meanings and signals, with meanings on the rows and signals on the columns. Axis 0 = lexicons, axis 1 = meanings, axis 2 = signals.
    """
    all_lexicons = []
    # 1) First we create a list of all possible rows, that contains each possible combination of 1s (c.e. m-s mappings) in the row, ranging from one m-s mapping to all signals mapping to the meaning:
    all_rows = []
    for i in range(1, (n_signals + 1)):
        signal_combis = list(itertools.combinations(range(n_signals), i))
        for signal_combination in signal_combis:
            row = np.zeros(n_signals)
            for index in signal_combination:
                row[index] = 1.
            all_rows.append(row)
    # 2) Then we create a list of all possible combinations of rows (with repetition), where combinations are of length n_meanings:
    row_combis = list(itertools.product(range(len(all_rows)), repeat=n_meanings))
    # 3) Then we turn each of these possible row combinations into an actual lexicon, and we append each new lexicon to the list 'all_lexicons':
    for row_combination in row_combis:
        lexicon = np.zeros((n_meanings, n_signals))
        for i in range(len(row_combination)):
            row_index = row_combination[i]
            lexicon[i] = all_rows[row_index]
        all_lexicons.append(lexicon)
    # 4) We then turn the list 'all_lexicons' into a numpy array:
    all_lexicons = np.asarray(all_lexicons)
    return all_lexicons



def get_sorted_lex_hyp_order(lexicon_matrix):
    # n_meanings = len(lexicon_matrix[0])
    # n_signals = len(lexicon_matrix[0][0])
    # print "n_meanings is:"
    # print n_meanings
    # print "n_signals is:"
    # print n_signals
    # new_order = []
    # unambiguous_lex_sum = np.ones(n_meanings)
    # unambiguous_lex_sum.reshape(1, n_meanings)
    # print unambiguous_lex_sum
    #
    # for lex_index in range(len(lexicon_matrix)):
    #     lexicon = lexicon_matrix[lex_index]
    #     # First we'll get the fully unambiguous lexicons:
    #     if np.sum(lexicon, axis=0) == unambiguous_lex_sum:
    #         new_order.append(lex_index)
    #     # Then the ones with 1 ambiguous mapping:
    #     else:
    #         for row in lexicon:
    #             if

    lex_indices = np.arange(len(lexicon_matrix))
    unambiguous_signals_array = np.zeros(len(lexicon_matrix))
    for lex_index in range(len(lexicon_matrix)):
        lexicon = lexicon_matrix[lex_index]
        column_sums = np.sum(lexicon, axis=0)
        unambiguous_signals = 0
        for signal in column_sums:
            if signal == 1:
                unambiguous_signals += 1
        unambiguous_signals_array[lex_index] = unambiguous_signals
    sorted_lex_index_order_inverse = np.argsort(unambiguous_signals_array)
    sorted_lex_index_order = sorted_lex_index_order_inverse[::-1]
    return sorted_lex_index_order





def remove_duplicate_lexicons(lexicon_array):
    """
    :param lexicon_array: A 3D numpy array filled with lexicons (where lexicons have meanings on the rows and signals on the columns)
    :return: The same lexicon array but with all duplicate lexicons removed.
    """
    for i in range(len(lexicon_array)):
        if i >= len(lexicon_array): # this if-statement is necessary because lexicons are being removed from the lexicon_array as we loop through it
            break
        base_lexicon = lexicon_array[i]
        for j in range(len(lexicon_array)):
            if i != j:
                comparison_lexicon = lexicon_array[j]
                if np.array_equal(base_lexicon, comparison_lexicon):
                    lexicon_array = np.delete(lexicon_array, j, axis=0)
    return lexicon_array



def remove_subset_of_signals_lexicons(lexicon_array):
    """
    :param lexicon_array: A 3D numpy array filled with lexicons (where lexicons have meanings on the rows and signals on the columns)
    :return: The same lexicon array but with all lexicons that don't use all the signals in the system removed
    """
    for i in range(len(lexicon_array)):
        if i >= len(lexicon_array): # this if-statement is necessary because lexicons are being removed from the lexicon_array as we loop through it
            break
        lexicon_transposed = np.transpose(lexicon_array[i])
        for signal_row in lexicon_transposed:
            if np.sum(signal_row) == 0.:
                lexicon_array = np.delete(lexicon_array, i, axis=0)
    return lexicon_array




def create_all_optimal_lexicons(n_meanings, n_signals):
    """
    :param n_meanings: The number of meanings
    :param n_signals: The number of signals
    :return: A 3D numpy array containing all possible OPTIMAL lexicons that can be created with this number of meanings and signals (see the Lexicon class for definition of an 'optimal' lexicon). Axis 0 = lexicons, axis 1 = meanings, axis 2 = signals.
    """
    if n_meanings > n_signals or n_meanings == n_signals:
        optimal_lexicon = Lexicon('optimal_lex', n_meanings, n_signals)
        optimal_lexicon = optimal_lexicon.lexicon
        possible_rows = optimal_lexicon
    if n_meanings < n_signals:
        possible_rows = np.zeros((n_signals, n_signals))
        initial_row = np.zeros(n_signals)
        initial_row[0] = 1.
        for i in range(len(initial_row)):
            new_row = np.roll(initial_row, i)
            possible_rows[i] = new_row
    permutations = list(itertools.permutations(possible_rows, n_meanings))
    n_possible_lexicons = len(permutations)
    all_optimal_lexicons = np.zeros((n_possible_lexicons, n_meanings, n_signals))
    counter = 0
    for row_order in permutations:
        new_lexicon = np.zeros((n_meanings, n_signals))
        for i in range(len(row_order)):
            new_lexicon[i] = row_order[i]
        all_optimal_lexicons[counter] = new_lexicon
        counter += 1
    if n_meanings != n_signals:
        # The code below serves to remove duplicated lexicons, which are created with the procedure above if the number of meanings and signals is not equal (or at least if n_meanings > n_signals).
        #TODO: Check whether this only happens if n_meanings > n_signals but not the other way around
        all_optimal_lexicons = remove_duplicate_lexicons(all_optimal_lexicons)
    return all_optimal_lexicons



#TODO: Think about whether it's possible to do this also for asymmetrical lexicons
def create_mirror_image_lexicon(n_meanings, n_signals):
    """
    :param n_meanings: The number of meanings
    :param n_signals: The number of signals
    :return: A 2D numpy matrix with meanings on the rows and signals on the columns. This lexicon constitutes the exact mirror image of having 1s on the diagonal. (c.e. signal n maps to meaning 0, signal n-1 to meaning 1 etc.)
    """
    if n_meanings != n_signals:
        print "Sorry, the create_system_opposite() function only works with a symmetrical matrix"
    column_indices = np.arange(0, n_signals)
    columns_reversed = column_indices[::-1]
    diagonal_lexicon = np.zeros((n_meanings, n_signals))
    np.fill_diagonal(diagonal_lexicon, 1.)
    mirror_lexicon = np.zeros_like(diagonal_lexicon)
    for i in range(len(mirror_lexicon)):
        mirror_lexicon[i][columns_reversed[i]] = 1.
    return mirror_lexicon





def list_hypothesis_space(perspective_hyps, lexicon_hyps):
    """
    :param perspective_hyps: A 1D numpy array of all perspective hypotheses
    :param lexicon_hyps: A 1D numpy array of all lexicon hypotheses
    :return: A 2D numpy matrix of all possible combinations of perspective hypothesis and lexicon hypothesis. Composite hypotheses are on the rows; column 0 contains the perspective hypothesis, column 1 the lexicon hypothesis.
    """
    perspective_indices = range(len(perspective_hyps))
    lexicon_indices = range(len(lexicon_hyps))
    hypothesis_space = list(itertools.product(perspective_indices, lexicon_indices))
    hypothesis_space = np.asarray(hypothesis_space)
    return hypothesis_space


#TODO: Write this function:
def list_hypothesis_space_with_speaker_distinction(perspective_hyps, lexicon_hyps, n_speakers):
    """
    :param perspective_hyps: A 1D numpy array of all perspective hypotheses
    :param lexicon_hyps: A 1D numpy array of all lexicon hypotheses
    :param n_speakers: The number of speakers
    :return: A 2D numpy matrix of all possible combinations of perspective hypothesis and lexicon hypothesis. Composite hypotheses are on the rows; column 0 contains the perspective hypothesis, column 1 the lexicon hypothesis.
    """
    perspective_indices = range(len(perspective_hyps))
    lexicon_indices = range(len(lexicon_hyps))
    perspective_hyp_space = list(itertools.product(perspective_indices, repeat=n_speakers))
    perspective_hyp_space = np.asarray(perspective_hyp_space)
    hypothesis_space = list(itertools.product(perspective_hyp_space, lexicon_indices))
    hypothesis_space = np.asarray(hypothesis_space)
    return hypothesis_space
