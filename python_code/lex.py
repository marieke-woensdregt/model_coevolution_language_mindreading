__author__ = 'Marieke Woensdregt'



import numpy as np




class Lexicon(object):
    """A lexicon object consists of a 2D numpy matrix containing mappings between meanings (rows / axis 0) and signals (columns / axis 1), cf. the :return: part of __init__ docstring."""
    def __init__(self, lex_type, n_meanings, n_signals, ambiguous_lex=None, specified_lexicon=None):
        """
        :param lex_type: The type of lexicon (either 'empty_lex', 'optimal_lex', 'half_ambiguous_lex' or 'fully_ambiguous_lex')
        :param n_meanings: Number of meanings
        :param n_signals: Number of signals
        :return: Creates a lexicon of type lex_type with n_meanings as the number of meanings and n_signals as the number of signals. Each lexicon is a 2D numpy array with meanings on the rows and signals on the columns. Each cell in the matrix has a binary value (0 or 1) that determines whether a meaning and signal are associated or not.
        """
        self.lex_type = lex_type
        self.n_meanings = n_meanings
        self.n_signals = n_signals
        self.ambiguous_lex = ambiguous_lex
        self.specified_lexicon = specified_lexicon
        self.lexicon = self.create_lexicon()


    def empty_lex(self):
        """
        :return: An empty lexicon (c.e. all m-s mappings are 0.0)
        """
        lexicon = np.zeros((self.n_meanings, self.n_signals))
        return lexicon

    def optimal_lex(self):
        """
        :return: An optimal lexicon in which each meaning is associated with just 1 signal, distributing the available signals over the meanings as evenly as possible (c.e. avoiding homonymy)
        """
        lexicon = self.empty_lex()
        if self.n_meanings > self.n_signals:
            counter = 0
            for i in range(self.n_meanings):
                if i % self.n_signals == 0:
                    counter = 0
                for j in range(self.n_signals):
                    if counter == j:
                        lexicon[i][j] = 1.
                counter += 1
        else:
            np.fill_diagonal(lexicon, 1.)
        return lexicon

    def half_ambiguous_lex(self):
        """
        :return: A 'half ambiguous' lexicon. This is an optimal lexicon (see method above) in which HALF of the other possible m-s mappings are also set to 1.0. If this half of the other possible mappings is an uneven number, it is rounded down.
        """
        lexicon = self.optimal_lex()
        zeros_coordinates = np.where(lexicon==0.)
        zeros_coordinates = np.vstack((zeros_coordinates[0], zeros_coordinates[1]))
        n_zeros = len(zeros_coordinates[0])
        half_zeros = int(n_zeros/2.)
        mapping_coordinates = np.random.choice(np.arange(n_zeros), half_zeros, replace=False)
        for coordinate in mapping_coordinates:
            first_coordinate = zeros_coordinates[0][coordinate]
            second_coordinate = zeros_coordinates[1][coordinate]
            lexicon[first_coordinate][second_coordinate] = 1.
        return lexicon

    def fully_ambiguous_lex(self):
        """
        :return: A lexicon in which all meanings are mapped to all signals (c.e. all m-s mappings are 1.0)
        """
        lexicon = np.ones((self.n_meanings, self.n_signals))
        return lexicon

    def mirror_of_optimal_lex(self):
        """
        :return: This function creates the mirror image of the optimal_lex by flipping all the rows left-right (which means that it might not give the exact mirror image in cases of asymmetrical lexicons)
        """
        optimal_lex = self.optimal_lex()
        mirror_lex = np.fliplr(optimal_lex)
        return mirror_lex

    def mirror_of_ambiguous_lex(self, ambiguous_lex):
        """
        :return: This function creates the mirror image of the optimal_lex by flipping all the rows left-right (which means that it might not give the exact mirror image in cases of asymmetrical lexicons)
        """
        mirror_lex = np.flipud(ambiguous_lex.lexicon)
        return mirror_lex

    def create_lexicon(self):
        """
        :return: Creates the lexicon, depending on attribute self.lex_type
        """
        if self.lex_type == 'empty_lex':
            lexicon = self.empty_lex()
        elif self.lex_type == 'optimal_lex':
            lexicon = self.optimal_lex()
        elif self.lex_type == 'half_ambiguous_lex':
            lexicon = self.half_ambiguous_lex()
        elif self.lex_type == 'fully_ambiguous_lex':
            lexicon = self.fully_ambiguous_lex()
        elif self.lex_type == 'mirror_of_optimal_lex':
            lexicon = self.mirror_of_optimal_lex()
        elif self.lex_type == 'mirror_of_ambiguous_lex':
            lexicon = self.mirror_of_ambiguous_lex(self.ambiguous_lex)
        elif self.lex_type == 'specified_lexicon':
            lexicon = self.specified_lexicon
        return lexicon

    def print_lexicon(self):
        """
        :return: Prints the lexicon (doesn't return anything)
        """
        print 
        print self.lexicon
        print 




### CATEGORISING LEXICONS BY INFORMATIVENESS BELOW:


def calc_lex_informativity(lexicon, error_prob):
    signal_probs = np.zeros_like(lexicon)
    for m in range(len(signal_probs)):
        meaning_row = lexicon[m]
        new_signal_probs = np.zeros_like(meaning_row)
        associated_signals_indices = np.where(meaning_row == 1.)[0]
        unassociated_signals_indices = np.where(meaning_row == 0.)[0]
        new_signal_probs[associated_signals_indices] = np.divide((1. - error_prob), len(associated_signals_indices))
        new_signal_probs[unassociated_signals_indices] = np.divide(error_prob, len(unassociated_signals_indices))
        if len(unassociated_signals_indices) == 0:
            new_signal_probs[associated_signals_indices] += np.divide(error_prob, len(associated_signals_indices))
        signal_probs[m] = new_signal_probs
    joint_probs = np.divide(signal_probs, np.sum(signal_probs))
    column_sums = joint_probs.sum(axis=0)
    row_sums = joint_probs.sum(axis=1)
    mutual_info_per_signal = np.zeros_like(lexicon)
    for m in range(len(joint_probs)):
        for s in range(len(meaning_row)):
            joint_prob_s = joint_probs[m][s]
            mutual_info_s = joint_prob_s * np.log2(joint_prob_s / (row_sums[m] * column_sums[s]))
            mutual_info_per_signal[m][s] = mutual_info_s
    mutual_info_lex = np.sum(mutual_info_per_signal)
    return mutual_info_lex






def calc_prod_probs(lexicon, error_prob):
    prod_probs = np.zeros_like(lexicon)
    for m in range(len(lexicon)):
        meaning_row = lexicon[m]
        new_signal_probs = np.zeros_like(meaning_row)
        associated_signals_indices = np.where(meaning_row == 1.)[0]
        unassociated_signals_indices = np.where(meaning_row == 0.)[0]
        new_signal_probs[associated_signals_indices] = np.divide((1. - error_prob), len(associated_signals_indices))
        new_signal_probs[unassociated_signals_indices] = np.divide(error_prob, len(unassociated_signals_indices))
        if len(unassociated_signals_indices) == 0:
            new_signal_probs[associated_signals_indices] += np.divide(error_prob, len(associated_signals_indices))
        prod_probs[m] = new_signal_probs
    return prod_probs




def calc_rec_probs(lexicon, error_prob):
    """
    :param lexicon: 2D numpy array with meanings on the rows and signals on the columns (expect only binary association weights between meanings and signals, c.e. 0 or 1!)
    :param error_prob: probability of making an error in production
    :return: reception probabilities with SIGNALS ON THE ROWS AND MEANINGS ON THE COLUMNS. So rows sum to 1.
    """
    rec_probs = np.zeros_like(lexicon)
    t_lexicon = lexicon.T
    for s in range(len(t_lexicon)):
        signal_column = t_lexicon[s]
        new_meaning_probs = np.zeros_like(signal_column)
        associated_meanings_indices = np.where(signal_column == 1.)[0]
        unassociated_meanings_indices = np.where(signal_column == 0.)[0]
        new_meaning_probs[associated_meanings_indices] = np.divide((1. - error_prob), len(associated_meanings_indices))
        new_meaning_probs[unassociated_meanings_indices] = np.divide(error_prob, len(unassociated_meanings_indices))
        if len(unassociated_meanings_indices) == 0:
            new_meaning_probs[associated_meanings_indices] += np.divide(error_prob, len(associated_meanings_indices))
        if len(unassociated_meanings_indices) == len(lexicon):
            new_meaning_probs = [1. / len(lexicon) for x in range(len(lexicon))]
        rec_probs[s] = new_meaning_probs
    return rec_probs



def calc_ca_single_lex(lexicon, error_prob):
    prod_probs = calc_prod_probs(lexicon, error_prob)
    rec_probs = calc_rec_probs(lexicon, error_prob)
    # Below the reception probs have to be transposed because the calc_rec_probs() function returns signals on the rows and meanings on the columns.
    prod_times_rec = np.multiply(prod_probs, rec_probs.T)
    sum_over_meanings = np.sum(prod_times_rec, axis=1)
    avg_ca_lex = np.mean(sum_over_meanings)
    return avg_ca_lex


def calc_ca_all_lexicons(lexicon_hyps, error_prob, lex_measure):
    ca_per_lexicon = np.zeros(len(lexicon_hyps))
    for l in range(len(lexicon_hyps)):
        lexicon = lexicon_hyps[l]
        if lex_measure == 'mi':
            ca_lex = calc_lex_informativity(lexicon, error_prob)
        elif lex_measure == 'ca':
            ca_lex = calc_ca_single_lex(lexicon, error_prob)
        ca_per_lexicon[l] = ca_lex
    return ca_per_lexicon


def create_lex_indices_per_inf_value_sorted_dict(informativity_per_lexicon_sorted, unique_inf_values):
    lex_indices_per_inf_value_sorted_dict = {}
    for inf_value in unique_inf_values:
        # TODO: Why don't I just use np.where() here? I think the output of that function IS suitable for indexing (as opposed to np.argwhere)
        inf_value_indices = np.argwhere(informativity_per_lexicon_sorted == inf_value)
        inf_value_indices = inf_value_indices.flatten()
        lex_indices_per_inf_value_sorted_dict[str(inf_value)] = inf_value_indices
    return lex_indices_per_inf_value_sorted_dict