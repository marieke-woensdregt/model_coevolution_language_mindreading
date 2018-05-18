__author__ = 'Marieke Woensdregt'


import numpy as np
import itertools
from math import factorial


np.set_printoptions(threshold=np.nan)


#######################################################################################################################
# STEP 5: The class below constitutes the data_dict:


class Data(object):
    """
    A data_dict object simply consists of two attributes: 1) a 2D numpy matrix of contexts and 2) a 2D numpy matrix of corresponding speaker utterances, where the rows with matching indices of these two matrices belong together.
    """
    #TODO: Shouldn't this class take as input a speaker or population of speakers and then create the utterance_matrix_same_first?
    def __init__(self, context_matrix, topic_matrix, utterance_matrix):
        """
        :param context_matrix: A 2D numpy matrix of contexts
        :param topic_matrix: A 2D numpy matrix of topics (i.e. intended meanings)
        :param utterance_matrix: A 2D numpy matrix of utterances
        :return: Creates a data_dict object with initial context matrix attribute and utterance_matrix_same_first attribute
        """
        self.contexts = context_matrix
        self.topics = topic_matrix
        self.utterances = utterance_matrix

    def print_data(self):
        """
        :return: Prints the data_dict on separate lines with each context (with index) followed by the corresponding array of utterances (with index). (Does not return anything.)
        """
        for i in range(len(self.contexts)):
            print 'context '+str(i)+' is:'
            print self.contexts[i]
            print 'topics '+str(i)+' is:'
            print self.topics[i]
            print 'utterances '+str(i)+' is:'
            print self.utterances[i]
            print 



class FixedContextsData(object):
    """
    A data_dict object simply consists of two attributes: 1) a 2D numpy matrix of contexts and 2) a 2D numpy matrix of corresponding speaker utterances, where the rows with matching indices of these two matrices belong together.
    """
    #TODO: Shouldn't this class take as input a speaker or population of speakers and then create the utterance_matrix_same_first?
    def __init__(self, context_matrix, topic_matrix, signal_counts_per_context_matrix, helpful_contexts):
        """
        :param context_matrix: A 2D numpy matrix of contexts
        :param topic_matrix: A 2D numpy matrix of topics (i.e. intended meanings)
        :param signal_counts_per_context_matrix: A 2D numpy array containing the signal counts for each context (should have shape = (n_contexts, n_signals)).
        :return: Creates a data_dict object with initial context matrix attribute and utterance_matrix_same_first attribute
        """
        self.contexts = context_matrix
        self.topics = topic_matrix
        self.signal_counts_per_context_matrix = signal_counts_per_context_matrix
        self.helpful_contexts = helpful_contexts

    def print_data(self):
        """
        :return: Prints the data_dict on separate lines with each context (with index) followed by the corresponding signal counts (with index). (Does not return anything.)
        """
        for c in range(len(self.contexts)):
            context_index = c % len(self.helpful_contexts)
            print 'context_index is:'
            print context_index
            print 'context '+str(c)+' is:'
            print self.contexts[c]
            print 'topics '+str(c)+' is:'
            print self.topics[c]
            print 'signal counts for this context are '+str(context_index)+' is:'
            print self.signal_counts_per_context_matrix[context_index]
            print


class SpeakerAnnotatedData(Data):
    """
    Same as data_dict superclass, but with an extra attribute 'speaker_array', containing for each utterance the index of the speaker that produced it (these indices matching those in the Population object).
    """

    def __init__(self, context_matrix, topic_matrix, utterance_matrix, speaker_id_matrix):
        """
        :param context_matrix: A 2D numpy matrix of contexts
        :param utterance_matrix: A 2D numpy matrix of utterances
        :param speaker_matrix: A 2D numpy matrix with exactly the same shape as parameter 'utterance_matrix_same_first', specifying the identity of the speaker (by index) for each of the utterances.
        :return: Creates a data_dict object with initial context matrix attribute, utterance_matrix_same_first attribute and speaker_array attribute
        """
        super(SpeakerAnnotatedData, self).__init__(context_matrix, topic_matrix, utterance_matrix)
        self.contexts = context_matrix
        self.utterances = utterance_matrix
        self.speaker_id_matrix = speaker_id_matrix


    def print_data(self):
        """
        :return: Prints the data_dict on separate lines with each context (with index) followed by the corresponding speaker id and array of utterances (with index). (Does not return anything.)
        """
        for i in range(len(self.contexts)):
            print 'context '+str(i)+' is:'
            print self.contexts[i]
            print 'speaker id '+str(i)+' is:'
            print self.speaker_id_matrix[i]
            print 'utterances '+str(i)+' is:'
            print self.utterances[i]
            print





def calc_n_datasets(n_meanings, n_observations):
    """
    :param n_meanings: int -- the number of meanings in the lexicon
    :param n_observations: int -- the number of observations each learner receives
    :return: int -- the total number of possible UNIQUE datasets (i.e. combinations, without permutations).
    """
    n_datasets = (factorial(n_observations+(n_meanings-1))) / ((factorial(n_observations))*factorial(n_meanings-1))
    return n_datasets



def create_all_possible_datasets(n_observations, n_signals):
    """
    :param n_observations: integer corresponding to the number of observations each learner gets to see
    :param n_signals: int -- the number of signals in the lexicon
    :return: 2D numpy array containing all possible datasets, where each dataset has size equal to n_observations.
    """
    all_datasets = list(itertools.combinations_with_replacement(np.arange(n_signals), n_observations))
    all_datasets = np.asarray(all_datasets)
    return all_datasets


def convert_dataset_to_signal_counts_per_context(dataset, n_helpful_contexts, n_signals):
    dataset_as_signal_counts_per_context = np.zeros((n_helpful_contexts, n_signals)).astype(int)
    for c in range(len(dataset)):
        context_index = c % n_helpful_contexts
        utterance = int(dataset[c])
        dataset_as_signal_counts_per_context[context_index][utterance] += 1
    return dataset_as_signal_counts_per_context


def convert_dataset_array_to_signal_counts_per_context(dataset_array, n_helpful_contexts, n_signals):
    datasets_as_signal_counts_per_context = np.zeros((len(dataset_array), n_helpful_contexts, n_signals))
    for d in range(len(dataset_array)):
        dataset_as_signal_counts_per_context = convert_dataset_to_signal_counts_per_context(dataset_array[d], n_helpful_contexts, n_signals)
        datasets_as_signal_counts_per_context[d] = dataset_as_signal_counts_per_context
    return datasets_as_signal_counts_per_context



def create_all_possible_dataset_permutations_old(n_observations, n_signals):
    all_dataset_permutations = np.array(list(itertools.product(np.arange(n_signals), repeat=n_observations)))
    return all_dataset_permutations



def create_all_possible_dataset_permutations_new(all_dataset_combinations):
    all_dataset_permutations = np.array(list(itertools.permutations(all_dataset_combinations)))
    return all_dataset_permutations



def create_all_possible_signal_counts_list(n_signals, n_observations, n_helpful_contexts):
    n_utterances_per_context = n_observations/n_helpful_contexts
    numbers = np.arange(n_utterances_per_context+1)
    possible_signal_counts_list = [seq for seq in itertools.product(numbers, repeat=n_signals) if sum(seq) == n_utterances_per_context]
    print ''
    print ''
    print "possible_signal_counts_list is:"
    print possible_signal_counts_list
    print "np.asarray(possible_signal_counts_list).shape is:"
    print np.asarray(possible_signal_counts_list).shape
    return possible_signal_counts_list


def create_all_possible_signal_counts_per_context_datasets(n_signals, n_observations, n_helpful_contexts):
    possible_signal_counts_list = create_all_possible_signal_counts_list(n_signals, n_observations, n_helpful_contexts)
    # all_datasets_as_signal_counts_per_context = np.asarray(list(itertools.combinations_with_replacement(possible_signal_counts_list, n_helpful_contexts)))
    all_datasets_as_signal_counts_per_context = np.asarray(list(itertools.product(possible_signal_counts_list, repeat=n_helpful_contexts)))
    print ''
    print ''
    print "all_datasets_as_signal_counts_per_context[1] is:"
    print all_datasets_as_signal_counts_per_context[1]
    print "all_datasets_as_signal_counts_per_context.shape is:"
    print all_datasets_as_signal_counts_per_context.shape
    return all_datasets_as_signal_counts_per_context




def create_all_possible_signal_counts_per_context_dataset_permtuations(n_signals, n_observations, n_helpful_contexts):
    all_datasets_as_signal_counts_per_context = create_all_possible_signal_counts_per_context_datasets(n_signals, n_observations, n_helpful_contexts)
    all_permutations = np.asarray(list(itertools.permutations(all_datasets_as_signal_counts_per_context)))
    print ''
    print ''
    print "all_permutations[0][1] are:"
    print all_permutations[0][1]
    print "all_permutations.shape are:"
    print all_permutations.shape
    return all_permutations





def calc_n_permutations_per_dataset(all_datasets):
    """
    :param all_datasets: 2D numpy array of datasets, where each dataset has size equal to n_observations.
    :return: 1D numpy array containing the number of permutations for each dataset. This number is calculated using the formula for distinguishable permutations: N! / (n1!)(n2!)...(nk!) where n1, n2, ..., nk stands for the frequency of the different distinguishable elements (i.e. different signals) in the combination (i.e. the dataset), and n1+n2+...+nk = N.
    """
    n_permutations_per_dataset_matrix = np.zeros(len(all_datasets))
    for d in range(len(all_datasets)):
        dataset = all_datasets[d]
        signal_freqs = np.bincount(dataset)
        denominator = 1.
        for f in signal_freqs:
            f_permutations = factorial(f)
            denominator *= f_permutations
        n_permutations = np.divide(factorial(len(dataset)), denominator)
        n_permutations_per_dataset_matrix[d] = n_permutations
    return n_permutations_per_dataset_matrix








# n_helpful_contexts = 12
# n_observations = 24
# n_signals = 3
#
#
#
# all_datasets_as_signal_counts_per_context = create_all_possible_signal_counts_per_context_datasets(n_signals, n_observations, n_helpful_contexts)

# all_dataset_permutations_as_signal_counts_per_context = create_all_possible_signal_counts_per_context_dataset_permtuations(n_signals, n_observations, n_helpful_contexts)


#
#
# some_dataset =  np.array([[0, 0, 2],
#  [0, 0, 2],
#  [0, 0, 2],
#  [0, 0, 2],
#  [0, 0, 2],
#  [0, 0, 2],
#  [0, 0, 2],
#  [0, 0, 2],
#  [0, 0, 2],
#  [0, 0, 2],
#  [0, 0, 2],
#  [0, 1, 1]])
#
#
# some_dataset_flattened = some_dataset.flatten()
# print ''
# print ''
# print "some_dataset_flattened is:"
# print some_dataset_flattened
# print "some_dataset_flattened.shape is:"
# print some_dataset_flattened.shape
#
#
# all_datasets_as_signal_counts_per_context_reshaped = all_datasets_as_signal_counts_per_context.reshape((all_datasets_as_signal_counts_per_context.shape[0], (all_datasets_as_signal_counts_per_context.shape[1]*all_datasets_as_signal_counts_per_context.shape[2])))
# print ''
# print ''
# print "all_datasets_as_signal_counts_per_context_reshaped[324] is:"
# print all_datasets_as_signal_counts_per_context_reshaped[324]
# print "all_datasets_as_signal_counts_per_context_reshaped[323] is:"
# print all_datasets_as_signal_counts_per_context_reshaped[323]
# print "all_datasets_as_signal_counts_per_context_reshaped[322] is:"
# print all_datasets_as_signal_counts_per_context_reshaped[322]
# print "all_datasets_as_signal_counts_per_context_reshaped.shape is:"
# print all_datasets_as_signal_counts_per_context_reshaped.shape
#
#
# some_dataset_index = np.argwhere((all_datasets_as_signal_counts_per_context_reshaped == some_dataset_flattened).all(axis=1))
# print ''
# print ''
# print "some_dataset_index is:"
# print some_dataset_index
#
