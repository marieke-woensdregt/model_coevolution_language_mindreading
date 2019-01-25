__author__ = 'Marieke Woensdregt'



import numpy as np
from math import factorial
import itertools

np.set_printoptions(threshold=np.nan)



#######################################################################################################################
# STEP 4: The functions below are used to generate contexts:


def gen_context_abs(n_meanings, context_size):
    """
    :param n_meanings: Total number of meanings
    :param context_size: Number of meanings present in the context
    :return: An absolute context where meanings have discrete binary values of 1 ('in') or 0 ('out').  A context is a 1D numpy array where each meaning has one value on its corresponding index.
    """
    context = np.zeros(n_meanings)
    meanings_present = np.random.choice(n_meanings, context_size, replace=False)
    for meaning in meanings_present:
        context[meaning] = 1.
    return context



def gen_context_contin(n_meanings):
    """
    :param n_meanings: Number of meanings
    :return: Returns a continuous context, where each of the meanings has a position between 0.0 and 1.0. A context is a 1D numpy array where each meaning has one value on its corresponding index. Importantly this array is NOT normalized.
    """
    context = np.random.random(n_meanings)
    return context



def gen_context_matrix(context_type, n_meanings, context_size, n_contexts):
    """
    :param context_type: Type of context; either 'absolute' or 'continuous'
    :param n_meanings: Number of meanings
    :param context_size: Number of meanings present in the context (only counts if context_type is 'absolute'!)
    :param n_contexts: Total number of contexts to be generated
    :return: A 2D numpy array containing all the contexts that the learner will get to observe during one run of the model. Axis 0 = contexts, axis 1 = meaning attributes
    """
    context_matrix = np.zeros((n_contexts, n_meanings))
    # if context_type == 'absolute':
    #     for c in range(n_contexts):
    #         context = gen_context_abs(n_meanings, context_size)
    #         context_matrix[c] = context
    # elif context_type == 'continuous':

    for i in range(n_contexts):
        context = gen_context_contin(n_meanings)
        context_matrix[i] = context
    return context_matrix


def gen_helpful_context_matrix(n_meanings, n_contexts, helpful_contexts):
    context_matrix = np.zeros((n_contexts, n_meanings))
    context_indices = np.arange(len(helpful_contexts))
    for i in range(n_contexts):
        index = np.random.choice(context_indices)
        context_matrix[i] = helpful_contexts[index]
    return context_matrix



def gen_helpful_context_matrix_fixed_order(n_meanings, n_contexts, helpful_contexts):
    if n_contexts % len(helpful_contexts) != 0:
        raise ValueError("n_contexts must be a multiple of the number of helpful contexts")
    # context_matrix = np.full(((n_contexts/len(helpful_contexts)), len(helpful_contexts), n_meanings), helpful_contexts)
    context_matrix = [helpful_contexts for x in range(n_contexts/len(helpful_contexts))]
    context_matrix = np.asarray(context_matrix)
    context_matrix = context_matrix.flatten()
    context_matrix = np.reshape(context_matrix, (n_contexts, n_meanings))
    return context_matrix



def create_all_possible_contexts(n_meanings):
    column_lengths = [9 for x in range(n_meanings)]
    context_matrix = np.indices(column_lengths, dtype=float).reshape(n_meanings, -1).T
    context_matrix = np.add(context_matrix, 1.)
    context_matrix = np.divide(context_matrix, 10.)
    return context_matrix


def remove_duplicate_attribute_contexts(context_matrix):
    new_context_matrix = []
    for c in range(len(context_matrix)):
        context = context_matrix[c]
        hist_context = np.histogram(context, bins=np.arange(0.09, 1.09, 0.1))
        if len(np.argwhere(hist_context[0]>1)) == 0:
            new_context_matrix.append(context)
    new_context_matrix = np.array(new_context_matrix)
    return new_context_matrix


def calc_distances(agent_perspective, context):
    """
    :param agent_perspective: The perspective of the agent for whom we want to calculate the distances (can be self.perspective or that of another agent) (float)
    :param context: The context for which we want to calculate the distances to agent_perspective (1D numpy array)
    :return: The distances between the meanings and agent_perspective in the context (1D numpy array)
    """
    distances = np.subtract(agent_perspective, context)
    # elif self.context_type == 'absolute':
    #     distances = np.subtract(1., context)
    abs_distances = np.absolute(distances)
    return abs_distances


def calc_saliencies(agent_perspective, context, alpha):
    """
    :param agent_perspective: The perspective of the agent for whom we want to calculate the saliencies (can be self.perspective or that of another agent) (float)
    :param context: The context for which we want to calculate the saliencies from agent_perspective (1D numpy array)
    :return: The saliency of the meanings based on distance from agent_perspective (1D numpy array)
    """
    distances = calc_distances(agent_perspective, context)
    inverse_distances = np.subtract(1, distances)
    saliencies = np.power(inverse_distances, alpha)
    return saliencies


def calc_intention(agent_perspective, context, alpha):
    """
    :param agent_perspective: The perspective of the agent for whom we want to calculate the intention distribution (can be self.perspective or that of another agent) (float)
    :param context: The context for which we want to calculate the intention distribution given agent_perspective (1D numpy array)
    :return: The intention distribution (c.e. probability of choosing meanings as topic meaning) given agent_perspective and context (1D numpy array) (This is simply a normalized version of the saliency array.)
    """
    saliencies = calc_saliencies(agent_perspective, context, alpha)
    intention = np.divide(saliencies, np.sum(saliencies))
    return intention


def context_informativeness(context_matrix, perspectives, alpha):
    print ''
    print ''
    print ''
    print 'This is the context_informativeness() function'
    print "context_matrix is:"
    print context_matrix
    print "context_matrix.shape is:"
    print context_matrix.shape
    if len(perspectives) > 2:
        print 'Sorry, this function only works for 2 perspectives.'
    context_informativeness_array = np.zeros(len(context_matrix))
    for c in range(len(context_matrix)):
        print ''
        print ''
        print ''
        print "c is:"
        print c
        context = context_matrix[c]
        print "context is:"
        print context
        n_m_combinations = np.divide(factorial(len(context)), (factorial(2) * factorial(len(context) - 2)))
        print "n_m_combinations is:"
        print n_m_combinations
        m_ratios_per_perspective = np.zeros((len(perspectives), n_m_combinations))
        for p in range(len(perspectives)):
            print "p is:"
            print p
            perspective = perspectives[p]
            print "perspective is:"
            print perspective
            p_intention = calc_intention(perspective, context, alpha)
            print "p_intention is:"
            print p_intention
            m_combinations = list(itertools.combinations(p_intention, 2))
            print "m_combinations is:"
            print m_combinations
            for m in range(len(m_combinations)):
                print "m is:"
                print m
                m_combi = m_combinations[m]
                if len(context) == 2 and p == 1:
                    m_combi = m_combi[::-1]
                print "m_combi is:"
                print m_combi
                ratio = np.divide(m_combi[0], m_combi[1])
                print "ratio is:"
                print ratio
                m_ratios_per_perspective[p][m] = ratio
        print "m_ratios_per_perspective is:"
        print m_ratios_per_perspective
        print "m_ratios_per_perspective.shape is:"
        print m_ratios_per_perspective.shape
        r_diff_array = np.zeros((len(m_ratios_per_perspective[0]), len(m_ratios_per_perspective[0])))
        for i in range(len(m_ratios_per_perspective[0])):
            print ''
            print "c is:"
            print i
            print "m_ratios_per_perspective[0] is:"
            print m_ratios_per_perspective[0]
            print "m_ratios_per_perspective[1] is:"
            print m_ratios_per_perspective[1]
            print "np.absolute(np.subtract(m_ratios_per_perspective[0], m_ratios_per_perspective[1])) is:"
            print np.absolute(np.subtract(m_ratios_per_perspective[0], m_ratios_per_perspective[1]))
            r_diff_array[i] = np.absolute(np.subtract(m_ratios_per_perspective[0], m_ratios_per_perspective[1]))
            print "r_diff_array[c] is:"
            print r_diff_array[i]
            m_ratios_per_perspective[1] = np.roll(m_ratios_per_perspective[1], 1)
            print "m_ratios_per_perspective[1] is:"
            print m_ratios_per_perspective[1]
        sum_diff = np.sum(r_diff_array.flatten())
        print "sum_diff is:"
        print sum_diff
        context_informativeness_array[c] = sum_diff
    print "context_informativeness_array is:"
    print context_informativeness_array
    print "context_informativeness_array.shape is:"
    print context_informativeness_array.shape
    return context_informativeness_array



def calc_most_informative_contexts(n_meanings, perspectives, alpha):
    all_possible_contexts = create_all_possible_contexts(n_meanings)
    all_possible_contexts = remove_duplicate_attribute_contexts(all_possible_contexts)
    context_informativeness_array = context_informativeness(all_possible_contexts, perspectives, alpha)
    context_informativeness_array = np.round(context_informativeness_array, decimals=2)
    max_context_informativeness = np.amax(context_informativeness_array)
    indices_max_context_informativeness = np.argwhere(context_informativeness_array == max_context_informativeness)
    most_informative_contexts = all_possible_contexts[indices_max_context_informativeness]
    ##
    context_informativeness_array_second = np.delete(context_informativeness_array, indices_max_context_informativeness)
    max_context_informativeness_second = np.amax(context_informativeness_array_second)
    indices_max_context_informativeness_second = np.argwhere(
        context_informativeness_array == max_context_informativeness_second)
    second_most_informative_contexts = all_possible_contexts[indices_max_context_informativeness_second]
    # ##
    # context_informativeness_array_third = np.delete(context_informativeness_array, np.vstack((indices_max_context_informativeness, indices_max_context_informativeness_second)))
    # max_context_informativeness_third = np.amax(context_informativeness_array_third)
    # indices_max_context_informativeness_third = np.argwhere(
    #     context_informativeness_array == max_context_informativeness_third)
    # third_most_informative_contexts = all_possible_contexts[indices_max_context_informativeness_third]
    return most_informative_contexts, second_most_informative_contexts



if __name__ == "__main__":


    n_meanings = 3
    perspectives = np.array([0., 1.])
    sal_alpha = 1.

    most_informative_contexts, second_most_informative_contexts = calc_most_informative_contexts(n_meanings, perspectives, sal_alpha)
    print ''
    print ''
    print "most_informative_contexts are:"
    print most_informative_contexts

