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
    #     for i in range(n_contexts):
    #         context = gen_context_abs(n_meanings, context_size)
    #         context_matrix[i] = context
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
    :return: The intention distribution (i.e. probability of choosing meanings as topic meaning) given agent_perspective and context (1D numpy array) (This is simply a normalized version of the saliency array.)
    """
    saliencies = calc_saliencies(agent_perspective, context, alpha)
    intention = np.divide(saliencies, np.sum(saliencies))
    return intention


def context_informativeness(context_matrix, perspectives, alpha):
    if len(perspectives) > 2:
        print 'Sorry, this function only works for 2 perspectives.'
    context_informativeness_array = np.zeros(len(context_matrix))
    for c in range(len(context_matrix)):
        context = context_matrix[c]
        n_m_combinations = np.divide(factorial(len(context)), (factorial(2) * factorial(len(context) - 2)))
        m_ratios_per_perspective = np.zeros((len(perspectives), n_m_combinations))
        for p in range(len(perspectives)):
            perspective = perspectives[p]
            p_intention = calc_intention(perspective, context, alpha)
            m_combinations = list(itertools.combinations(p_intention, 2))
            for m in range(len(m_combinations)):
                m_combi = m_combinations[m]
                if len(context) == 2 and p == 1:
                    m_combi = m_combi[::-1]
                ratio = np.divide(m_combi[0], m_combi[1])
                m_ratios_per_perspective[p][m] = ratio
        r_diff_array = np.zeros((len(m_ratios_per_perspective[0]), len(m_ratios_per_perspective[0])))
        for i in range(len(m_ratios_per_perspective[0])):
            r_diff_array[i] = np.absolute(np.subtract(m_ratios_per_perspective[0], m_ratios_per_perspective[1]))
            m_ratios_per_perspective[1] = np.roll(m_ratios_per_perspective[1], 1)
        sum_diff = np.sum(r_diff_array.flatten())
        context_informativeness_array[c] = sum_diff
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



# n_meanings = 4
# perspectives = np.array([0., 1.])
# sal_alpha = 1.
#
# most_informative_contexts, second_most_informative_contexts = calc_most_informative_contexts(n_meanings, perspectives, sal_alpha)
# print ''
# print ''
# print "most_informative_contexts are:"
# print most_informative_contexts
# print ''
# print ''
# print "second_most_informative_contexts are:"
# print second_most_informative_contexts
#

