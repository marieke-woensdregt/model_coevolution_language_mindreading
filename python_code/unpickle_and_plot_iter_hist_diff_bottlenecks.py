__author__ = 'Marieke Woensdregt'


import numpy as np
from scipy import stats
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import plots

# np.set_printoptions(threshold=np.nan)



#######################################################################################################################
# STEP 2: THE PARAMETERS:


# 2.1: The parameters defining the lexicon size (and thus the number of meanings in the world):

n_meanings = 3  # The number of meanings
n_signals = 3  # The number of signals


# 2.2: The parameters defining the contexts and how they map to the agent's saliencies:

context_generation = 'optimal'  # This can be set to either 'random', 'only_helpful' or 'optimal'
if n_meanings == 2:
    helpful_contexts = np.array([[0.1, 0.7], [0.3, 0.9],
                                 [0.7, 0.1], [0.9, 0.3]])
elif n_meanings == 3:
    helpful_contexts = np.array([[0.1, 0.2, 0.9], [0.1, 0.8, 0.9],
                                 [0.1, 0.9, 0.2], [0.1, 0.9, 0.8],
                                 [0.2, 0.1, 0.9], [0.8, 0.1, 0.9],
                                 [0.2, 0.9, 0.1], [0.8, 0.9, 0.1],
                                 [0.9, 0.1, 0.2], [0.9, 0.1, 0.8],
                                 [0.9, 0.2, 0.1], [0.9, 0.8, 0.1]])

elif n_meanings == 4:
    helpful_contexts = np.array([[0.1, 0.2, 0.3, 0.9], [0.1, 0.7, 0.8, 0.9],
                                 [0.1, 0.2, 0.9, 0.3], [0.1, 0.7, 0.9, 0.8],
                                 [0.1, 0.3, 0.2, 0.9], [0.1, 0.8, 0.7, 0.9],
                                 [0.1, 0.3, 0.9, 0.2], [0.1, 0.8, 0.9, 0.7],
                                 [0.1, 0.9, 0.2, 0.3], [0.1, 0.9, 0.7, 0.8],
                                 [0.1, 0.9, 0.3, 0.2], [0.1, 0.9, 0.8, 0.7],
                                 [0.2, 0.1, 0.3, 0.9], [0.7, 0.1, 0.8, 0.9],
                                 [0.2, 0.1, 0.9, 0.3], [0.7, 0.1, 0.9, 0.8],
                                 [0.2, 0.3, 0.1, 0.9], [0.7, 0.8, 0.1, 0.9],
                                 [0.2, 0.3, 0.9, 0.1], [0.7, 0.8, 0.9, 0.1],
                                 [0.2, 0.9, 0.1, 0.3], [0.7, 0.9, 0.1, 0.8],
                                 [0.2, 0.9, 0.3, 0.1], [0.7, 0.9, 0.8, 0.1],
                                 [0.3, 0.1, 0.2, 0.9], [0.8, 0.1, 0.7, 0.9],
                                 [0.3, 0.1, 0.9, 0.2], [0.8, 0.1, 0.9, 0.7],
                                 [0.3, 0.2, 0.1, 0.9], [0.8, 0.7, 0.1, 0.9],
                                 [0.3, 0.2, 0.9, 0.1], [0.8, 0.7, 0.9, 0.1],
                                 [0.3, 0.9, 0.1, 0.2], [0.8, 0.9, 0.1, 0.7],
                                 [0.3, 0.9, 0.2, 0.1], [0.8, 0.9, 0.7, 0.1],
                                 [0.9, 0.1, 0.2, 0.3], [0.9, 0.1, 0.7, 0.8],
                                 [0.9, 0.1, 0.3, 0.2], [0.9, 0.1, 0.8, 0.7],
                                 [0.9, 0.2, 0.1, 0.3], [0.9, 0.7, 0.1, 0.8],
                                 [0.9, 0.2, 0.3, 0.1], [0.9, 0.7, 0.8, 0.1],
                                 [0.9, 0.3, 0.1, 0.2], [0.9, 0.8, 0.1, 0.7],
                                 [0.9, 0.3, 0.2, 0.1], [0.9, 0.8, 0.7, 0.1]])


# 2.3: The parameters that determine the make-up of the population:

pop_size = 100
pragmatic_level = 'prag'  # This can be set to either 'literal', 'perspective-taking' or 'prag'
optimality_alpha = 3.0  # Goodman & Stuhlmuller (2013) fitted sal_alpha = 3.4 to participant data with 4x3 lexicon of three number words ('one', 'two', and 'three') that could be mapped to four different world states (0, 1, 2, and 3)
teacher_type = 'sng_teacher'  # This can be set to either 'sng_teacher' or 'multi_teacher'
agent_type = 'no_p_distinction'  # This can be set to either 'p_distinction' or 'no_p_distinction'. Determines whether the population is made up of DistinctionAgent objects or Agent objects (DistinctionAgents can learn different perspectives for different agents, Agents can only learn one perspective for all agents they learn from).


lexicon_types = ['optimal_lex', 'half_ambiguous_lex', 'fully_ambiguous_lex']  # The lexicon types that can be present in the population
#FIXME: In the current implementation 'half_ambiguous_lex' can have different instantiations. Therefore, if we run a simulation where there are different lexicon_type_probs, we will want the run_type to be 'population_same_pop' to make sure that all the 'half ambiguous' speakers do have the SAME half ambiguous lexicon.
lexicon_type_probs = np.array([0., 0., 1.])  # The ratios with which the different lexicon types will be present in the population
lexicon_type_probs_string = convert_array_to_string(lexicon_type_probs)  # Turns the lexicon type probs into a string in order to add it to file names


perspectives = np.array([0., 1.])  # The different perspectives that agents can have
perspective_probs = np.array([0., 1.])  # The ratios with which the different perspectives will be present in the population
perspective_probs_string = convert_array_to_string(perspective_probs)  # Turns the perspective probs into a string in order to add it to file names


learning_types = ['map', 'sample']  # The types of learning that the learners can do
learning_type_probs = np.array([0., 1.])  # The ratios with which the different learning types will be present in the population
learning_type_probs_string = convert_array_to_string(learning_type_probs)  # Turns the learning type probs into a string in order to add it to file names
if learning_type_probs[0] == 1.:
    learning_type_string = learning_types[0]
elif learning_type_probs[1] == 1.:
    learning_type_string = learning_types[1]
#learning_type_string = learning_types[np.where(learning_type_probs==1.)[0]]


# 2.6: The parameters that determine the learner's hypothesis space:

perspective_hyps = np.array([0., 1.])  # The perspective hypotheses that the learner will consider (1D numpy array)

which_lexicon_hyps = 'all'  # Determines the set of lexicon hypotheses that the learner will consider. This can be set to either 'all', 'all_with_full_s_space' or 'only_optimal'
if which_lexicon_hyps == 'all':
    lexicon_hyps = create_all_lexicons(n_meanings, n_signals)  # The lexicon hypotheses that the learner will consider (1D numpy array)
elif which_lexicon_hyps == 'all_with_full_s_space':
    all_lexicon_hyps = create_all_lexicons(n_meanings, n_signals)
    lexicon_hyps = remove_subset_of_signals_lexicons(all_lexicon_hyps)  # The lexicon hypotheses that the learner will consider (1D numpy array)
elif which_lexicon_hyps == 'only_optimal':
    lexicon_hyps = create_all_optimal_lexicons(n_meanings, n_signals)  # The lexicon hypotheses that the learner will consider (1D numpy array)


if agent_type == 'no_p_distinction':
    hypothesis_space = list_hypothesis_space(perspective_hyps, lexicon_hyps)  # The full space of composite hypotheses that the learner will consider (2D numpy matrix with composite hypotheses on the rows, perspective hypotheses on column 0 and lexicon hypotheses on column 1)

elif agent_type == 'p_distinction':
    hypothesis_space = list_hypothesis_space_with_speaker_distinction(perspective_hyps, lexicon_hyps, pop_size)  # The full space of composite hypotheses that the learner will consider (2D numpy matrix with composite hypotheses on the rows, perspective hypotheses on column 0 and lexicon hypotheses on column 1)


# 2.7: The parameters that determine the learner's prior:

learner_perspective = 0.  # The learner's perspective

perspective_prior_type = 'egocentric'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
perspective_prior_strength = 0.9  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
perspective_prior_strength_string = convert_array_to_string(perspective_prior_strength)


lexicon_prior_type = 'neutral'  # This can be set to either 'neutral', 'ambiguous_fixed', 'half_ambiguous_fixed', 'expressivity_bias' or 'compressibility_bias'.
lexicon_prior_constant = 0.0  # Determines the strength of the lexicon prior, with small c creating a STRONG prior and large c creating a WEAK prior. (And with c = 1000 creating an almost uniform prior.) For the expressivity bias I chose c = 0.3, for the compressibility bias c = 0.0003.
lexicon_prior_constant_string = convert_array_to_string(lexicon_prior_constant)


# 2.5: The parameters that determine the amount of data_dict that the learner gets to see, and the amount of runs of the simulation:

n_utterances = 1  # This parameter determines how many signals the learner gets to observe in each context
n_contexts = 120

# The number of contexts that the learner gets to see.


# 2.7: The parameters that determine the type and number of simulations that are run:

#FIXME: In the current implementation 'half_ambiguous_lex' can have different instantiations. Therefore, if we run a simulation where there are different lexicon_type_probs, we will want the run_type to be 'population_same_pop' to make sure that all the 'half ambiguous' speakers do have the SAME half ambiguous lexicon.
run_type = 'iter'  # This parameter determines whether the learner communicates with only one speaker ('dyadic') or with a population ('population_diff_pop' if there are no speakers with the 'half_ambiguous' lexicon type, 'population_same_pop' if there are, 'population_same_pop_dist_learner' if the learner can distinguish between different speakers), or whether we do an iterated learning model ('iter')
turnover_type = 'whole_pop'  # The type of turnover with which the population is replaced (for the iteration simulation). This can be set to either 'chain' (one agent at a time) or 'whole_pop' (entire population at once)

communication_type = 'prag'  # This can be set to either 'lex_only', 'lex_n_context', 'lex_n_p' or 'prag'
ca_measure_type = 'comp_only'  # This can be set to either "comp_n_prod" or "comp_only"
n_interactions = 6  # The number of interactions used to calculate communicative accuracy

selection_type = 'ca_with_parent'  # This can be set to either 'none', 'p_taking', 'l_learning' or 'ca_with_parent'
selection_weighting = 'none'  # This is a factor with which the fitness of the agents (determined as the probability they assign to the correct perspective hypothesis) is multiplied and then exponentiated in order to weight the relative agent fitness (which in turn determines the probability of becoming a teacher for the next generation). A value of 0. implements neutral selection. A value of 1.0 creates weighting where the fitness is pretty much equal to relative posterior probability on correct p hyp), and the higher the value, the more skewed the weighting in favour of agents with better perspective-taking.
if isinstance(selection_weighting, float):
    selection_weight_string = str(np.int(selection_weighting))
else:
    selection_weight_string = selection_weighting

n_iterations = 500  # The number of iterations (i.e. new agents in the case of 'chain', generations in the case of 'whole_pop')
report_every_i = 100  # Determines how often the progress is printed as the simulation runs
n_runs = 1  # The number of runs of the simulation
report_every_r = 1  # Determines how often the progress is printed as the simulation runs

recording = 'minimal'  # This can be set to either 'everything' or 'minimal'. If this is set to 'everything' the posteriors distributions for every single time step (in developmental time) for every single agent of every single generation are recorded for every single run. If this is set to 'minimal' only the selected hypotheses per generation are recorded for every single run.


which_hyps_on_graph = 'lex_hyps_only'  # This is used in the plot_post_probs_over_lex_hyps() function, and can be set to either 'all_hyps' or 'lex_hyps_only' or 'lex_hyps_collapsed'

n_copies = 20  # Specifies the number of copies of the results file
copy_specification = ''  # Can be set to e.g. '_c1' or simply to '' if there is only one copy


lex_measure = 'ca'  # This can be set to either 'mi' for mutual information or 'ca' for communicative accuracy (of the lexicon with itself)


random_parent = False


low_cut_off = 100
high_cut_off = 501

legend = False
text_size = 1.7
error_bars = True
#######################################################################################################################



def get_mean_percentiles_avg_fitness(pickle_directory, n_runs, n_copies, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type):
    perspective_prior_strength_string = convert_array_to_string(perspective_prior_strength)
    ##
    if selection_type == 'none' or selection_type == 'p_taking':
        if context_generation == 'random':
            filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type+'_random_'+str(random_parent)
        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type+'_random_'+str(random_parent)
    elif selection_type == 'ca_with_parent':
        if context_generation == 'random':
            filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type+'_random_'+str(random_parent)
        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type+'_random_'+str(random_parent)

    ##
    percentiles_n_mean_fitness = pickle.load(open(pickle_directory+'fitness_tmcrse_'+filename_short+'.p', 'rb'))
    avg_fitness_matrix_all_runs = percentiles_n_mean_fitness['raw_data']
    mean_fitness_over_gens = percentiles_n_mean_fitness['mean_over_gens']
    conf_invs_fitness_over_gens = percentiles_n_mean_fitness['conf_invs_over_gens']
    percentiles_fitness_over_gens = percentiles_n_mean_fitness['percentiles_over_gens']
    return avg_fitness_matrix_all_runs, mean_fitness_over_gens, conf_invs_fitness_over_gens, percentiles_fitness_over_gens



def get_mean_percentiles_avg_success(pickle_directory, n_runs, n_copies, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type, success_type, random_parent=False):
    perspective_prior_strength_string = convert_array_to_string(perspective_prior_strength)
    ##
        ##
    if selection_type == 'none' or selection_type == 'p_taking':
        if context_generation == 'random':
            filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type+'_random_'+str(random_parent)
        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type+'_random_'+str(random_parent)
    elif selection_type == 'ca_with_parent':
        if context_generation == 'random':
            filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type+'_random_'+str(random_parent)
        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type+'_random_'+str(random_parent)
    ##
    if success_type == 'communication':
        percentiles_n_mean_success = pickle.load(open(pickle_directory+'comm_success_tmcrse_'+filename_short+'.p', 'rb'))
    elif success_type == 'p_taking':
        percentiles_n_mean_success = pickle.load(open(pickle_directory+'p_taking_success_tmcrse_'+filename_short+'.p', 'rb'))
    avg_matrix_all_runs = percentiles_n_mean_success['raw_data']
    mean_over_gens = percentiles_n_mean_success['mean_over_gens']
    conf_invs_over_gens = percentiles_n_mean_success['conf_invs_over_gens']
    percentiles_over_gens = percentiles_n_mean_success['percentiles_over_gens']
    return avg_matrix_all_runs, mean_over_gens, conf_invs_over_gens, percentiles_over_gens



def get_mean_percentiles_avg_informativeness(pickle_directory, n_runs, n_copies, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type):
    perspective_prior_strength_string = convert_array_to_string(perspective_prior_strength)
    ##
    if selection_type == 'none' or selection_type == 'p_taking':
        if context_generation == 'random':
            filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    ##
    elif selection_type == 'ca_with_parent':
        if context_generation == 'random':
            filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type

    ##
    percentiles_n_mean_informativity = pickle.load(open(pickle_directory+'lex_inf_data_'+lex_measure+'_'+filename_short+'.p', 'rb'))
    avg_inf_over_gens_matrix = percentiles_n_mean_informativity['raw_data']
    mean_informativity_over_gens = percentiles_n_mean_informativity['mean_inf_over_gens']
    conf_invs_informativity_over_gens = percentiles_n_mean_informativity['conf_intervals_inf_over_gens']
    percentiles_informativity_over_gens = percentiles_n_mean_informativity['percentiles_inf_over_gens']
    baseline_inf = percentiles_n_mean_informativity['baseline_inf']
    max_inf = percentiles_n_mean_informativity['max_inf']
    return avg_inf_over_gens_matrix, mean_informativity_over_gens, conf_invs_informativity_over_gens, percentiles_informativity_over_gens, baseline_inf, max_inf


def calc_mean_and_ci_over_runs(avg_over_gens_matrix, n_runs, n_copies, low_cut_off, high_cut_off):
    excerpt = avg_over_gens_matrix[:, low_cut_off:high_cut_off]
    means_per_run = np.mean(excerpt, axis=1)
    grand_mean = np.mean(means_per_run)
    grand_std = np.std(means_per_run)
    conf_invs_over_runs = stats.norm.interval(0.95, loc=grand_mean, scale=grand_std / np.sqrt(n_runs*n_copies))
    conf_invs_over_runs = np.array(conf_invs_over_runs)
    yerr_for_plot = np.subtract(grand_mean, conf_invs_over_runs[0])
    return grand_mean, yerr_for_plot



def make_barplot(plot_file_path, plot_file_title, plot_title, x_tick_labels, x_label, y_label, text_size, high_cut_off, grand_means_per_bottleneck, yerrs_per_bottleneck, selection_type, max, baseline='None', legend=True, error_bars=True):
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    # sns.set(font_scale=text_size)
    # if selection_type == 'none':
    #     color = sns.color_palette()[0]
    # elif selection_type == 'ca_with_parent':
    #     color = sns.color_palette()[3]
    # elif selection_type == 'p_taking':
    #     color = sns.color_palette()[4]
    color = sns.color_palette()[0]
    with sns.axes_style("whitegrid"):
        if len(bottleneck_sizes) < 4:
            fig, ax = plt.subplots(figsize=(6, 6))
            width = 1. / 2.
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            width = 1. / 1.3
        N = len(grand_means_per_bottleneck)
        x = np.arange(N)
        ax.axhline(max, color='0.2', linewidth=2, label='maximum')
        if baseline != 'None':
            ax.axhline(baseline, color='0.6', linewidth=2, linestyle='--', label='baseline')
        if error_bars:
            ax.bar(x+(width/2.), grand_means_per_bottleneck, width, color=color, yerr=yerrs_per_bottleneck, error_kw=dict(ecolor='black', lw=1, capsize=5, capthick=1))
        else:
            ax.bar(x+(width/2.), grand_means_per_bottleneck, width, color=color)
        ax.set_xlim(-(width/3), x[-1]+(width*1.3333))
        ax.set_xticks(np.array(x) + (width / 2.))
        ax.set_xticklabels(x_tick_labels)
        ax.set_ylim(-0.05, (max + 0.05))
        ax.set_yticks(np.arange(0.0, (max + 0.05), 0.1))
        fig.subplots_adjust(bottom=0.13)
        if legend == True:
            # Shrink current axis by 28%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
            # Put a legend to the right of the current axis
            legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, fontsize=12, borderaxespad=0.8)
            legend.get_frame().set_linewidth(1.5)
    plt.tick_params(labelsize=13)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title(plot_title, fontsize=18)
    plt.savefig(plot_file_path+plot_file_title+'_bounds_'+str(low_cut_off)+'_'+str(high_cut_off)+'.pdf', bbox_inches='tight')
    plt.show()



def make_boxplot(plot_file_path, plot_file_title, plot_title, x_tick_labels, x_label, y_label, text_size, high_cut_off, data, selection_type, max, baseline='None', legend=True):
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    # sns.set(font_scale=text_size)
    # if selection_type == 'none':
    #     color = sns.color_palette()[0]
    # elif selection_type == 'ca_with_parent':
    #     color = sns.color_palette()[3]
    # elif selection_type == 'p_taking':
    #     color = sns.color_palette()[4]
    color = sns.color_palette()[0]
    # if selection_type == 'none':
    #     color_palette = itertools.cycle(sns.light_palette("french blue", input="xkcd", n_colors=len(data)))  # could also be 'mid blue' or 'flat blue'
    # elif selection_type == 'ca_with_parent':
    #     color_palette = itertools.cycle(sns.light_palette("dark lavender", input="xkcd", n_colors=len(data))) # could also be 'deep lilac'
    # elif selection_type == 'p_taking':
    #     color_palette = itertools.cycle(sns.light_palette("desert", input="xkcd", n_colors=len(data)))  # could also be 'sandy' for a bit lighter and yellower.
    with sns.axes_style("whitegrid"):
        if len(bottleneck_sizes) < 4:
            fig, ax = plt.subplots(figsize=(6, 6))
            width = 1. / 2.
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            width = 1. / 1.3
        ax.axhline(max, color='0.2', linewidth=2, label='maximum')
        if baseline != 'None':
            ax.axhline(baseline, color='0.6', linestyle='--', linewidth=2, label='baseline')
        sns.boxplot(data=data.T, color=color, width=width)
        # sns.boxplot(data=data.T, palette=color_palette, width=width)
        ax.set_xticklabels(x_tick_labels)
        ax.set_ylim(-0.05, (max + 0.05))
        ax.set_yticks(np.arange(0.0, (max + 0.05), 0.1))
        fig.subplots_adjust(bottom=0.13)
        if legend == True:
            # Shrink current axis by 28%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
            # Put a legend to the right of the current axis
            legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, fontsize=12, borderaxespad=0.8)
            legend.get_frame().set_linewidth(1.5)
    plt.tick_params(labelsize=13)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title(plot_title, fontsize=18)
    plt.savefig(plot_file_path+plot_file_title+'_bounds_'+str(low_cut_off)+'_'+str(high_cut_off)+'.pdf', bbox_inches='tight')
    plt.show()



def make_violinplot(plot_file_path, plot_file_title, plot_title, x_tick_labels, x_label, y_label, text_size, high_cut_off, data, selection_type, max, baseline='None', legend=True):
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    # sns.set(font_scale=text_size)
    # if selection_type == 'none':
    #     color = sns.color_palette()[0]
    # elif selection_type == 'ca_with_parent':
    #     color = sns.color_palette()[3]
    # elif selection_type == 'p_taking':
    #     color = sns.color_palette()[4]
    color = sns.color_palette()[0]
    # if selection_type == 'none':
    #     color_palette = itertools.cycle(sns.light_palette("french blue", input="xkcd", n_colors=len(data))) # could also be 'mid blue' or 'flat blue'
    # elif selection_type == 'ca_with_parent':
    #     color_palette = itertools.cycle(sns.light_palette("dark lavender", input="xkcd", n_colors=len(data))) # could also be 'deep lilac'
    # elif selection_type == 'p_taking':
    #     color_palette = itertools.cycle(sns.light_palette("desert", input="xkcd", n_colors=len(data)))  # could also be 'sandy' for a bit lighter and yellower.
    with sns.axes_style("whitegrid"):
        if len(bottleneck_sizes) < 4:
            fig, ax = plt.subplots(figsize=(6, 6))
            width = 1. / 2.
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            width = 1. / 1.3
        ax.axhline(max, color='0.2', linewidth=2, label='maximum')
        if baseline != 'None':
            ax.axhline(baseline, color='0.6', linestyle='--', linewidth=2, label='baseline')
        sns.violinplot(data=data.T, color=color, width=width)
        # sns.violinplot(data=data.T, palette=color_palette, width=width)
        ax.set_xticklabels(x_tick_labels)
        ax.set_ylim(-0.05, (max + 0.05))
        ax.set_yticks(np.arange(0.0, (max + 0.05), 0.1))
        fig.subplots_adjust(bottom=0.13)
        if legend == True:
            # Shrink current axis by 28%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
            # Put a legend to the right of the current axis
            legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, fontsize=12, borderaxespad=0.8)
            legend.get_frame().set_linewidth(1.5)
    plt.tick_params(labelsize=13)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title(plot_title, fontsize=18)
    plt.savefig(plot_file_path+plot_file_title+'_bounds_'+str(low_cut_off)+'_'+str(high_cut_off)+'.pdf', bbox_inches='tight')
    plt.show()




def make_barplot_two_conditions(plot_file_path, plot_file_title, text_size, high_cut_off, grand_means_per_bottleneck_comm_success, yerrs_per_bottleneck_comm_success, grand_means_per_bottleneck_p_taking_success, yerrs_per_bottleneck_p_taking_success,max_comm_success, max_p_taking_success, legend, error_bars=True):
    sns.set_style("ticks")
    sns.set_palette("deep")
    # sns.set(font_scale=text_size)
    # if selection_type == 'none':
    #     color = sns.color_palette()[0]
    # elif selection_type == 'ca_with_parent':
    #     color = sns.color_palette()[3]
    # elif selection_type == 'p_taking':
    #     color = sns.color_palette()[4]
    color_comm_success = sns.color_palette()[3]
    color_p_taking_success = sns.color_palette()[4]
    with sns.axes_style("ticks"):
        if len(bottleneck_sizes) < 4:
            fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(9.6, 6))
            width = 1. / 2.
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(12, 6))
            width = 1. / 1.3
        N = len(grand_means_per_bottleneck_comm_success)
        x = np.arange(N)
        if error_bars:
            ax1.bar(x+(width/2.), grand_means_per_bottleneck_comm_success, width=width, color=color_comm_success, yerr=yerrs_per_bottleneck_comm_success, error_kw=dict(ecolor='black', lw=1, capsize=5, capthick=1))
        else:
            ax1.bar(x + (width / 2.), grand_means_per_bottleneck_comm_success, width=width, color=color_comm_success)
        ax1.axhline(max_comm_success, color='0.2', linewidth=2, label='maximum')
        ax1.set_xlim(-(width / 3), x[-1] + (width * 1.3333))
        ax1.set_xticks(np.array(x) + (width / 2.))
        # ax1.set_xticklabels([str(b) for b in bottleneck_sizes])
        ax1.set_xticklabels(["" for b in bottleneck_sizes])
        # ax1.set_ylim(-0.05, (max_comm_success + 0.05))
        ax1.set_ylim(-0.05, 1.05)
        # ax1.set_yticks(np.arange(0.0, (max_comm_success + 0.05), 0.1))
        ax1.set_yticks(np.arange(0.0, 1.05, 0.1))
        if error_bars:
            ax2.bar(x+(width/2.), grand_means_per_bottleneck_p_taking_success, width=width, color=color_p_taking_success, yerr=yerrs_per_bottleneck_p_taking_success, error_kw=dict(ecolor='black', lw=1, capsize=5, capthick=1))
        else:
            ax2.bar(x+(width/2.), grand_means_per_bottleneck_p_taking_success, width=width, color=color_p_taking_success)
        ax2.axhline(max_p_taking_success, color='0.2', linewidth=2, label='maximum')
        ax2.set_xlim(-(width / 3), x[-1] + (width * 1.3333))
        ax2.set_xticks(np.array(x) + (width / 2.))
        # ax2.set_xticklabels([str(b) for b in bottleneck_sizes])
        ax2.set_xticklabels(["" for b in bottleneck_sizes])
        # ax2.set_ylim(-0.05, (max_p_taking_success + 0.05))
        ax2.set_ylim(-0.05, 1.05)
        # ax2.set_yticks(np.arange(0.0, (max_p_taking_success + 0.05), 0.1))
        ax2.set_yticks(np.arange(0.0, 1.05, 0.1))
        if legend == True:
            fig.subplots_adjust(wspace=0.0, bottom=0.13)
            # Shrink current axis by 75%
            box = ax1.get_position()
            ax1.set_position([box.x0, box.y0, box.width * 0.75, box.height])
            box = ax2.get_position()
            ax2.set_position([box.x0, box.y0, box.width * 0.75, box.height])
            # Put a legend to the right of the current axis
            legend = ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, fontsize=12, borderaxespad=0.8)
            legend.get_frame().set_linewidth(1.5)
    ax1.tick_params(labelsize=13)
    ax2.tick_params(labelsize=13)
    ax1.set_ylabel('average communication success', fontsize=14)
    ax2.set_ylabel('average perspective-inference success', fontsize=14)
    sns.despine()
    fig.text(0.5, 0.04, 'bottleneck size (=amount of data transmitted)', ha='center', fontsize=14)
    plt.suptitle(plot_title, fontsize=18)
    plt.savefig(plot_file_path+plot_file_title+'_bounds_'+str(low_cut_off)+'_'+str(high_cut_off)+'.png', bbox_inches='tight')
    plt.show()


def make_boxplot_two_conditions(plot_file_path, plot_file_title, text_size, high_cut_off, data_comm_success, data_p_taking_success, max_comm_success, max_p_taking_success, legend):
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    # sns.set(font_scale=text_size)
    color_comm_success = sns.color_palette()[3]
    color_p_taking_success = sns.color_palette()[4]
    with sns.axes_style("whitegrid"):
        if len(bottleneck_sizes) < 4:
            fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(9.6, 6))
            width = 1. / 2.
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(12, 6))
            width = 1. / 1.3
        N = len(data_comm_success)
        sns.boxplot(data=data_comm_success.T, color=color_comm_success, width=width, ax=ax1)
        ax1.axhline(max_comm_success, color='0.2', linewidth=2, label='maximum')
        ax1.set_xticks(np.arange(len(bottleneck_sizes)))
        ax1.set_xticklabels([str(b) for b in bottleneck_sizes])
        ax1.set_ylim(-0.05, (max_comm_success + 0.05))
        ax1.set_yticks(np.arange(0.0, (max_comm_success + 0.05), 0.1))
        sns.boxplot(data=data_p_taking_success.T, color=color_p_taking_success, width=width, ax=ax2)
        ax2.axhline(max_p_taking_success, color='0.2', linewidth=2, label='maximum')
        ax2.set_xticks(np.arange(len(bottleneck_sizes)))
        ax2.set_xticklabels([str(b) for b in bottleneck_sizes])
        ax2.set_ylim(-0.05, (max_p_taking_success + 0.05))
        ax2.set_yticks(np.arange(0.0, (max_p_taking_success + 0.05), 0.1))
        if legend == True:
            fig.subplots_adjust(wspace=0.0, bottom=0.13)
            # Shrink current axis by 75%
            box = ax1.get_position()
            ax1.set_position([box.x0, box.y0, box.width * 0.75, box.height])
            box = ax2.get_position()
            ax2.set_position([box.x0, box.y0, box.width * 0.75, box.height])
            # Put a legend to the right of the current axis
            legend = ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, fontsize=12, borderaxespad=0.8)
            legend.get_frame().set_linewidth(1.5)
    ax1.tick_params(labelsize=13)
    ax2.tick_params(labelsize=13)
    ax1.set_ylabel('average communication success', fontsize=14)
    ax2.set_ylabel('average perspective-inference success', fontsize=14)
    fig.text(0.5, 0.04, 'bottleneck size (=amount of data transmitted)', ha='center', fontsize=14)
    plt.suptitle(plot_title, fontsize=18)
    plt.savefig(plot_file_path+plot_file_title+'_bounds_'+str(low_cut_off)+'_'+str(high_cut_off)+'.pdf', bbox_inches='tight')
    plt.show()




def make_barplot_three_conditions(plot_file_path, plot_file_title, high_cut_off, grand_means_per_bottleneck_informativeness, yerrs_per_bottleneck_informativeness, grand_means_per_bottleneck_comm_success, yerrs_per_bottleneck_comm_success, grand_means_per_bottleneck_p_taking_success, yerrs_per_bottleneck_p_taking_success, baseline_informativeness, max_informativeness, max_comm_success, max_p_taking_success, legend, error_bars=True):
    sns.set_style("darkgrid")
    sns.set_palette("deep")
    # sns.set(font_scale=text_size)
    # if selection_type == 'none':
    #     color = sns.color_palette()[0]
    # elif selection_type == 'ca_with_parent':
    #     color = sns.color_palette()[3]
    # elif selection_type == 'p_taking':
    #     color = sns.color_palette()[4]
    color_inf = sns.color_palette()[0]
    color_comm_success = sns.color_palette()[3]
    color_p_taking_success = sns.color_palette()[4]
    with sns.axes_style("darkgrid"):
        if len(bottleneck_sizes) < 4:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, figsize=(9.6, 6))
            width = 1. / 2.
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, figsize=(12, 6))
            width = 1. / 1.3
        N = len(grand_means_per_bottleneck_comm_success)
        x = np.arange(N)
        if error_bars:
            ax1.bar(x+(width/2.), grand_means_per_bottleneck_informativeness, width=width, color=color_inf, yerr=yerrs_per_bottleneck_informativeness, error_kw=dict(ecolor='black', lw=1, capsize=5, capthick=1))
        else:
            ax1.bar(x + (width / 2.), grand_means_per_bottleneck_informativeness, width=width, color=color_inf)
        l1 = ax1.axhline(baseline_informativeness, color='0.6', linewidth=2, linestyle='--', label='baseline')
        l2 = ax1.axhline(max_informativeness, color='0.2', linewidth=2, label='maximum')
        ax1.set_xlim(-(width / 3), x[-1] + (width * 1.3333))
        ax1.set_xticks(np.array(x) + (width / 2.))
        ax1.set_xticklabels([str(b) for b in bottleneck_sizes])
        ax1.set_ylim(-0.05, (max_informativeness + 0.05))
        ax1.set_yticks(np.arange(0.0, (max_informativeness + 0.05), 0.1))
        if error_bars:
            ax2.bar(x+(width/2.), grand_means_per_bottleneck_comm_success, width=width, color=color_comm_success, yerr=yerrs_per_bottleneck_comm_success, error_kw=dict(ecolor='black', lw=1, capsize=5, capthick=1))
        else:
            ax2.bar(x + (width / 2.), grand_means_per_bottleneck_comm_success, width=width, color=color_comm_success)
        ax2.axhline(max_comm_success, color='0.2', linewidth=2, label='maximum')
        ax2.set_xlim(-(width / 3), x[-1] + (width * 1.3333))
        ax2.set_xticks(np.array(x) + (width / 2.))
        ax2.set_xticklabels([str(b) for b in bottleneck_sizes])
        ax2.set_ylim(-0.05, (max_comm_success + 0.05))
        ax2.set_yticks(np.arange(0.0, (max_comm_success + 0.05), 0.1))
        if error_bars:
            ax3.bar(x+(width/2.), grand_means_per_bottleneck_p_taking_success, width=width, color=color_p_taking_success, yerr=yerrs_per_bottleneck_p_taking_success, error_kw=dict(ecolor='black', lw=1, capsize=5, capthick=1))
        else:
            ax3.bar(x+(width/2.), grand_means_per_bottleneck_p_taking_success, width=width, color=color_p_taking_success)
        ax3.axhline(max_p_taking_success, color='0.2', linewidth=2, label='maximum')
        ax3.set_xlim(-(width / 3), x[-1] + (width * 1.3333))
        ax3.set_xticks(np.array(x) + (width / 2.))
        ax3.set_xticklabels([str(b) for b in bottleneck_sizes])
        ax3.set_ylim(-0.05, (max_p_taking_success + 0.05))
        ax3.set_yticks(np.arange(0.0, (max_p_taking_success + 0.05), 0.1))
        if legend == True:
            fig.subplots_adjust(wspace=0.1, bottom=0.13)
            # Shrink current axis by 70%
            box = ax1.get_position()
            ax1.set_position([box.x0, box.y0, box.width * 0.65, box.height])
            box = ax2.get_position()
            ax2.set_position([box.x0, box.y0, box.width * 0.65, box.height])
            box = ax3.get_position()
            ax3.set_position([box.x0, box.y0, box.width * 0.65, box.height])
            # Put a legend to the right of the current axis
            legend = plt.legend(handles=[l1, l2], loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=True, fontsize=13, borderaxespad=0.8)
            legend.get_frame().set_linewidth(1.5)
    ax1.tick_params(labelsize=13)
    ax2.tick_params(labelsize=13)
    ax3.tick_params(labelsize=13)
    ax1.set_ylabel('average informativeness', fontsize=18)
    ax2.set_ylabel('average communication success', fontsize=18)
    ax3.set_ylabel('average perspective-inference success', fontsize=18)
    fig.text(0.5, 0.04, 'bottleneck size (=amount of data transmitted)', ha='center', fontsize=18)
    plt.suptitle(plot_title, fontsize=26)
    plt.savefig(plot_file_path+plot_file_title+'_bounds_'+str(low_cut_off)+'_'+str(high_cut_off)+'.png', bbox_inches='tight')
    plt.show()




def make_barplot_cogsci_london_poster(plot_file_path, plot_file_title, text_size, high_cut_off, grand_means_per_bottleneck_comm_success, yerrs_per_bottleneck_comm_success, grand_means_per_bottleneck_p_taking_success, yerrs_per_bottleneck_p_taking_success,chance_level_comm_success, max_comm_success, chance_level_p_taking_success, max_p_taking_success):
    sns.set_style("ticks")
    sns.set_palette("deep")
    # sns.set(font_scale=text_size)
    # if selection_type == 'none':
    #     color = sns.color_palette()[0]
    # elif selection_type == 'ca_with_parent':
    #     color = sns.color_palette()[3]
    # elif selection_type == 'p_taking':
    #     color = sns.color_palette()[4]
    color_comm_success = sns.color_palette()[3]
    color_p_taking_success = sns.color_palette()[4]
    with sns.axes_style("ticks"):
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(7, 6))
        N = len(grand_means_per_bottleneck_comm_success)
        x = np.arange(N)
        width = 1 / 1.8
        ax1.bar(x+(width/2.), grand_means_per_bottleneck_comm_success, width=width, color=color_comm_success, yerr=yerrs_per_bottleneck_comm_success, error_kw=dict(ecolor='black', lw=1, capsize=5, capthick=1))
        # ax1.bar(x+(width/2.), grand_means_per_bottleneck_comm_success, width=width, color=color_comm_success)
        ax1.axhline(chance_level_comm_success, color='0.6', linewidth=2, linestyle='--', label='baseline')
        ax1.axhline(max_comm_success, color='0.2', linewidth=2, label='maximum')
        ax1.set_xlim(-(width / 3), x[-1] + (width * 1.3333))
        ax1.set_xticks(np.array(x) + (width / 2.))
        # ax1.set_xticklabels([str(b) for b in bottleneck_sizes])
        ax1.set_xticklabels(["" for b in bottleneck_sizes])
        ax1.set_xlabel('Communication', fontsize=18)
        # ax1.set_ylim(-0.05, (max_comm_success + 0.05))
        ax1.set_ylim(0.0, 1.05)
        # ax1.set_yticks(np.arange(0.0, (max_comm_success + 0.05), 0.1))
        ax1.set_yticks(np.arange(0.0, 1.05, 0.1))
        ax1.set_ylabel('Avg. success of population', fontsize=20)
        ax2.bar(x+(width/2.), grand_means_per_bottleneck_p_taking_success, width=width, color=color_p_taking_success, yerr=yerrs_per_bottleneck_p_taking_success, error_kw=dict(ecolor='black', lw=1, capsize=5, capthick=1))
        # ax2.bar(x+(width/2.), grand_means_per_bottleneck_p_taking_success, width=width, color=color_p_taking_success)
        ax2.axhline(chance_level_p_taking_success, color='0.6', linewidth=2, linestyle='--', label='baseline')
        ax2.axhline(max_p_taking_success, color='0.2', linewidth=2, label='maximum')
        ax2.set_xlim(-(width / 3), x[-1] + (width * 1.3333))
        ax2.set_xticks(np.array(x) + (width / 2.))
        # ax2.set_xticklabels([str(b) for b in bottleneck_sizes])
        ax2.set_xticklabels(["" for b in bottleneck_sizes])
        ax2.set_xlabel('Perspective-inference', fontsize=18)
        # ax2.set_ylim(-0.05, (max_p_taking_success + 0.05))
        ax2.set_ylim(0.0, 1.05)
        # ax2.set_yticks(np.arange(0.0, (max_p_taking_success + 0.05), 0.1))
        ax2.set_yticks(np.arange(0.0, 1.05, 0.1))
    ax1.tick_params(labelsize=14)
    ax2.tick_params(labelsize=14)
    plt.suptitle('Success after convergence', fontsize=26)
    sns.despine()
    f.subplots_adjust(wspace=0.25)
    plt.savefig(plot_file_path+plot_file_title+'_bounds_'+str(low_cut_off)+'_'+str(high_cut_off)+'.pdf')
    plt.show()


def make_boxplot_cogsci_london_poster(plot_file_path, plot_file_title, text_size, high_cut_off, avg_comm_success_matrix_all_runs_per_bottleneck, avg_p_taking_success_matrix_all_runs_per_bottleneck, max_comm_success, max_p_taking_success):
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    sns.set(font_scale=text_size)
    # if selection_type == 'none':
    #     color = sns.color_palette()[0]
    # elif selection_type == 'ca_with_parent':
    #     color = sns.color_palette()[3]
    # elif selection_type == 'p_taking':
    #     color = sns.color_palette()[4]
    color_comm_success = sns.color_palette()[3]
    color_p_taking_success = sns.color_palette()[4]
    df = pd.DataFrame({'b':avg_comm_success_matrix_all_runs_per_bottleneck[0], 'c': avg_p_taking_success_matrix_all_runs_per_bottleneck[0]})
    with sns.axes_style("whitegrid"):
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(7, 6))
        # N = len(avg_comm_success_matrix_all_runs_per_bottleneck[0])
        # x = np.arange(N)
        # width = 1 / 1.3
        sns.boxplot(y="b", data=df,  orient='v' , color=color_comm_success, ax=ax1)
        ax1.axhline(max_comm_success, color='0.2', linewidth=3, label='maximum')
        # ax1.set_xlim(-(width / 3), x[-1] + (width * 1.3333))
        # ax1.set_xticks(np.arange(0))
        ax1.set_xlabel('Comm. success')
        ax1.set_ylim(-0.05, (max_comm_success + 0.05))
        ax1.set_yticks(np.arange(0.0, (max_comm_success + 0.05), 0.1))
        ax1.set_ylabel('Average success of population')
        sns.boxplot(y="c", data=df, orient='v', color=color_p_taking_success, ax=ax2)
        ax2.axhline(max_p_taking_success, color='0.2', linewidth=3, label='maximum')
        # ax2.set_xlim(-(width / 3), x[-1] + (width * 1.3333))
        # ax1.set_xticks(np.arange(0))
        ax2.set_xlabel('P-inference success')
        ax2.set_ylim(-0.05, (max_p_taking_success + 0.05))
        ax2.set_yticks(np.arange(0.0, (max_p_taking_success + 0.05), 0.1))
        ax2.set_ylabel('')
    plt.suptitle('Success after convergence', fontsize=32)
    f.subplots_adjust(wspace=0.25)
    plt.savefig(plot_file_path+plot_file_title+'_bounds_'+str(low_cut_off)+'_'+str(high_cut_off)+'.png')
    plt.show()



def boxplot_three_cond(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, low_cut_off, high_cut_off, data_none, data_p_taking, data_ca, baseline_inf, max_inf, legend):
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    sns.set(font_scale=text_size)
    color_none = sns.color_palette()[0]
    color_ca = sns.color_palette()[3]
    color_p_taking = sns.color_palette()[4]
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_ylim(-0.05, (max_inf + 0.05))
        ax.set_yticks(np.arange(0.0, (max_inf + 0.05), 0.1))
        width = 1 / 1.3
        sns.boxplot(data=data_none, color=color_none, width=width)
        ax.axhline(baseline_inf, color='0.6', linewidth=3, linestyle='--', label='baseline')
        ax.axhline(max_inf, color='0.2', linewidth=3, label='maximum')
        # ax.axvline(low_cut_off, color='0.6', linewidth=4, linestyle=':', label='burn-in')
        fig.subplots_adjust(bottom=0.15)
        if legend == True:
            # Shrink current axis by 28%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.65, box.height])
            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    plt.xlabel('Generations')
    plt.ylabel(y_axis_label)
    plt.suptitle(plot_title, fontsize=16)
    plt.savefig(plot_file_path+'Plot_Inf_Perc_No_selection_vs_P_taking_'+lex_measure+'_'+plot_file_title+'_high_bound_'+str(high_cut_off)+'.png')
    plt.show()



#####################################################################################

if __name__ == "__main__":


    pickle_directory = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/Iteration/'


    # # literal agents + any prior type + any selection type:
    # bottleneck_sizes = np.array([48, 60, 72, 84, 96, 108, 120, 132, 144, 156])
    # n_copies_array = np.array([20, 20, 20, 20, 20, 20, 20, 20, 20, 20])
    #
    #
    # # pragmatic agents + any prior type + any selection type:
    # bottleneck_sizes = np.array([48, 60, 72, 84, 96, 108, 120])
    # n_copies_array = np.array([20, 20, 20, 20, 20, 20, 20])

    # # PNAS paper:
    # bottleneck_sizes = np.array([72, 120])
    # n_copies_array = np.array([20, 20])

    # CogSci poster:
    bottleneck_sizes = np.array([120])
    n_copies_array = np.array([20])


    if selection_type == 'ca_with_parent':
        if n_meanings == 2 and n_signals == 2:
            max_fitness = (np.power((1.-error), 2)+np.power(error, 2))
        elif n_meanings == 3 and n_signals == 3:
            max_fitness = (np.power((1.-error), 2)+np.power((error/2), 2)++np.power((error/3), 2))
        elif n_meanings == 4 and n_signals == 4:
            max_fitness = (np.power((1.-error), 2)+np.power((error/3), 2)+np.power((error/3), 2)+np.power((error/3), 2))
        else:
            raise ValueError("Haven't calculated max_fitness for n_meanings > 4 or for asymmetrical lexicon")
        print ''
        print ''
        print "max_fitness is:"
        print max_fitness

    elif selection_type == 'p_taking':
        # if perspective_prior_type == 'neutral':
        #     fitness_baseline = 1./perspective_hyps
        # elif perspective_prior_type == 'egocentric':
        #     fitness_baseline = 1.-perspective_prior_strength

        max_fitness = 1.0
        print ''
        print ''
        print "max_fitness is:"
        print max_fitness



    print "n_copies is:"
    print n_copies
    avg_comm_success_matrix_all_runs_per_bottleneck = np.zeros((len(bottleneck_sizes), (n_runs*n_copies)))
    avg_p_taking_success_matrix_all_runs_per_bottleneck = np.zeros((len(bottleneck_sizes), (n_runs * n_copies)))
    avg_fitness_matrix_all_runs_per_bottleneck = np.zeros((len(bottleneck_sizes), (n_runs*n_copies)))
    avg_inf_matrix_all_runs_per_bottleneck = np.zeros((len(bottleneck_sizes), (n_runs*n_copies)))
    mean_comm_success_per_bottleneck = []
    yerrs_comm_success_per_bottleneck = []
    mean_p_taking_success_per_bottleneck = []
    yerrs_p_taking_success_per_bottleneck = []
    mean_fitness_per_bottleneck = []
    yerrs_fitness_per_bottleneck = []
    mean_inf_per_bottleneck = []
    yerrs_inf_per_bottleneck = []
    for i in range(len(bottleneck_sizes)):
        n_contexts = bottleneck_sizes[i]

        print ''
        print ''
        print 'This is bottleneck size:'
        print n_contexts

        n_copies = n_copies_array[i]
        print "n_copies is:"
        print n_copies

        avg_comm_success_matrix_all_runs, mean_comm_success_over_gens, conf_invs_comm_success_over_gens, percentiles_comm_success_over_gens = get_mean_percentiles_avg_success(pickle_directory, n_runs, n_copies, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type, 'communication', random_parent=False)

        avg_p_taking_success_matrix_all_runs, mean_p_taking_success_over_gens, conf_invs_p_taking_success_over_gens, percentiles_p_taking_success_over_gens = get_mean_percentiles_avg_success(pickle_directory, n_runs, n_copies, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type, 'p_taking', random_parent=False)




        avg_comm_success_matrix_excerpt = avg_comm_success_matrix_all_runs[:,low_cut_off:high_cut_off]

        comm_success_excerpt_mean_over_gens = np.mean(avg_comm_success_matrix_excerpt, axis=1)

        # if n_copies != 20:
        #     nans = ['nan' for x in range(20-n_copies)]
        #     nans = np.array(nans)
        #     comm_success_excerpt_mean_over_gens = np.hstack((comm_success_excerpt_mean_over_gens, nans))


        avg_comm_success_matrix_all_runs_per_bottleneck[i] = comm_success_excerpt_mean_over_gens

        grand_mean_comm_success, yerr_for_plot_comm_success = calc_mean_and_ci_over_runs(avg_comm_success_matrix_all_runs, n_runs, n_copies, low_cut_off, high_cut_off)
        print ''
        print "grand_mean_comm_success is:"
        print grand_mean_comm_success
        print ''
        print 'yerr_for_plot_comm_success is:'
        print yerr_for_plot_comm_success

        mean_comm_success_per_bottleneck.append(grand_mean_comm_success)
        yerrs_comm_success_per_bottleneck.append(yerr_for_plot_comm_success)



        avg_p_taking_success_matrix_excerpt = avg_p_taking_success_matrix_all_runs[:,low_cut_off:high_cut_off]

        p_taking_success_excerpt_mean_over_gens = np.mean(avg_p_taking_success_matrix_excerpt, axis=1)

        # if n_copies != 20:
        #     nans = ['nan' for x in range(20-n_copies)]
        #     nans = np.array(nans)
        #     p_taking_success_excerpt_mean_over_gens = np.hstack((p_taking_success_excerpt_mean_over_gens, nans))


        avg_p_taking_success_matrix_all_runs_per_bottleneck[i] = p_taking_success_excerpt_mean_over_gens

        grand_mean_p_taking_success, yerr_for_plot_p_taking_success = calc_mean_and_ci_over_runs(avg_p_taking_success_matrix_all_runs, n_runs, n_copies, low_cut_off, high_cut_off)
        print ''
        print "grand_mean_p_taking_success is:"
        print grand_mean_p_taking_success
        print ''
        print 'yerr_for_plot_p_taking_success is:'
        print yerr_for_plot_p_taking_success

        mean_p_taking_success_per_bottleneck.append(grand_mean_p_taking_success)
        yerrs_p_taking_success_per_bottleneck.append(yerr_for_plot_p_taking_success)




        if selection_type != 'none':
            avg_fitness_matrix_all_runs, mean_fitness_over_gens, conf_invs_fitness_over_gens, percentiles_fitness_over_gens = get_mean_percentiles_avg_fitness(pickle_directory, n_runs, n_copies, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type)

            avg_fitness_matrix_excerpt = avg_fitness_matrix_all_runs[:,low_cut_off:high_cut_off]

            fitness_excerpt_mean_over_gens = np.mean(avg_fitness_matrix_excerpt, axis=1)

            if n_copies != 20:
                nans = ['nan' for x in range(20-n_copies)]
                nans = np.array(nans)
                fitness_excerpt_mean_over_gens = np.hstack((fitness_excerpt_mean_over_gens, nans))


            avg_fitness_matrix_all_runs_per_bottleneck[i] = fitness_excerpt_mean_over_gens

            grand_mean_fitness, yerr_for_plot_fitness = calc_mean_and_ci_over_runs(avg_fitness_matrix_all_runs, n_runs, n_copies, low_cut_off, high_cut_off)
            print ''
            print "grand_mean_fitness is:"
            print grand_mean_fitness
            print ''
            print 'yerr_for_plot_fitness is:'
            print yerr_for_plot_fitness

            mean_fitness_per_bottleneck.append(grand_mean_fitness)
            yerrs_fitness_per_bottleneck.append(yerr_for_plot_fitness)






        avg_inf_over_gens_matrix, mean_informativity_over_gens, conf_invs_informativity_over_gens, percentiles_informativity_over_gens, baseline_inf, max_inf = get_mean_percentiles_avg_informativeness(pickle_directory, n_runs, n_copies, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type)

        avg_inf_matrix_excerpt = avg_inf_over_gens_matrix[:, low_cut_off:high_cut_off]

        avg_inf_excerpt_mean_over_gens = np.mean(avg_inf_matrix_excerpt, axis=1)

        # if n_copies != 20:
        #     nans = ['nan' for x in range(20-n_copies)]
        #     nans = np.array(nans)
        #     avg_inf_excerpt_mean_over_gens = np.hstack((avg_inf_excerpt_mean_over_gens, nans))

        avg_inf_matrix_all_runs_per_bottleneck[i] = avg_inf_excerpt_mean_over_gens

        grand_mean_inf, yerr_for_plot_inf = calc_mean_and_ci_over_runs(avg_inf_over_gens_matrix, n_runs, n_copies, low_cut_off, high_cut_off)
        print ''
        print "grand_mean_inf is:"
        print grand_mean_inf
        print ''
        print 'yerr_for_plot_inf is:'
        print yerr_for_plot_inf

        mean_inf_per_bottleneck.append(grand_mean_inf)
        yerrs_inf_per_bottleneck.append(yerr_for_plot_inf)

    print ''
    print ''
    print "mean_fitness_per_bottleneck is:"
    print mean_fitness_per_bottleneck
    print ''
    print ''
    print "mean_inf_per_bottleneck is:"
    print mean_inf_per_bottleneck


#####################################################################################
## GENERATING THE PLOT:

    plot_file_path = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Plots/Iteration/'


    if selection_type == 'ca_with_parent':
        if context_generation == 'random':
            plot_file_title = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            plot_file_title = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    elif selection_type == 'none' or selection_type == 'p_taking':
        if context_generation == 'random':
            plot_file_title = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            plot_file_title = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type



    if selection_type != 'none':


        if selection_type == 'none':
            plot_title = 'No selection'
        elif selection_type == 'ca_with_parent':
            plot_title = 'Selection for communication'
        elif selection_type == 'p_taking':
            plot_title = 'Selection on perspective-inference'


        plot_file_title_specific = 'Barplot_Fitness_Diff_Bottlenecks_'+plot_file_title


        plot_title = 'Fitness per bottleneck'

        if selection_type == 'none':
            plot_title = 'No selection'
        elif selection_type == 'ca_with_parent':
            plot_title = 'Selection for communication'
        elif selection_type == 'p_taking':
            plot_title = 'Selection on perspective-inference'

        x_label = 'bottleneck size (=amount of data transmitted)'

        y_label = 'average fitness of population'

        make_barplot(plot_file_path, plot_file_title_specific, plot_title, bottleneck_sizes.astype(str), x_label, y_label, text_size, high_cut_off, mean_fitness_per_bottleneck, yerrs_fitness_per_bottleneck, selection_type, max_fitness, legend=legend, error_bars=error_bars)



        plot_file_title_specific = 'Boxplot_Fitness_Diff_Bottlenecks_' + plot_file_title

        make_boxplot(plot_file_path, plot_file_title_specific, plot_title, bottleneck_sizes.astype(str), x_label, y_label, text_size, high_cut_off, avg_fitness_matrix_all_runs_per_bottleneck, selection_type, max_fitness, legend=legend)

        plot_file_title_specific = 'Violinplot_Fitness_Diff_Bottlenecks_' + plot_file_title

        make_violinplot(plot_file_path, plot_file_title_specific, plot_title, bottleneck_sizes.astype(str), x_label, y_label, text_size, high_cut_off, avg_fitness_matrix_all_runs_per_bottleneck, selection_type, max_fitness, legend=legend)





        means_blank = np.full_like(mean_fitness_per_bottleneck, -2.)
        yerrs_blank = np.zeros_like(yerrs_fitness_per_bottleneck)


        plot_file_title_specific = 'Barplot_Fitness_Blank_'+lex_measure+'_'+selection_type

        make_barplot(plot_file_path, plot_file_title_specific, plot_title, bottleneck_sizes.astype(str), x_label, y_label, text_size, high_cut_off, means_blank, yerrs_blank, selection_type, max_fitness)



        #
        # avg_fitness_blank = np.full_like(avg_fitness_matrix_all_runs_per_bottleneck, -2.)
        #
        # plot_file_title_specific = 'Boxplot_Fitness_Blank_'+lex_measure+'_'+selection_type
        #

        # make_boxplot(plot_file_path, plot_file_title_specific, plot_title, bottleneck_sizes.astype(str), x_label, y_label, text_size, high_cut_off, avg_fitness_blank, selection_type, max_fitness)
        #
        #
        # plot_file_title_specific = 'Violinplot_Fitness_Blank_'+lex_measure+'_'+selection_type

        # make_violinplot(plot_file_path, plot_file_title_specific, plot_title, bottleneck_sizes.astype(str), x_label, y_label, text_size, high_cut_off, avg_fitness_blank, selection_type, max_fitness)
        #






    plot_file_title_specific = 'Barplot_Success_Diff_Bottlenecks_'+plot_file_title

    plot_title = 'Success per bottleneck'

    if selection_type == 'none':
        plot_title = 'No selection'
    elif selection_type == 'ca_with_parent':
        plot_title = 'Selection for communication'
    elif selection_type == 'p_taking':
        plot_title = 'Selection on perspective-inference'


    x_label = 'bottleneck size (=amount of data transmitted)'
    y_label = 'average success of population'

    chance_level_comm_success = 1./float(n_meanings)
    chance_level_p_taking_success = 1./float(len(perspective_hyps))

    if pragmatic_level == 'literal' or pragmatic_level == 'perspective-taking':
        if n_meanings == 2 and n_signals == 2:
            max_comm_success = (np.power((1.-error), 2)+np.power(error, 2))
        elif n_meanings == 3 and n_signals == 3:
            max_comm_success = (np.power((1.-error), 2)+np.power((error/2), 2)++np.power((error/3), 2))
        elif n_meanings == 4 and n_signals == 4:
            max_comm_success = (np.power((1.-error), 2)+np.power((error/3), 2)+np.power((error/3), 2)+np.power((error/3), 2))
        else:
            raise ValueError("Haven't calculated max_fitness for n_meanings > 4 or for asymmetrical lexicon")
    elif pragmatic_level == 'prag' and optimality_alpha == 3.0:
        max_comm_success = 1.0
    print ''
    print ''
    print "max_comm_success is:"
    print max_comm_success


    max_p_taking_success = 1.0
    print ''
    print ''
    print "max_p_taking_success is:"
    print max_p_taking_success


    make_barplot_two_conditions(plot_file_path, plot_file_title_specific, text_size, high_cut_off, mean_comm_success_per_bottleneck, yerrs_comm_success_per_bottleneck, mean_p_taking_success_per_bottleneck, yerrs_p_taking_success_per_bottleneck, max_comm_success, max_p_taking_success, legend, error_bars=error_bars)


    plot_file_title_specific = 'Boxplot_Success_Diff_Bottlenecks_' + plot_file_title

    # make_boxplot_two_conditions(plot_file_path, plot_file_title_specific, text_size, high_cut_off, avg_comm_success_matrix_all_runs_per_bottleneck, avg_p_taking_success_matrix_all_runs_per_bottleneck, max_comm_success, max_p_taking_success, legend)
    #

    if selection_type == 'none':
        plot_file_title_specific = 'Blank_MMIEL_Barplot_Success_Diff_Bottlenecks_'+plot_file_title

        means_blank = np.zeros_like(mean_comm_success_per_bottleneck)
        yerrs_blank = np.zeros_like(yerrs_comm_success_per_bottleneck)


        make_barplot_cogsci_london_poster(plot_file_path, plot_file_title_specific, text_size, high_cut_off, means_blank, yerrs_blank, means_blank, yerrs_blank, chance_level_comm_success, max_comm_success, chance_level_p_taking_success, max_p_taking_success)



    plot_file_title_specific = 'MMIEL_Barplot_Success_Diff_Bottlenecks_'+plot_file_title

    make_barplot_cogsci_london_poster(plot_file_path, plot_file_title_specific, text_size, high_cut_off, mean_comm_success_per_bottleneck, yerrs_comm_success_per_bottleneck, mean_p_taking_success_per_bottleneck, yerrs_p_taking_success_per_bottleneck, chance_level_comm_success, max_comm_success, chance_level_p_taking_success, max_p_taking_success)


    plot_file_title_specific = 'MMIEL_Boxplot_Success_Diff_Bottlenecks_' + plot_file_title

    # make_boxplot_cogsci_london_poster(plot_file_path, plot_file_title_specific, text_size, high_cut_off, avg_comm_success_matrix_all_runs_per_bottleneck, avg_p_taking_success_matrix_all_runs_per_bottleneck, max_comm_success, max_p_taking_success)



    plot_title = 'Informativeness per bottleneck'

    if selection_type == 'none':
        plot_title = 'No selection'
    elif selection_type == 'ca_with_parent':
        plot_title = 'Selection for communication'
    elif selection_type == 'p_taking':
        plot_title = 'Selection on perspective-inference'


    x_label = 'bottleneck size (=amount of data transmitted)'

    y_label = 'average informativeness of population'



    plot_file_title_specific = 'Barplot_Inf_Diff_Bottlenecks_'+lex_measure+'_'+plot_file_title

    make_barplot(plot_file_path, plot_file_title_specific, plot_title, bottleneck_sizes.astype(str), x_label, y_label, text_size, high_cut_off, mean_inf_per_bottleneck, yerrs_inf_per_bottleneck, selection_type, max_inf, baseline_inf, legend, error_bars=error_bars)


    plot_file_title_specific = 'Boxplot_Inf_Diff_Bottlenecks_'+lex_measure+'_'+plot_file_title

    # make_boxplot(plot_file_path, plot_file_title_specific, plot_title, bottleneck_sizes.astype(str), x_label, y_label, text_size, high_cut_off, avg_inf_matrix_all_runs_per_bottleneck, selection_type, max_inf, baseline_inf, legend)

    plot_file_title_specific = 'Violinplot_Inf_Diff_Bottlenecks_'+lex_measure+'_'+plot_file_title

    # make_violinplot(plot_file_path, plot_file_title_specific, plot_title, bottleneck_sizes.astype(str), x_label, y_label, text_size, high_cut_off, avg_inf_matrix_all_runs_per_bottleneck, selection_type, max_inf, baseline_inf, legend)



    plot_file_title_specific = 'Barplot_Inf_plus_Two_Conditions_'+lex_measure+'_'+plot_file_title



    make_barplot_three_conditions(plot_file_path, plot_file_title_specific, high_cut_off, mean_inf_per_bottleneck, yerrs_inf_per_bottleneck, mean_comm_success_per_bottleneck, yerrs_comm_success_per_bottleneck, mean_p_taking_success_per_bottleneck, yerrs_p_taking_success_per_bottleneck, baseline_inf, max_inf, max_comm_success, max_p_taking_success, legend, error_bars=error_bars)



    means_blank = np.full_like(mean_inf_per_bottleneck, -2.)
    yerrs_blank = np.zeros_like(yerrs_inf_per_bottleneck)


    plot_file_title_specific = 'Barplot_Inf_Blank_'+lex_measure+'_'+selection_type

    make_barplot(plot_file_path, plot_file_title_specific, plot_title, bottleneck_sizes.astype(str), x_label, y_label, text_size, high_cut_off, means_blank, yerrs_blank, selection_type, max_inf, baseline_inf)


    #
    # avg_inf_blank = np.full_like(avg_inf_matrix_all_runs_per_bottleneck, -2.)
    #
    # plot_file_title_specific = 'Boxplot_Inf_Blank_'+lex_measure+'_'+selection_type
    #
    # make_boxplot(plot_file_path, plot_file_title_specific, plot_title, bottleneck_sizes.astype(str), x_label, y_label, text_size, high_cut_off, avg_inf_blank, selection_type, max_inf, baseline_inf)
    #
    #
    # plot_file_title_specific = 'Violinplot_Inf_Blank_'+lex_measure+'_'+selection_type
    #
    # make_violinplot(plot_file_path, plot_file_title_specific, plot_title, bottleneck_sizes.astype(str), x_label, y_label, text_size, high_cut_off, avg_inf_blank, selection_type, max_inf, baseline_inf)






















    #
    #
    # selection_type = 'none'
    #
    # if selection_type == 'ca_with_parent':
    #     if n_meanings == 2 and n_signals == 2:
    #         max_fitness = (np.power((1.-error), 2)+np.power(error, 2))
    #     elif n_meanings == 3 and n_signals == 3:
    #         max_fitness = (np.power((1.-error), 2)+np.power((error/2), 2)++np.power((error/3), 2))
    #     elif n_meanings == 4 and n_signals == 4:
    #         max_fitness = (np.power((1.-error), 2)+np.power((error/3), 2)+np.power((error/3), 2)+np.power((error/3), 2))
    #     else:
    #         raise ValueError("Haven't calculated max_fitness for n_meanings > 4 or for asymmetrical lexicon")
    #     print ''
    #     print ''
    #     print "max_fitness is:"
    #     print max_fitness
    #
    # elif selection_type == 'p_taking':
    #     # if perspective_prior_type == 'neutral':
    #     #     fitness_baseline = 1./perspective_hyps
    #     # elif perspective_prior_type == 'egocentric':
    #     #     fitness_baseline = 1.-perspective_prior_strength
    #
    #     max_fitness = 1.0
    #     print ''
    #     print ''
    #     print "max_fitness is:"
    #     print max_fitness
    #
    #
    #
    # print "n_copies is:"
    # print n_copies
    # avg_comm_success_matrix_all_runs_per_bottleneck = np.zeros((len(bottleneck_sizes), (n_runs*n_copies)))
    # avg_p_taking_success_matrix_all_runs_per_bottleneck = np.zeros((len(bottleneck_sizes), (n_runs * n_copies)))
    # avg_fitness_matrix_all_runs_per_bottleneck = np.zeros((len(bottleneck_sizes), (n_runs*n_copies)))
    # avg_inf_matrix_all_runs_per_bottleneck_select_none = np.zeros((len(bottleneck_sizes), (n_runs*n_copies)))
    # mean_comm_success_per_bottleneck = []
    # yerrs_comm_success_per_bottleneck = []
    # mean_p_taking_success_per_bottleneck = []
    # yerrs_p_taking_success_per_bottleneck = []
    # mean_fitness_per_bottleneck = []
    # yerrs_fitness_per_bottleneck = []
    # mean_inf_per_bottleneck = []
    # yerrs_inf_per_bottleneck = []
    # for i in range(len(bottleneck_sizes)):
    #     n_contexts = bottleneck_sizes[i]
    #
    #     print ''
    #     print ''
    #     print 'This is bottleneck size:'
    #     print n_contexts
    #
    #     n_copies = n_copies_array[i]
    #     print "n_copies is:"
    #     print n_copies
    #
    #     avg_inf_over_gens_matrix, mean_informativity_over_gens, conf_invs_informativity_over_gens, percentiles_informativity_over_gens, baseline_inf, max_inf = get_mean_percentiles_avg_informativeness(pickle_directory, n_runs, n_copies, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type)
    #
    #     avg_inf_matrix_excerpt = avg_inf_over_gens_matrix[:, low_cut_off:high_cut_off]
    #
    #     avg_inf_excerpt_mean_over_gens = np.mean(avg_inf_matrix_excerpt, axis=1)
    #
    #     if n_copies != 20:
    #         nans = ['nan' for x in range(20-n_copies)]
    #         nans = np.array(nans)
    #         avg_inf_excerpt_mean_over_gens = np.hstack((avg_inf_excerpt_mean_over_gens, nans))
    #
    #     avg_inf_matrix_all_runs_per_bottleneck_select_none[i] = avg_inf_excerpt_mean_over_gens
    #
    #     grand_mean_inf, yerr_for_plot_inf = calc_mean_and_ci_over_runs(avg_inf_over_gens_matrix, n_runs, n_copies, low_cut_off, high_cut_off)
    #     print ''
    #     print "grand_mean_inf is:"
    #     print grand_mean_inf
    #     print ''
    #     print 'yerr_for_plot_inf is:'
    #     print yerr_for_plot_inf
    #
    #     mean_inf_per_bottleneck.append(grand_mean_inf)
    #     yerrs_inf_per_bottleneck.append(yerr_for_plot_inf)
    #
    # print ''
    # print ''
    # print "avg_inf_matrix_all_runs_per_bottleneck_select_none is:"
    # print avg_inf_matrix_all_runs_per_bottleneck_select_none
    #
    #
    #
    #
    #
    # selection_type = 'ca_with_parent'
    #
    # if selection_type == 'ca_with_parent':
    #     if n_meanings == 2 and n_signals == 2:
    #         max_fitness = (np.power((1.-error), 2)+np.power(error, 2))
    #     elif n_meanings == 3 and n_signals == 3:
    #         max_fitness = (np.power((1.-error), 2)+np.power((error/2), 2)++np.power((error/3), 2))
    #     elif n_meanings == 4 and n_signals == 4:
    #         max_fitness = (np.power((1.-error), 2)+np.power((error/3), 2)+np.power((error/3), 2)+np.power((error/3), 2))
    #     else:
    #         raise ValueError("Haven't calculated max_fitness for n_meanings > 4 or for asymmetrical lexicon")
    #     print ''
    #     print ''
    #     print "max_fitness is:"
    #     print max_fitness
    #
    # elif selection_type == 'p_taking':
    #     # if perspective_prior_type == 'neutral':
    #     #     fitness_baseline = 1./perspective_hyps
    #     # elif perspective_prior_type == 'egocentric':
    #     #     fitness_baseline = 1.-perspective_prior_strength
    #
    #     max_fitness = 1.0
    #     print ''
    #     print ''
    #     print "max_fitness is:"
    #     print max_fitness
    #
    #
    #
    # print "n_copies is:"
    # print n_copies
    # avg_comm_success_matrix_all_runs_per_bottleneck = np.zeros((len(bottleneck_sizes), (n_runs*n_copies)))
    # avg_p_taking_success_matrix_all_runs_per_bottleneck = np.zeros((len(bottleneck_sizes), (n_runs * n_copies)))
    # avg_fitness_matrix_all_runs_per_bottleneck = np.zeros((len(bottleneck_sizes), (n_runs*n_copies)))
    # avg_inf_matrix_all_runs_per_bottleneck_select_ca = np.zeros((len(bottleneck_sizes), (n_runs*n_copies)))
    # mean_comm_success_per_bottleneck = []
    # yerrs_comm_success_per_bottleneck = []
    # mean_p_taking_success_per_bottleneck = []
    # yerrs_p_taking_success_per_bottleneck = []
    # mean_fitness_per_bottleneck = []
    # yerrs_fitness_per_bottleneck = []
    # mean_inf_per_bottleneck = []
    # yerrs_inf_per_bottleneck = []
    # for i in range(len(bottleneck_sizes)):
    #     n_contexts = bottleneck_sizes[i]
    #
    #     print ''
    #     print ''
    #     print 'This is bottleneck size:'
    #     print n_contexts
    #
    #     n_copies = n_copies_array[i]
    #     print "n_copies is:"
    #     print n_copies
    #
    #     avg_inf_over_gens_matrix, mean_informativity_over_gens, conf_invs_informativity_over_gens, percentiles_informativity_over_gens, baseline_inf, max_inf = get_mean_percentiles_avg_informativeness(pickle_directory, n_runs, n_copies, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type)
    #
    #     avg_inf_matrix_excerpt = avg_inf_over_gens_matrix[:, low_cut_off:high_cut_off]
    #
    #     avg_inf_excerpt_mean_over_gens = np.mean(avg_inf_matrix_excerpt, axis=1)
    #
    #     if n_copies != 20:
    #         nans = ['nan' for x in range(20-n_copies)]
    #         nans = np.array(nans)
    #         avg_inf_excerpt_mean_over_gens = np.hstack((avg_inf_excerpt_mean_over_gens, nans))
    #
    #     avg_inf_matrix_all_runs_per_bottleneck_select_ca[i] = avg_inf_excerpt_mean_over_gens
    #
    #     grand_mean_inf, yerr_for_plot_inf = calc_mean_and_ci_over_runs(avg_inf_over_gens_matrix, n_runs, n_copies, low_cut_off, high_cut_off)
    #     print ''
    #     print "grand_mean_inf is:"
    #     print grand_mean_inf
    #     print ''
    #     print 'yerr_for_plot_inf is:'
    #     print yerr_for_plot_inf
    #
    #     mean_inf_per_bottleneck.append(grand_mean_inf)
    #     yerrs_inf_per_bottleneck.append(yerr_for_plot_inf)
    #
    # print ''
    # print ''
    # print "avg_inf_matrix_all_runs_per_bottleneck_select_ca is:"
    # print avg_inf_matrix_all_runs_per_bottleneck_select_ca
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # selection_type = 'p_taking'
    #
    # if selection_type == 'ca_with_parent':
    #     if n_meanings == 2 and n_signals == 2:
    #         max_fitness = (np.power((1.-error), 2)+np.power(error, 2))
    #     elif n_meanings == 3 and n_signals == 3:
    #         max_fitness = (np.power((1.-error), 2)+np.power((error/2), 2)++np.power((error/3), 2))
    #     elif n_meanings == 4 and n_signals == 4:
    #         max_fitness = (np.power((1.-error), 2)+np.power((error/3), 2)+np.power((error/3), 2)+np.power((error/3), 2))
    #     else:
    #         raise ValueError("Haven't calculated max_fitness for n_meanings > 4 or for asymmetrical lexicon")
    #     print ''
    #     print ''
    #     print "max_fitness is:"
    #     print max_fitness
    #
    # elif selection_type == 'p_taking':
    #     # if perspective_prior_type == 'neutral':
    #     #     fitness_baseline = 1./perspective_hyps
    #     # elif perspective_prior_type == 'egocentric':
    #     #     fitness_baseline = 1.-perspective_prior_strength
    #
    #     max_fitness = 1.0
    #     print ''
    #     print ''
    #     print "max_fitness is:"
    #     print max_fitness
    #
    #
    #
    # print "n_copies is:"
    # print n_copies
    # avg_comm_success_matrix_all_runs_per_bottleneck = np.zeros((len(bottleneck_sizes), (n_runs*n_copies)))
    # avg_p_taking_success_matrix_all_runs_per_bottleneck = np.zeros((len(bottleneck_sizes), (n_runs * n_copies)))
    # avg_fitness_matrix_all_runs_per_bottleneck = np.zeros((len(bottleneck_sizes), (n_runs*n_copies)))
    # avg_inf_matrix_all_runs_per_bottleneck_select_p_taking = np.zeros((len(bottleneck_sizes), (n_runs*n_copies)))
    # mean_comm_success_per_bottleneck = []
    # yerrs_comm_success_per_bottleneck = []
    # mean_p_taking_success_per_bottleneck = []
    # yerrs_p_taking_success_per_bottleneck = []
    # mean_fitness_per_bottleneck = []
    # yerrs_fitness_per_bottleneck = []
    # mean_inf_per_bottleneck = []
    # yerrs_inf_per_bottleneck = []
    # for i in range(len(bottleneck_sizes)):
    #     n_contexts = bottleneck_sizes[i]
    #
    #     print ''
    #     print ''
    #     print 'This is bottleneck size:'
    #     print n_contexts
    #
    #     n_copies = n_copies_array[i]
    #     print "n_copies is:"
    #     print n_copies
    #
    #     avg_inf_over_gens_matrix, mean_informativity_over_gens, conf_invs_informativity_over_gens, percentiles_informativity_over_gens, baseline_inf, max_inf = get_mean_percentiles_avg_informativeness(pickle_directory, n_runs, n_copies, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type)
    #
    #     avg_inf_matrix_excerpt = avg_inf_over_gens_matrix[:, low_cut_off:high_cut_off]
    #
    #     avg_inf_excerpt_mean_over_gens = np.mean(avg_inf_matrix_excerpt, axis=1)
    #
    #     if n_copies != 20:
    #         nans = ['nan' for x in range(20-n_copies)]
    #         nans = np.array(nans)
    #         avg_inf_excerpt_mean_over_gens = np.hstack((avg_inf_excerpt_mean_over_gens, nans))
    #
    #     avg_inf_matrix_all_runs_per_bottleneck_select_p_taking[i] = avg_inf_excerpt_mean_over_gens
    #
    #     grand_mean_inf, yerr_for_plot_inf = calc_mean_and_ci_over_runs(avg_inf_over_gens_matrix, n_runs, n_copies, low_cut_off, high_cut_off)
    #     print ''
    #     print "grand_mean_inf is:"
    #     print grand_mean_inf
    #     print ''
    #     print 'yerr_for_plot_inf is:'
    #     print yerr_for_plot_inf
    #
    #     mean_inf_per_bottleneck.append(grand_mean_inf)
    #     yerrs_inf_per_bottleneck.append(yerr_for_plot_inf)
    #
    # print ''
    # print ''
    # print "avg_inf_matrix_all_runs_per_bottleneck_select_p_taking is:"
    # print avg_inf_matrix_all_runs_per_bottleneck_select_p_taking
    #
    #
    #
    # boxplot_three_cond(plot_file_path, plot_file_title_specific, plot_title, 'Avg. informativeness', text_size, low_cut_off, high_cut_off, avg_inf_matrix_all_runs_per_bottleneck_none, avg_inf_matrix_all_runs_per_bottleneck_select_p_taking, avg_inf_matrix_all_runs_per_bottleneck_select_ca, baseline_inf, max_inf, legend)


