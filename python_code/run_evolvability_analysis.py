__author__ = 'Marieke Woensdregt'


import numpy as np
from scipy import stats
import time

import hypspace
import lex
import pickle
import pop
from saveresults import convert_array_to_string


# np.set_printoptions(threshold=np.nan)


#######################################################################################################################
# 1: THE PARAMETERS:


##!!!!!! MAKE SURE TO CHANGE THE PATHS BELOW TO MATCH THE FILE SYSTEM OF YOUR MACHINE:
# iteration_results_directory = '/exports/eddie/scratch/s1370641/'

iteration_results_directory = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Eddie_Output/'

output_pickle_file_directory = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/Evolvability_Analysis/'


# 1.1: The parameters defining the lexicon size (and thus the number of meanings in the world):

n_meanings = 3  # The number of meanings
n_signals = 3  # The number of signals

# 1.2: The parameters defining the contexts and how they map to the agent's saliencies:

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
    helpful_contexts = np.array([[0.1, 0.2, 0.3, 0.9], [0.1, 0.3, 0.6, 0.7],
                                 [0.1, 0.2, 0.9, 0.3], [0.1, 0.3, 0.7, 0.6],
                                 [0.1, 0.3, 0.2, 0.9], [0.1, 0.6, 0.3, 0.7],
                                 [0.1, 0.3, 0.9, 0.2], [0.1, 0.6, 0.7, 0.3],
                                 [0.1, 0.9, 0.2, 0.3], [0.1, 0.7, 0.3, 0.6],
                                 [0.1, 0.9, 0.3, 0.2], [0.1, 0.7, 0.6, 0.3],
                                 [0.2, 0.1, 0.3, 0.9], [0.3, 0.1, 0.6, 0.7],
                                 [0.2, 0.1, 0.9, 0.3], [0.3, 0.1, 0.7, 0.6],
                                 [0.2, 0.3, 0.1, 0.9], [0.3, 0.6, 0.1, 0.7],
                                 [0.2, 0.3, 0.9, 0.1], [0.3, 0.6, 0.7, 0.1],
                                 [0.2, 0.9, 0.1, 0.3], [0.3, 0.7, 0.1, 0.6],
                                 [0.2, 0.9, 0.3, 0.1], [0.3, 0.7, 0.6, 0.1],
                                 [0.3, 0.1, 0.2, 0.9], [0.6, 0.1, 0.3, 0.7],
                                 [0.3, 0.1, 0.9, 0.2], [0.6, 0.1, 0.7, 0.3],
                                 [0.3, 0.2, 0.1, 0.9], [0.6, 0.3, 0.1, 0.7],
                                 [0.3, 0.2, 0.9, 0.1], [0.6, 0.3, 0.7, 0.1],
                                 [0.9, 0.1, 0.2, 0.3], [0.6, 0.7, 0.1, 0.3],
                                 [0.9, 0.1, 0.3, 0.2], [0.6, 0.7, 0.3, 0.1],
                                 [0.9, 0.2, 0.1, 0.3], [0.7, 0.1, 0.3, 0.6],
                                 [0.9, 0.2, 0.3, 0.1], [0.7, 0.1, 0.6, 0.3],
                                 [0.9, 0.3, 0.1, 0.2], [0.7, 0.3, 0.1, 0.6],
                                 [0.9, 0.3, 0.2, 0.1], [0.7, 0.3, 0.6, 0.1]])

context_type = 'continuous' # This can be set to either 'absolute' for meanings being in/out or 'continuous' for all meanings being present (but in different positions)
context_size = 1 # This parameter is only used if the context_type is 'absolute' and determines the number of meanings present

sal_alpha = 1. # The exponent that is used by the saliency function. sal_alpha=1 gives a directly proportional relationship between meaning distances and meaning saliencies, with sal_alpha>1 the saliency of meanings goes down exponentially as their distance from the agent increases (and with sal_alpha<1 the other way around)


error = 0.05  # The error term on production
error_string = convert_array_to_string(error)
extra_error = True # Determines whether the error specified above gets added on top AFTER the pragmatic speaker has calculated its production probabilities by maximising the utility to the listener.


# 1.3: The parameters that determine the make-up of the population:

pop_size_initial_pop = 100
pop_size_mixed_pop = 10
n_mutants = 1  # the number of mutants initially inserted into the first generation of the mixed population


agent_type = 'no_p_distinction'

lexicon_types = ['optimal_lex', 'half_ambiguous_lex', 'fully_ambiguous_lex'] # The lexicon types that can be present in the population
#FIXME: In the current implementation 'half_ambiguous_lex' can have different instantiations. Therefore, if we run a simulation where there are different lexicon_type_probs, we will want the run_type to be 'population_same_pop' to make sure that all the 'half ambiguous' speakers do have the SAME half ambiguous lexicon.
lexicon_type_probs = np.array([0., 0., 1.]) # The ratios with which the different lexicon types will be present in the population
lexicon_type_probs_string = convert_array_to_string(lexicon_type_probs) # Turns the lexicon type probs into a string in order to add it to file names


perspectives = np.array([0., 1.]) # The different perspectives that agents can have
perspective_probs = np.array([0., 1.]) # The ratios with which the different perspectives will be present in the population
perspective_probs_string = convert_array_to_string(perspective_probs) # Turns the perspective probs into a string in order to add it to file names


learning_types = ['map', 'sample'] # The types of learning that the learners can do
learning_type_probs = np.array([0., 1.]) # The ratios with which the different learning types will be present in the population
learning_type_probs_string = convert_array_to_string(learning_type_probs) # Turns the learning type probs into a string in order to add it to file names
if learning_type_probs[0] == 1.:
    learning_type_string = learning_types[0]
elif learning_type_probs[1] == 1.:
    learning_type_string = learning_types[1]
#learning_type_string = learning_types[np.where(learning_type_probs==1.)[0]]


pragmatic_level_initial_pop = 'literal'  # This can be set to either 'literal', 'perspective-taking' or 'prag'
optimality_alpha_initial_pop = 1.0  # Goodman & Stuhlmuller (2013) fitted sal_alpha = 3.4 to participant data with 4x3 lexicon of three number words ('one', 'two', and 'three') that could be mapped to four different world states (0, 1, 2, and 3)


pragmatic_level_mutants = 'prag'  # This can be set to either 'literal', 'perspective-taking' or 'prag'
optimality_alpha_mutants = 3.0  # Goodman & Stuhlmuller (2013) fitted sal_alpha = 3.4 to participant data with 4x3 lexicon of three number words ('one', 'two', and 'three') that could be mapped to four different world states (0, 1, 2, and 3)


pragmatic_level_parent_hyp = pragmatic_level_mutants  # This can be set to either 'literal', 'perspective-taking' or 'prag'

teacher_type = 'sng_teacher'  # This can be set to either 'sng_teacher' or 'multi_teacher'



# 1.4: The parameters that determine the learner's hypothesis space:

perspective_hyps = np.array([0., 1.]) # The perspective hypotheses that the learner will consider (1D numpy array)

which_lexicon_hyps = 'all' # Determines the set of lexicon hypotheses that the learner will consider. This can be set to either 'all', 'all_with_full_s_space' or 'only_optimal'
if which_lexicon_hyps == 'all':
    lexicon_hyps = hypspace.create_all_lexicons(n_meanings, n_signals) # The lexicon hypotheses that the learner will consider (1D numpy array)
elif which_lexicon_hyps == 'all_with_full_s_space':
    all_lexicon_hyps = hypspace.create_all_lexicons(n_meanings, n_signals)
    lexicon_hyps = hypspace.remove_subset_of_signals_lexicons(all_lexicon_hyps) # The lexicon hypotheses that the learner will consider (1D numpy array)
elif which_lexicon_hyps == 'only_optimal':
    lexicon_hyps = hypspace.create_all_optimal_lexicons(n_meanings, n_signals) # The lexicon hypotheses that the learner will consider (1D numpy array)


hypothesis_space = hypspace.list_hypothesis_space(perspective_hyps, lexicon_hyps) # The full space of composite hypotheses that the learner will consider (2D numpy matrix with composite hypotheses on the rows, perspective hypotheses on column 0 and lexicon hypotheses on column 1)


# 1.5: The parameters that determine the learner's prior:

learner_perspective = 0.  # The learner's perspective

perspective_prior_type = 'egocentric' # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
perspective_prior_strength = 0.9 # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
perspective_prior_strength_string = convert_array_to_string(perspective_prior_strength)



lexicon_prior_type = 'neutral' # This can be set to either 'neutral', 'ambiguous_fixed', 'half_ambiguous_fixed', 'expressivity_bias' or 'compressibility_bias'
lexicon_prior_constant = 0.0 # Determines the strength of the lexicon bias, with small c creating a STRONG prior and large c creating a WEAK prior. (And with c = 1000 creating an almost uniform prior.)
lexicon_prior_constant_string = convert_array_to_string(lexicon_prior_constant)

n_utterances = 1  # This parameter determines how many signals the learner gets to observe in each context
n_contexts = 60  # The number of contexts that the learner gets to see. If context_generation = 'optimal' n_contexts for n_meanings = 2 can be anything in the table of 4 (e.g. 20, 40, 60, 80, 100, etc.); for n_meanings = 3 anything in the table of 12 (e.g. 12, 36, 60, 84, 108, etc.); and for n_meanings = 4 anything in the table of 48 (e.g. 48, 96, 144, 192 etc.).
n_contexts_original = 60

turnover_type = 'whole_pop'  # The type of turnover with which the population is replaced (for the iteration simulation). This can be set to either 'chain' (one agent at a time) or 'whole_pop' (entire population at once)


communication_type_initial_pop = 'lex_n_p'  # This can be set to either 'lex_only', 'lex_n_context' or 'lex_n_p' or 'prag'
ca_measure_type_initial_pop = 'comp_only'  # This can be set to either "comp_n_prod" or "comp_only"

communication_type_mutants = 'prag'  # This can be set to either 'lex_only', 'lex_n_context' or 'lex_n_p' or 'prag'
ca_measure_type_mutants = 'comp_only'  # This can be set to either "comp_n_prod" or "comp_only"
n_interactions = 6  # The number of interactions used to calculate communicative accuracy

selection_type = 'none'  # This can be set to either 'none', 'p_taking', 'l_learning' or 'ca_with_parent'
selection_weighting = 'none'  # This is a factor with which the fitness of the agents (determined as the probability they assign to the correct perspective hypothesis) is multiplied and then exponentiated in order to weight the relative agent fitness (which in turn determines the probability of becoming a teacher for the next generation). A value of 0. implements neutral selection. A value of 1.0 creates weighting where the fitness is pretty much equal to relative posterior probability on correct p hyp), and the higher the value, the more skewed the weighting in favour of agents with better perspective-taking.
if isinstance(selection_weighting, float):
    selection_weight_string = str(np.int(selection_weighting))
else:
    selection_weight_string = selection_weighting


n_iterations_initial_pop = 500  # The number of iterations (i.e. new agents in the case of 'chain', generations in the case of 'whole_pop')
n_iterations_mixed_pop = 10  # The number of iterations (i.e. new agents in the case of 'chain', generations in the case of 'whole_pop')
report_every_i = 1
cut_off_point = 5
n_runs_initial_pop = 1  # The number of runs of the simulation
report_every_r = 1

which_hyps_on_graph = 'lex_hyps_collapsed'  # This is used in the plot_post_probs_over_lex_hyps() function, and can be set to either 'all_hyps', 'lex_hyps_only' or 'lex_hyps_collapsed'

posterior_threshold = 0.99  # This threshold determines how much posterior probability a learner needs to have assigned to the (set of) correct hypothesis/hypotheses in order to say they have 'learned' the correct hypothesis.

lex_measure = 'ca'  # This can be set to either 'mi' for mutual information or 'ca' for communicative accuracy (of the lexicon with itself)


run_type = 'iter'


decoupling = True  # This can be set to either True or False. It determines whether genetic and cultural inheritance are coupled (i.e. from the same cultural parent) or decoupled.


run_number = 1


#######################################################################################################################



def get_final_pop_lex_indices(directory, copy_specification, n_runs, n_iterations, n_contexts_original, pop_size, pragmatic_level, optimality_alpha, communication_type, ca_measure_type):


    if selection_type == 'none' or selection_type == 'p_taking':
        folder_name = 'results_'+run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S_select_'+selection_type+'_p_prior_'+perspective_prior_type[:4]+'_'+perspective_prior_strength_string+'_l_prior_'+lexicon_prior_type[:4]+'_'+pragmatic_level_initial_pop+'_a_'+str(int(optimality_alpha_initial_pop))+'/'
    elif selection_type == 'ca_with_parent':
        folder_name = 'results_'+run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S_select_'+selection_type+'_'+communication_type_initial_pop+'_'+ca_measure_type_initial_pop+'_p_prior_'+perspective_prior_type[:4]+'_'+perspective_prior_strength_string+'_l_prior_'+lexicon_prior_type[:4]+'_'+pragmatic_level_initial_pop+'_a_'+str(int(optimality_alpha_initial_pop))+'/'



    if selection_type == 'none' or selection_type == 'p_taking':
        if context_generation == 'random':
            filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_select_'+selection_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts_original)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type+copy_specification
        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_select_'+selection_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts_original)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type+copy_specification
    elif selection_type == 'ca_with_parent':
        if context_generation == 'random':
            filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts_original)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type+copy_specification

        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts_original)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type+copy_specification


    pickle_filename_all_results = 'Results_'+filename
    results_dict = pickle.load(open(directory+folder_name+pickle_filename_all_results+'.p', 'rb'))

    multi_run_selected_hyps_per_generation_matrix = results_dict[
        'multi_run_selected_hyps_per_generation_matrix']
    print ''
    print ''
    print "multi_run_selected_hyps_per_generation_matrix.shape is:"
    print multi_run_selected_hyps_per_generation_matrix.shape
    selected_hyps_final_gen = multi_run_selected_hyps_per_generation_matrix[0][-1]
    print ''
    print "selected_hyps_final_gen BEFORE CONVERSION are:"
    print selected_hyps_final_gen

    for h in range(len(selected_hyps_final_gen)):
        if selected_hyps_final_gen[h] >= len(lexicon_hyps):
            selected_hyps_final_gen[h] = selected_hyps_final_gen[h] - len(lexicon_hyps)
    print ''
    print "selected_hyps_final_gen AFTER CONVERSION are:"
    print selected_hyps_final_gen

    bin_count_selected_lex_hyps = np.bincount(selected_hyps_final_gen.astype(int))
    print ''
    print ''
    print "bin_count_selected_lex_hyps is:"
    print bin_count_selected_lex_hyps

    lex_variation = np.count_nonzero(bin_count_selected_lex_hyps)
    print "lex_variation is:"
    print lex_variation
    for i in range(len(bin_count_selected_lex_hyps)):
        if bin_count_selected_lex_hyps[i] > 0:
            print ''
            print "YAY! bin_count_selected_lex_hyps[i] > 0"
            print 'i is:'
            print i
            lexicon = lexicon_hyps[i]
            print "lexicon is:"
            print lexicon
            lex_frequency = bin_count_selected_lex_hyps[i]
            print "lex_frequency is:"
            print lex_frequency


    return selected_hyps_final_gen



def get_selected_hyps_ordered(n_runs, n_iterations, pop_size, multi_run_selected_hyps_per_generation_matrix, hyp_order):
            selected_hyps_new_lex_order_all_runs = np.zeros_like(multi_run_selected_hyps_per_generation_matrix)
            for r in range(n_runs):
                for i in range(n_iterations):
                    for a in range(pop_size):
                        this_agent_hyp = multi_run_selected_hyps_per_generation_matrix[r][i][a]
                        if this_agent_hyp >= len(hyp_order):
                            this_agent_hyp = this_agent_hyp-len(hyp_order)
                        new_order_index = np.argwhere(hyp_order == this_agent_hyp)
                        selected_hyps_new_lex_order_all_runs[r][i][a] = new_order_index
            return selected_hyps_new_lex_order_all_runs


def calc_mean_and_conf_invs_distribution(n_runs, n_copies, lexicon_hyps, which_hyps_on_graph, min_info_indices, intermediate_info_indices, max_info_indices, n_iterations, cut_off_point, selected_hyps_new_lex_order_all_runs):
    hist_values_averaged_per_run = np.zeros((n_runs, 3))
    for r in range(n_runs):
        selected_hyps_new_lex_order_final_section = selected_hyps_new_lex_order_all_runs[r][cut_off_point:n_iterations]
        selected_hyps_new_lex_order_final_section = selected_hyps_new_lex_order_final_section.flatten()
        if which_hyps_on_graph == 'lex_hyps_only' or which_hyps_on_graph == 'lex_hyps_collapsed':
            n_lex_hyps = len(lexicon_hyps)
            for i in range(len(selected_hyps_new_lex_order_final_section)):
                hyp_index = selected_hyps_new_lex_order_final_section[i]
                if hyp_index > (n_lex_hyps - 1):
                    selected_hyps_new_lex_order_final_section[i] = hyp_index - n_lex_hyps
        hist_values = np.histogram(selected_hyps_new_lex_order_final_section, bins=[0, min_info_indices[-1]+1, intermediate_info_indices[-1]+1, max_info_indices[-1]])
        hist_values_averaged = np.divide(hist_values[0].astype(float), [float(len(min_info_indices)), float(len(intermediate_info_indices)), float(len(max_info_indices))])
        hist_values_averaged_per_run[r] = hist_values_averaged
    hist_values_averaged_per_run_optimal_first = np.fliplr(hist_values_averaged_per_run)
    mean_selected_hyps_by_lex_type = np.mean(hist_values_averaged_per_run_optimal_first, axis=0)
    std_selected_hyps_by_lex_type = np.std(hist_values_averaged_per_run_optimal_first, axis=0)
    conf_invs_selected_hyps_by_lex_type = stats.norm.interval(0.95, loc=mean_selected_hyps_by_lex_type, scale=std_selected_hyps_by_lex_type / np.sqrt(n_runs*n_copies))
    conf_invs_selected_hyps_by_lex_type = np.array(conf_invs_selected_hyps_by_lex_type)
    lower_yerr_selected_hyps_for_plot = np.subtract(mean_selected_hyps_by_lex_type, conf_invs_selected_hyps_by_lex_type[0])
    upper_yerr_selected_hyps_for_plot = np.subtract(conf_invs_selected_hyps_by_lex_type[1], mean_selected_hyps_by_lex_type)
    yerr_selected_hyps_for_plot = np.array([lower_yerr_selected_hyps_for_plot, upper_yerr_selected_hyps_for_plot])
    hypothesis_count_proportions = np.divide(mean_selected_hyps_by_lex_type, np.sum(mean_selected_hyps_by_lex_type))
    yerr_scaled_selected_hyps_for_plot = np.divide(yerr_selected_hyps_for_plot, np.sum(mean_selected_hyps_by_lex_type))
    return hypothesis_count_proportions, yerr_scaled_selected_hyps_for_plot




def evolvability_iteration(run_number, n_meanings, n_signals, n_iterations_mixed_pop, report_every_i, turnover_type, selection_type, selection_weighting, communication_type_initial_pop, ca_measure_type_initial_pop, communication_type_mutants, ca_measure_type_mutants, n_interactions, n_contexts, n_utterances, context_generation, context_type, context_size, helpful_contexts, pop_size_mixed_pop, n_mutants, teacher_type, perspectives, perspective_probs, sal_alpha, final_pop_lex_indices, final_pop_lexicons, error, extra_error, pragmatic_level_initial_pop, optimality_alpha_initial_pop, pragmatic_level_mutants, optimality_alpha_mutants, pragmatic_level_parent_hyp, learning_types, learning_type_probs, hypothesis_space, perspective_hyps, lexicon_hyps, learner_perspective, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant):

    t0 = time.clock()


    selected_hyps_per_generation_matrix = np.zeros((n_iterations_mixed_pop, pop_size_mixed_pop))
    avg_fitness_matrix = np.zeros(n_iterations_mixed_pop)
    parent_probs_matrix = np.zeros((n_iterations_mixed_pop, pop_size_mixed_pop))
    selected_parent_indices_matrix = np.zeros((n_iterations_mixed_pop, pop_size_mixed_pop))
    parent_lex_indices_matrix = np.zeros((n_iterations_mixed_pop, pop_size_mixed_pop))
    pragmatic_level_per_agent_matrix = [['' for a in range(pop_size_mixed_pop)] for i in range(n_iterations_mixed_pop)]


    print ''
    print "This is run number:"
    print str(run_number)
    print 'consisting of:'
    print str(n_iterations_mixed_pop)+' iterations'
    print ''
    print 'with population size:'
    print pop_size_mixed_pop
    print ''
    print 'and turnover type:'
    print turnover_type
    print 'the selection type is:'
    print selection_type
    print 'and the selection strength parameter is:'
    print selection_weighting
    if selection_type == 'ca_with_parent':
        print "the communication_type of the initial population is:"
        print communication_type_initial_pop
        print "and the ca_measure_type of the initial population is:"
        print ca_measure_type_initial_pop
        print "with a total of "+str(n_interactions)+" interactions"
        print ''
        print "the communication_type of the mutants is:"
        print communication_type_mutants
        print "and the ca_measure_type of the mutants is:"
        print ca_measure_type_mutants
    print 'communicating with:'
    print str(n_meanings)+' meanings'
    print 'and'
    print str(n_signals) + ' signals'
    print 'observing:'
    print str(n_contexts)+' contexts per learner'
    print 'with context type:'
    print context_generation
    if context_generation == 'only_helpful' or context_generation == 'optimal':
        print 'containing the '+str(len(helpful_contexts))+' most informative contexts'
    print ''
    print 'The teacher type is:'
    print teacher_type
    print ''
    print 'The possible perspectives are:'
    print perspectives
    print 'The perspective probabilities are:'
    print perspective_probs
    print ''
    print 'The pragmatic_level of the initial population is:'
    print pragmatic_level_initial_pop
    if pragmatic_level_initial_pop == 'prag':
        print 'The optimality_alpha of the initial population is:'
        print optimality_alpha_initial_pop
    print ''
    print 'The pragmatic_level of the mutants is:'
    print pragmatic_level_mutants
    if pragmatic_level_mutants == 'prag':
        print 'The optimality_alpha of the mutants is:'
        print optimality_alpha_mutants
    print ''
    print "pragmatic_level_parent_hyp is:"
    print pragmatic_level_parent_hyp
    print ''
    print 'The number of mutants to be inserted in generation 0 is:'
    print n_mutants
    print ''
    print 'The possible learning types are:'
    print learning_types
    print 'The learning type probabilities are:'
    print learning_type_probs
    print ''
    print 'The perspective_prior is:'
    print perspective_prior_type
    print 'With strength:'
    print perspective_prior_strength
    print 'The lexicon prior is:'
    print lexicon_prior_type
    print 'With strength:'
    print lexicon_prior_constant
    print ''
    print 'P(error) on production is:'
    print error
    print ''


    # 1) First the initial population is created:


    population = pop.MixedPopulation(pop_size_mixed_pop, n_meanings, n_signals, hypothesis_space, perspective_hyps, lexicon_hyps, learner_perspective, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant, perspectives, perspective_probs, sal_alpha, final_pop_lex_indices, final_pop_lexicons, error, extra_error, pragmatic_level_initial_pop, optimality_alpha_initial_pop, pragmatic_level_mutants, optimality_alpha_mutants, pragmatic_level_parent_hyp, n_contexts, context_type, context_generation, context_size, helpful_contexts, n_utterances, learning_types, learning_type_probs)


    for i in range(n_iterations_mixed_pop):
        if i == 0 or i % report_every_i == 0:
            print 'i = ' + str(i)


        if i == 0:
            selected_hyp_per_agent_matrix, avg_fitness, parent_probs, selected_parent_indices, parent_lex_indices, pragmatic_level_per_agent = population.insert_mutant(context_generation, helpful_contexts, n_signals, error, turnover_type, selection_type, selection_weighting, communication_type_initial_pop, ca_measure_type_initial_pop, communication_type_mutants, ca_measure_type_mutants, n_interactions, teacher_type, n_mutants)

        else:
            selected_hyp_per_agent_matrix, avg_fitness, parent_probs, selected_parent_indices, parent_lex_indices, pragmatic_level_per_agent = population.pop_update(context_generation, helpful_contexts, n_meanings, n_signals, error, turnover_type, selection_type, selection_weighting, communication_type_initial_pop, ca_measure_type_initial_pop, communication_type_mutants, ca_measure_type_mutants, n_interactions, teacher_type, perspectives_per_agent=None, decoupled=decoupling)


        print ''
        print ''
        print "selected_hyp_per_agent_matrix are:"
        print selected_hyp_per_agent_matrix

        print ''
        print ''
        print "avg_fitness is:"
        print avg_fitness


        print ''
        print ''
        print "parent_probs are:"
        print parent_probs


        print ''
        print ''
        print "selected_parent_indices are:"
        print selected_parent_indices


        print ''
        print ''
        print "np.bincount(selected_parent_indices, minlength=pop_size_mixed_pop) is:"
        print np.bincount(selected_parent_indices, minlength=pop_size_mixed_pop)

        print ''
        print ''
        print "parent_lex_indices are:"
        print parent_lex_indices

        print ''
        print ''
        print "pragmatic_level_per_agent is:"
        print pragmatic_level_per_agent
        print "len(pragmatic_level_per_agent) is:"
        print len(pragmatic_level_per_agent)

        selected_hyps_per_generation_matrix[i] = selected_hyp_per_agent_matrix
        avg_fitness_matrix[i] = avg_fitness
        parent_probs_matrix[i] = parent_probs
        selected_parent_indices_matrix[i] = selected_parent_indices
        parent_lex_indices_matrix[i] = parent_lex_indices
        pragmatic_level_per_agent_matrix[i] = pragmatic_level_per_agent

    run_time_mins = (time.clock()-t0)/60.

    results_dict = {'selected_hyps_per_generation_matrix':selected_hyps_per_generation_matrix,
                    'avg_fitness_matrix':avg_fitness_matrix,
                    'parent_probs_matrix':parent_probs_matrix,
                    'selected_parent_indices_matrix':selected_parent_indices_matrix,
                    'parent_lex_indices_matrix':parent_lex_indices_matrix,
                    'pragmatic_level_per_agent_matrix':pragmatic_level_per_agent_matrix,
                    'run_time_mins':run_time_mins}

    return results_dict





#######################################################################################################################


if __name__ == "__main__":

    ### CATEGORISING LEXICONS BY INFORMATIVENESS BELOW:



    print ''
    print ''
    # print "lexicon_hyps are:"
    # print lexicon_hyps
    print "lexicon_hyps.shape are:"
    print lexicon_hyps.shape



    informativity_per_lexicon = lex.calc_ca_all_lexicons(lexicon_hyps, error, lex_measure)
    print ''
    print ''
    # print "informativity_per_lexicon is:"
    # print informativity_per_lexicon
    print "informativity_per_lexicon.shape is:"
    print informativity_per_lexicon.shape


    argsort_informativity_per_lexicon = np.argsort(informativity_per_lexicon)
    print ''
    print ''
    # print "argsort_informativity_per_lexicon is:"
    # print argsort_informativity_per_lexicon
    print "argsort_informativity_per_lexicon.shape is:"
    print argsort_informativity_per_lexicon.shape


    ### RETRIEVING THE LEXCIONS OF THE INITIAL GENERATION:



    final_pop_lex_indices = get_final_pop_lex_indices(iteration_results_directory, '_c'+str(run_number+1), n_runs_initial_pop, n_iterations_initial_pop, n_contexts_original, pop_size_initial_pop, pragmatic_level_initial_pop, optimality_alpha_initial_pop, communication_type_initial_pop, ca_measure_type_initial_pop)


    print ''
    # print "final_pop_lex_indices are:"
    # print final_pop_lex_indices
    print "final_pop_lex_indices.shape are:"
    print final_pop_lex_indices.shape

    final_pop_lexicons = np.array([lexicon_hyps[l] for l in final_pop_lex_indices.astype(int)])
    print ''
    # print "final_pop_lexicons are:"
    # print final_pop_lexicons
    print "final_pop_lexicons.shape are:"
    print final_pop_lexicons.shape

    print ''
    print ''
    print "lexicon_hyps.shape is:"
    print lexicon_hyps.shape


    t0 = time.clock()


    print ''
    print ''
    print 'run_number is:'
    print run_number
    print ''

    all_results_dict = evolvability_iteration(run_number, n_meanings, n_signals, n_iterations_mixed_pop, report_every_i, turnover_type, selection_type, selection_weighting, communication_type_initial_pop, ca_measure_type_initial_pop, communication_type_mutants, ca_measure_type_mutants, n_interactions, n_contexts, n_utterances, context_generation, context_type, context_size, helpful_contexts, pop_size_mixed_pop, n_mutants, teacher_type, perspectives, perspective_probs, sal_alpha, final_pop_lex_indices, final_pop_lexicons, error, extra_error, pragmatic_level_initial_pop, optimality_alpha_initial_pop, pragmatic_level_mutants, optimality_alpha_mutants, pragmatic_level_parent_hyp, learning_types, learning_type_probs, hypothesis_space, perspective_hyps, lexicon_hyps, learner_perspective, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant)


    run_simulation_time = time.clock()-t0
    print
    print 'run_simulation_time is:'
    print str((run_simulation_time/60))+" m"


    t1 = time.clock()


    selected_hyps_per_generation_matrix = all_results_dict['selected_hyps_per_generation_matrix']
    # print "selected_hyps_per_generation_matrix is:"
    # print selected_hyps_per_generation_matrix
    print "selected_hyps_per_generation_matrix.shape is:"
    print selected_hyps_per_generation_matrix.shape

    avg_fitness_matrix = all_results_dict['avg_fitness_matrix']
    # print "avg_fitness_matrix is:"
    # print avg_fitness_matrix
    print "avg_fitness_matrix.shape is:"
    print avg_fitness_matrix.shape

    parent_probs_matrix = all_results_dict['parent_probs_matrix']
    # print "parent_probs_matrix is:"
    # print parent_probs_matrix
    print "parent_probs_matrix.shape is:"
    print parent_probs_matrix.shape

    selected_parent_indices_matrix = all_results_dict['selected_parent_indices_matrix']
    # print "selected_parent_indices_matrix is:"
    # print selected_parent_indices_matrix
    print "selected_parent_indices_matrix.shape is:"
    print selected_parent_indices_matrix.shape

    parent_lex_indices_matrix = all_results_dict['parent_lex_indices_matrix']
    # print "parent_lex_indices_matrix is:"
    # print parent_lex_indices_matrix
    print "parent_lex_indices_matrix.shape is:"
    print parent_lex_indices_matrix.shape

    pragmatic_level_per_agent_matrix = all_results_dict['pragmatic_level_per_agent_matrix']
    # print "pragmatic_level_per_agent_matrix is:"
    # print pragmatic_level_per_agent_matrix
    print "len(pragmatic_level_per_agent_matrix) is:"
    print len(pragmatic_level_per_agent_matrix)

    proportion_max_offspring_single_parent_matrix = np.zeros(n_iterations_mixed_pop)
    for i in range(n_iterations_mixed_pop):
        parent_indices = selected_parent_indices_matrix[i]
        parent_index_counts = np.bincount(parent_indices.astype(int))
        max_offspring_single_parent = np.amax(parent_index_counts)
        proportion_max_offspring_single_parent = np.divide(max_offspring_single_parent.astype(float), float(pop_size_mixed_pop))
        proportion_max_offspring_single_parent_matrix[i] = proportion_max_offspring_single_parent
    print ''
    print ''
    # print "proportion_max_offspring_single_parent_matrix is:"
    # print proportion_max_offspring_single_parent_matrix
    print "proportion_max_offspring_single_parent_matrix.shape is:"
    print proportion_max_offspring_single_parent_matrix.shape



    calc_performance_measures_time = time.clock()-t1
    print
    print 'calc_performance_measures_time is:'
    print str((calc_performance_measures_time/60))+" m"




    #############################################################################
    # Below the actual writing of the results to text and pickle files happens:


    t2 = time.clock()


    if context_generation == 'random':
        if selection_type == 'none' or selection_type == 'l_learning':
            filename = 'evo_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size_mixed_pop)+'_select_'+selection_type+'_'+str(n_iterations_mixed_pop)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_xtra_err_'+str(extra_error)[0:3]+'_init_pop_'+pragmatic_level_initial_pop+'_a_'+str(optimality_alpha_initial_pop)[0]+'_mutnts_'+pragmatic_level_mutants+'_a_'+str(optimality_alpha_mutants)[0]+'_s_hyp_'+pragmatic_level_parent_hyp+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_dcpl_'+str(decoupling)[0:3]+'_r'+str(run_number)
        elif selection_type == 'p_taking':
            filename = 'evo_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size_mixed_pop)+'_select_'+selection_type+'_'+str(n_iterations_mixed_pop)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_xtra_err_'+str(extra_error)[0:3]+'_init_pop_'+pragmatic_level_initial_pop+'_a_'+str(optimality_alpha_initial_pop)[0]+'_mutnts_'+pragmatic_level_mutants+'_a_'+str(optimality_alpha_mutants)[0]+'_s_hyp_'+pragmatic_level_parent_hyp+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_dcpl_'+str(decoupling)[0:3]+'_r'+str(run_number)
        elif selection_type == 'ca_with_parent':
            filename = 'evo_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size_mixed_pop)+'_select_'+selection_type+'_init_'+communication_type_initial_pop+'_'+ca_measure_type_initial_pop+'_mutnts_'+communication_type_mutants+'_'+ca_measure_type_mutants+'_'+str(n_iterations_mixed_pop)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_xtra_err_'+str(extra_error)[0:3]+'_init_pop_'+pragmatic_level_initial_pop+'_a_'+str(optimality_alpha_initial_pop)[0]+'_mutnts_'+pragmatic_level_mutants+'_a_'+str(optimality_alpha_mutants)[0]+'_s_hyp_'+pragmatic_level_parent_hyp+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_dcpl_'+str(decoupling)[0:3]+'_r'+str(run_number)


    elif context_generation == 'only_helpful' or context_generation == 'optimal':
        if selection_type == 'none' or selection_type == 'l_learning':
            filename = 'evo_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size_mixed_pop)+'_select_'+selection_type+'_'+str(n_iterations_mixed_pop)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_xtra_err_'+str(extra_error)[0:3]+'_init_pop_'+pragmatic_level_initial_pop+'_a_'+str(optimality_alpha_initial_pop)[0]+'_mutnts_'+pragmatic_level_mutants+'_a_'+str(optimality_alpha_mutants)[0]+'_s_hyp_'+pragmatic_level_parent_hyp+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_dcpl_'+str(decoupling)[0:3]+'_r'+str(run_number)
        elif selection_type == 'p_taking':
            filename = 'evo_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size_mixed_pop)+'_select_'+selection_type+'_'+str(n_iterations_mixed_pop)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_xtra_err_'+str(extra_error)[0:3]+'_init_pop_'+pragmatic_level_initial_pop+'_a_'+str(optimality_alpha_initial_pop)[0]+'_mutnts_'+pragmatic_level_mutants+'_a_'+str(optimality_alpha_mutants)[0]+'_s_hyp_'+pragmatic_level_parent_hyp+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_dcpl_'+str(decoupling)[0:3]+'_r'+str(run_number)
        elif selection_type == 'ca_with_parent':
            filename = 'evo_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size_mixed_pop)+'_select_'+selection_type+'_init_'+communication_type_initial_pop+'_'+ca_measure_type_initial_pop+'_mutnts_'+communication_type_mutants+'_'+ca_measure_type_mutants+'_'+str(n_iterations_mixed_pop)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_xtra_err_'+str(extra_error)[0:3]+'_init_pop_'+pragmatic_level_initial_pop+'_a_'+str(optimality_alpha_initial_pop)[0]+'_mutnts_'+pragmatic_level_mutants+'_a_'+str(optimality_alpha_mutants)[0]+'_s_hyp_'+pragmatic_level_parent_hyp+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_dcpl_'+str(decoupling)[0:3]+'_r'+str(run_number)



    pickle_file_title_all_results = output_pickle_file_directory + 'Results_' + filename

    pickle.dump(all_results_dict, open(pickle_file_title_all_results+'.p', 'wb'))



    pickle_file_title_max_offspring_single_parent = output_pickle_file_directory + 'Max_Offspr_' + filename

    pickle.dump(proportion_max_offspring_single_parent, open(pickle_file_title_max_offspring_single_parent+'.p', 'wb'))


    write_to_files_time = time.clock()-t2
    print
    print 'write_to_files_time is:'
    print str((write_to_files_time/60))+" m"




