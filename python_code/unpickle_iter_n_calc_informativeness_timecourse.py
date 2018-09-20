__author__ = 'Marieke Woensdregt'


import numpy as np
import pickle
from scipy import stats

import hypspace
import lex
import measur
import saveresults



# np.set_printoptions(threshold=np.nan)



#######################################################################################################################
# 1: THE PARAMETERS:


##!!!!!! MAKE SURE TO CHANGE THE PATHS BELOW TO MATCH THE FILE SYSTEM OF YOUR MACHINE:
directory = '/exports/eddie/scratch/s1370641/'

directory_laptop = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Eddie_Output/'

results_directory = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/Iteration/'



# 1.1: The parameters defining the lexicon size (and thus the number of meanings in the world):

n_meanings = 3  # The number of meanings
n_signals = 3  # The number of signals




# 1.2: The parameters defining the contexts and how they map to the agent's saliencies:

context_generation = 'optimal' # This can be set to either 'random', 'only_helpful', 'optimal'
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



error = 0.05 # The error term on production
error_string = saveresults.convert_array_to_string(error)
extra_error = True # Determines whether the error specified above gets added on top AFTER the pragmatic speaker has calculated its production probabilities by maximising the utility to the listener.





# 1.3: The parameters that determine the make-up of the population:

pop_size = 100

agent_type = 'no_p_distinction' # This can be set to either 'p_distinction' or 'no_p_distinction'. Determines whether the population is made up of DistinctionAgent objects or Agent objects (DistinctionAgents can learn different perspectives for different agents, Agents can only learn one perspective for all agents they learn from).




pragmatic_level = 'literal'  # This can be set to either 'literal', 'perspective-taking' or 'prag'
optimality_alpha = 1.0  # Goodman & Stuhlmuller (2013) fitted sal_alpha = 3.4 to participant data with 4x3 lexicon of three number words ('one', 'two', and 'three') that could be mapped to four different world states (0, 1, 2, and 3)
optimality_alpha_string = saveresults.convert_float_value_to_string(optimality_alpha)


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


if agent_type == 'no_p_distinction':
    hypothesis_space = hypspace.list_hypothesis_space(perspective_hyps, lexicon_hyps) # The full space of composite hypotheses that the learner will consider (2D numpy matrix with composite hypotheses on the rows, perspective hypotheses on column 0 and lexicon hypotheses on column 1)

elif agent_type == 'p_distinction':
    hypothesis_space = hypspace.list_hypothesis_space_with_speaker_distinction(perspective_hyps, lexicon_hyps, pop_size) # The full space of composite hypotheses that the learner will consider (2D numpy matrix with composite hypotheses on the rows, perspective hypotheses on column 0 and lexicon hypotheses on column 1)



# 1.5: More parameters that determine the make-up of the population:

lexicon_probs = np.array([0. for x in range(len(lexicon_hyps)-1)]+[1.])


perspectives = np.array([0., 1.]) # The different perspectives that agents can have
perspective_probs = np.array([0., 1.]) # The ratios with which the different perspectives will be present in the population
perspective_probs_string = saveresults.convert_array_to_string(perspective_probs) # Turns the perspective probs into a string in order to add it to file names


learning_types = ['map', 'sample'] # The types of learning that the learners can do
learning_type_probs = np.array([0., 1.]) # The ratios with which the different learning types will be present in the population
learning_type_probs_string = saveresults.convert_array_to_string(learning_type_probs) # Turns the learning type probs into a string in order to add it to file names
if learning_type_probs[0] == 1.:
    learning_type_string = learning_types[0]
elif learning_type_probs[1] == 1.:
    learning_type_string = learning_types[1]


# 1.6: The parameters that determine the learner's prior:

learner_type = 'both_unknown'  # This can be set to either 'perspective_unknown', 'lexicon_unknown' or 'both_unknown'


learner_perspective = 0.  # The learner's perspective


perspective_prior_type = 'egocentric'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
perspective_prior_strength = 0.9  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
perspective_prior_strength_string = saveresults.convert_array_to_string(perspective_prior_strength)



lexicon_prior_type = 'neutral'  # This can be set to either 'neutral', 'ambiguous_fixed', 'half_ambiguous_fixed', 'expressivity_bias' or 'compressibility_bias'
lexicon_prior_constant = 0.0  # Determines the strength of the lexicon bias, with small c creating a STRONG prior and large c creating a WEAK prior. (And with c = 1000 creating an almost uniform prior.)
lexicon_prior_constant_string = saveresults.convert_array_to_string(lexicon_prior_constant)



# 1.7: The parameters that determine the amount of data_dict that the learner gets to see, and the amount of runs of the simulation:

n_utterances = 1  # This parameter determines how many signals the learner gets to observe in each context
n_contexts = 120

# The number of contexts that the learner gets to see.


# 1.8: The parameters that determine the type and number of simulations that are run:

#FIXME: In the current implementation 'half_ambiguous_lex' can have different instantiations. Therefore, if we run a simulation where there are different lexicon_type_probs, we will want the run_type to be 'population_same_pop' to make sure that all the 'half ambiguous' speakers do have the SAME half ambiguous lexicon.
run_type = 'iter'  # This parameter determines whether the learner communicates with only one speaker ('dyadic') or with a population ('population_diff_pop' if there are no speakers with the 'half_ambiguous' lexicon type, 'population_same_pop' if there are, 'population_same_pop_dist_learner' if the learner can distinguish between different speakers), or whether we do an iterated learning model ('iter')
turnover_type = 'whole_pop'  # The type of turnover with which the population is replaced (for the iteration simulation). This can be set to either 'chain' (one agent at a time) or 'whole_pop' (entire population at once)

communication_type = 'lex_n_p'  # This can be set to either 'lex_only', 'lex_n_context', 'lex_n_p' or 'prag'
ca_measure_type = 'comp_only'  # This can be set to either "comp_n_prod" or "comp_only"
n_interactions = 6  # The number of interactions used to calculate communicative accuracy

selection_type = 'none'  # This can be set to either 'none', 'p_taking', 'l_learning' or 'ca_with_parent'
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

n_copies = 200  # Specifies the number of copies of the results file
copy_specification = ''  # Can be set to e.g. '_c1' or simply to '' if there is only one copy


lex_measure = 'ca'  # This can be set to either 'mi' for mutual information or 'ca' for communicative accuracy (of the lexicon with itself)


window_convergence_check = 50

bandwith_convergence_check = 0.1


pilot = True

#######################################################################################################################


def unpickle_selected_hyps_matrix(directory, filename, n_copies):
    if n_copies == 1:
        multi_run_selected_hyps_per_generation_matrix_all_copies = np.zeros((1, n_runs, n_iterations, pop_size))
        pickle_filename_all_results = 'Results_'+filename+copy_specification
        results_dict = pickle.load(open(directory+pickle_filename_all_results+'.p', 'rb'))
        multi_run_selected_hyps_per_generation_matrix = results_dict[
            'multi_run_selected_hyps_per_generation_matrix']
        multi_run_selected_hyps_per_generation_matrix_all_copies[0] = multi_run_selected_hyps_per_generation_matrix
    elif n_copies > 1:
        multi_run_selected_hyps_per_generation_matrix_all_copies = np.zeros(((n_copies*n_runs), n_iterations, pop_size))
        counter = 0
        for c in range(1, n_copies+1):
            pickle_filename_all_results = 'Results_'+filename+'_c'+str(c)
            results_dict = pickle.load(open(directory+pickle_filename_all_results+'.p', 'rb'))
            for r in range(n_runs):
                multi_run_selected_hyps_per_generation_matrix = results_dict['multi_run_selected_hyps_per_generation_matrix'][r]
                multi_run_selected_hyps_per_generation_matrix_all_copies[counter] = multi_run_selected_hyps_per_generation_matrix
                counter += 1
        multi_run_selected_hyps_per_generation_matrix = multi_run_selected_hyps_per_generation_matrix_all_copies
    return multi_run_selected_hyps_per_generation_matrix


def get_selected_hyps_ordered(multi_run_selected_hyps_per_generation_matrix, hyp_order):
    selected_hyps_new_lex_order_all_runs = np.zeros_like(multi_run_selected_hyps_per_generation_matrix)
    for r in range(n_runs*n_copies):
        for i in range(n_iterations):
            for a in range(pop_size):
                this_agent_hyp = multi_run_selected_hyps_per_generation_matrix[r][i][a]
                if this_agent_hyp >= len(hyp_order):
                    this_agent_hyp = this_agent_hyp-len(hyp_order)
                new_order_index = np.argwhere(hyp_order == this_agent_hyp)
                selected_hyps_new_lex_order_all_runs[r][i][a] = new_order_index
    return selected_hyps_new_lex_order_all_runs


print ''
print ''
print 'n_contexts:'
print n_contexts


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



informativity_per_lexicon_sorted = np.round(informativity_per_lexicon[argsort_informativity_per_lexicon], decimals=2)
print ''
print ''
# print "informativity_per_lexicon_sorted is:"
# print informativity_per_lexicon_sorted
print "informativity_per_lexicon_sorted.shape is:"
print informativity_per_lexicon_sorted.shape

minimum_informativity = np.amin(informativity_per_lexicon_sorted)
print ''
print ''
print "minimum_informativity is:"
print minimum_informativity

min_info_indices = np.argwhere(informativity_per_lexicon_sorted==minimum_informativity)
print ''
print ''
min_info_indices = min_info_indices.flatten()
print "min_info_indices are:"
print min_info_indices

maximum_informativity = np.amax(informativity_per_lexicon_sorted)
print ''
print ''
print "maximum_informativity is:"
print maximum_informativity

max_info_indices = np.argwhere(informativity_per_lexicon_sorted==maximum_informativity)
print ''
print ''
max_info_indices = max_info_indices.flatten()
print "max_info_indices are:"
print max_info_indices

intermediate_info_indices = np.arange(min_info_indices[-1]+1, max_info_indices[0])
print ''
print ''
# print "intermediate_info_indices are:"
# print intermediate_info_indices

lexicon_hyps_sorted = lexicon_hyps[argsort_informativity_per_lexicon]
print ''
print ''
# print "lexicon_hyps_sorted is:"
# print lexicon_hyps_sorted
print "lexicon_hyps_sorted.shape is:"
print lexicon_hyps_sorted.shape



def get_inf_values_last_generation(run_number, copy_number, n_iterations, inf_per_lex, selected_hyps_new_lex_order_all_runs):
    final_generation_hyps = selected_hyps_new_lex_order_all_runs[run_number*copy_number][-1]
    inf_per_agent_final_generation = inf_per_lex[final_generation_hyps.astype(int)]
    return inf_per_agent_final_generation


def avg_informativity_over_gens(n_runs, n_copies, n_iterations, inf_per_lex, selected_hyps_new_lex_order_all_runs):
    avg_inf_over_gens_matrix = np.zeros(((n_runs*n_copies), n_iterations))
    for r in range((n_runs*n_copies)):
        for i in range(n_iterations):
            generation_hyps = selected_hyps_new_lex_order_all_runs[r][i]
            inf_per_agent = inf_per_lex[generation_hyps.astype(int)]
            avg_inf = np.mean(inf_per_agent)
            avg_inf_over_gens_matrix[r][i] = avg_inf
    return avg_inf_over_gens_matrix


def calc_mean_and_conf_invs_inf_over_gens(avg_inf_over_gens_matrix):
    mean_inf_over_gens = np.mean(avg_inf_over_gens_matrix, axis=0)
    std_inf_over_gens = np.std(avg_inf_over_gens_matrix, axis=0)
    conf_intervals_inf_over_gens = stats.norm.interval(0.95, loc=mean_inf_over_gens, scale=std_inf_over_gens / np.sqrt(n_runs*n_copies))
    conf_intervals_inf_over_gens = np.array(conf_intervals_inf_over_gens)
    return mean_inf_over_gens, conf_intervals_inf_over_gens


def calc_percentiles_inf_over_gens(avg_inf_over_gens_matrix):
    percentile_25_inf_over_gens = np.percentile(avg_inf_over_gens_matrix, 25, axis=0)
    median_optimal_inf_over_gens = np.percentile(avg_inf_over_gens_matrix, 50, axis=0)
    percentile_75_inf_over_gens = np.percentile(avg_inf_over_gens_matrix, 75, axis=0)
    percentiles_inf_over_gens = np.array([percentile_25_inf_over_gens, median_optimal_inf_over_gens, percentile_75_inf_over_gens])
    return percentiles_inf_over_gens


#####################################################################################

if __name__ == "__main__":

    directory = '/exports/eddie/scratch/s1370641/'

    directory_laptop = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Eddie_Output/'

    inf_per_lex_full_hyp_space = informativity_per_lexicon_sorted
    for p in range(len(perspective_hyps)-1):
        inf_per_lex_full_hyp_space = np.hstack((inf_per_lex_full_hyp_space, informativity_per_lexicon_sorted))
    print "inf_per_lex_full_hyp_space is:"
    print inf_per_lex_full_hyp_space
    print "inf_per_lex_full_hyp_space.shape is:"
    print inf_per_lex_full_hyp_space.shape


    unique_inf_values = np.unique(inf_per_lex_full_hyp_space)
    print ''
    print "unique_inf_values are:"
    print unique_inf_values

    max_inf = np.amax(inf_per_lex_full_hyp_space)
    print ''
    print 'max_inf is:'
    print max_inf

    baseline_inf = np.mean(inf_per_lex_full_hyp_space)
    print ''
    print 'baseline_inf is:'
    print baseline_inf



    results_directory = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/Iteration/'

#
#
# #####################################################################################
# ## PRIOR 1: EXOCENTRIC
# ### CONDITION 1: No Selection:
#
#     print ''
#     print ''
#     print 'This is prior 1: Exocentric'
#     print ''
#     print 'This is condition 1: No selection'
#
#     perspective_prior_type = 'egocentric'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
#     perspective_prior_strength = 0.0  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
#     perspective_prior_strength_string = convert_array_to_string(perspective_prior_strength)
#     selection_type = 'none'
#
#     folder = 'results_iter_select_'+selection_type+'_p_prior_exo_10_l_prior_'+lexicon_prior_type[0:4]+'_'+pragmatic_level+'_alpha_'+str(optimality_alpha)[0]
#     print ''
#     print "folder is:"
#     print folder
#
#     output_pickle_file_directory = directory_laptop+folder+'/'
#     print ''
#     print "output_pickle_file_directory is:"
#     print output_pickle_file_directory
#
#
#     if context_generation == 'random':
#         filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_turn_'+str(turnover_type)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_alpha_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
#
#     elif context_generation == 'only_helpful' or context_generation == 'optimal':
#         filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_turn_'+str(turnover_type)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_alpha_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
#     print ''
#     print "filename is:"
#     print filename
#
#
#     selected_hyps_new_lex_order_all_runs = get_selected_hyps_ordered(output_pickle_file_directory, filename, n_copies, new_hyp_order_handsorted_on_lexicons)
#     # print "selected_hyps_new_lex_order_all_runs is:"
#     # print selected_hyps_new_lex_order_all_runs
#     print "selected_hyps_new_lex_order_all_runs.shape is:"
#     print selected_hyps_new_lex_order_all_runs.shape
#
#     avg_inf_over_gens_matrix = avg_informativity_over_gens(n_runs, n_copies, n_iterations, inf_per_lex_full_hyp_space, selected_hyps_new_lex_order_all_runs)
#     print "avg_inf_over_gens_matrix is:"
#     print avg_inf_over_gens_matrix
#     print "avg_inf_over_gens_matrix.shape is:"
#     print avg_inf_over_gens_matrix.shape
#
#
#     mean_inf_over_gens, conf_intervals_inf_over_gens = calc_mean_and_conf_invs_inf_over_gens(avg_inf_over_gens_matrix)
#     print ''
#     print ''
#     print "mean_inf_over_gens is:"
#     print mean_inf_over_gens
#     print "mean_inf_over_gens.shape is:"
#     print mean_inf_over_gens.shape
#     print ''
#     print "conf_intervals_inf_over_gens is:"
#     print conf_intervals_inf_over_gens
#     print "conf_intervals_inf_over_gens.shape is:"
#     print conf_intervals_inf_over_gens.shape
#
#
#     percentiles_inf_over_gens = calc_percentiles_inf_over_gens(avg_inf_over_gens_matrix)
#     print ''
#     print ''
#     print "percentiles_inf_over_gens is:"
#     print percentiles_inf_over_gens
#     print "percentiles_inf_over_gens.shape is:"
#     print percentiles_inf_over_gens.shape
#
#
# inf_over_gens_data_dict = {'raw_data': avg_inf_over_gens_matrix,
#                             'inf_values_last_generation': inf_values_last_generation,
#                            'mean_inf_over_gens': mean_inf_over_gens,
#                            'conf_intervals_inf_over_gens': conf_intervals_inf_over_gens,
#                            'percentiles_inf_over_gens': percentiles_inf_over_gens,
#                            'baseline_inf': baseline_inf,
#                            'max_inf': max_inf}
#
#
#     pickle.dump(inf_over_gens_data_dict, open(output_pickle_file_directory+'lex_inf_data_'+lex_measure+'_'+filename+'.p', "wb"))
#
#
#
#
#
#
#
# #####################################################################################
# ## PRIOR 1: EXOCENTRIC
# ### CONDITION 2: Selection on CS:
#
#     print ''
#     print ''
#     print 'This is prior 1: Exocentric'
#     print ''
#     print 'This is condition 2: Selection on CS'
#
#     perspective_prior_type = 'egocentric'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
#     perspective_prior_strength = 0.0  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
#     perspective_prior_strength_string = convert_array_to_string(perspective_prior_strength)
#     selection_type = 'ca_with_parent'
#     communication_type = 'lex_n_context'  # This can be set to either 'lex_only', 'lex_n_context' or 'lex_n_p'
#     ca_measure_type = 'comp_n_prod'  # This can be set to either "comp_n_prod" or "comp_only"
#     n_interactions = 12  # The number of interactions used to calculate communicative accuracy
#
#
#     folder = 'results_iter_select_' + selection_type+'_'+communication_type+'_'+ca_measure_type+ '_p_prior_exo_10_l_prior_' + lexicon_prior_type[0:4]+'_'+pragmatic_level+'_alpha_'+str(optimality_alpha)[0]
#     print ''
#     print "folder is:"
#     print folder
#
#     output_pickle_file_directory = directory_laptop+folder+'/'
#     print ''
#     print "output_pickle_file_directory is:"
#     print output_pickle_file_directory
#
#     if context_generation == 'random':
#         filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_turn_'+str(turnover_type)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_alpha_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
#
#     elif context_generation == 'only_helpful' or context_generation == 'optimal':
#         filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_turn_'+str(turnover_type)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_alpha_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
#     print ''
#     print "filename is:"
#     print filename
#
#
#     selected_hyps_new_lex_order_all_runs = get_selected_hyps_ordered(output_pickle_file_directory, filename, n_copies, new_hyp_order_handsorted_on_lexicons)
#     # print "selected_hyps_new_lex_order_all_runs is:"
#     # print selected_hyps_new_lex_order_all_runs
#     print "selected_hyps_new_lex_order_all_runs.shape is:"
#     print selected_hyps_new_lex_order_all_runs.shape
#
#     avg_inf_over_gens_matrix = avg_informativity_over_gens(n_runs, n_copies, n_iterations, inf_per_lex_full_hyp_space, selected_hyps_new_lex_order_all_runs)
#     print "avg_inf_over_gens_matrix is:"
#     print avg_inf_over_gens_matrix
#     print "avg_inf_over_gens_matrix.shape is:"
#     print avg_inf_over_gens_matrix.shape
#
#
#     mean_inf_over_gens, conf_intervals_inf_over_gens = calc_mean_and_conf_invs_inf_over_gens(avg_inf_over_gens_matrix)
#     print ''
#     print ''
#     print "mean_inf_over_gens is:"
#     print mean_inf_over_gens
#     print "mean_inf_over_gens.shape is:"
#     print mean_inf_over_gens.shape
#     print ''
#     print "conf_intervals_inf_over_gens is:"
#     print conf_intervals_inf_over_gens
#     print "conf_intervals_inf_over_gens.shape is:"
#     print conf_intervals_inf_over_gens.shape
#
#
#     percentiles_inf_over_gens = calc_percentiles_inf_over_gens(avg_inf_over_gens_matrix)
#     print ''
#     print ''
#     print "percentiles_inf_over_gens is:"
#     print percentiles_inf_over_gens
#     print "percentiles_inf_over_gens.shape is:"
#     print percentiles_inf_over_gens.shape
#
#
# inf_over_gens_data_dict = {'raw_data': avg_inf_over_gens_matrix,
#                             'inf_values_last_generation':inf_values_last_generation,
#                            'mean_inf_over_gens': mean_inf_over_gens,
#                            'conf_intervals_inf_over_gens': conf_intervals_inf_over_gens,
#                            'percentiles_inf_over_gens': percentiles_inf_over_gens,
#                            'baseline_inf': baseline_inf,
#                            'max_inf': max_inf}
#
#     pickle.dump(inf_over_gens_data_dict, open(output_pickle_file_directory+'lex_inf_data_'+lex_measure+'_'+filename+'.p', "wb"))
#
#
#
#
# #####################################################################################
# ## PRIOR 1: EXOCENTRIC
# ### CONDITION 3: Selection on P-taking:
#
#     print ''
#     print ''
#     print 'This is prior 1: Exocentric'
#     print ''
#     print 'This is condition 3: Selection on P-taking'
#
#     perspective_prior_type = 'egocentric'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
#     perspective_prior_strength = 0.0  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
#     perspective_prior_strength_string = convert_array_to_string(perspective_prior_strength)
#     selection_type = 'p_taking'
#
#     folder = 'results_iter_select_'+selection_type+'_p_prior_exo_10_l_prior_'+lexicon_prior_type[0:4]+'_'+pragmatic_level+'_alpha_'+str(optimality_alpha)[0]
#     print ''
#     print "folder is:"
#     print folder
#
#     output_pickle_file_directory = directory_laptop+folder+'/'
#     print ''
#     print "output_pickle_file_directory is:"
#     print output_pickle_file_directory
#
#     if context_generation == 'random':
#         filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_turn_'+str(turnover_type)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_alpha_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
#
#     elif context_generation == 'only_helpful' or context_generation == 'optimal':
#         filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_turn_'+str(turnover_type)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_alpha_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
#     print ''
#     print "filename is:"
#     print filename
#
#     selected_hyps_new_lex_order_all_runs = get_selected_hyps_ordered(output_pickle_file_directory, filename, n_copies, new_hyp_order_handsorted_on_lexicons)
#     # print "selected_hyps_new_lex_order_all_runs is:"
#     # print selected_hyps_new_lex_order_all_runs
#     print "selected_hyps_new_lex_order_all_runs.shape is:"
#     print selected_hyps_new_lex_order_all_runs.shape
#
#     avg_inf_over_gens_matrix = avg_informativity_over_gens(n_runs, n_copies, n_iterations, inf_per_lex_full_hyp_space, selected_hyps_new_lex_order_all_runs)
#     print "avg_inf_over_gens_matrix is:"
#     print avg_inf_over_gens_matrix
#     print "avg_inf_over_gens_matrix.shape is:"
#     print avg_inf_over_gens_matrix.shape
#
#
#     mean_inf_over_gens, conf_intervals_inf_over_gens = calc_mean_and_conf_invs_inf_over_gens(avg_inf_over_gens_matrix)
#     print ''
#     print ''
#     print "mean_inf_over_gens is:"
#     print mean_inf_over_gens
#     print "mean_inf_over_gens.shape is:"
#     print mean_inf_over_gens.shape
#     print ''
#     print "conf_intervals_inf_over_gens is:"
#     print conf_intervals_inf_over_gens
#     print "conf_intervals_inf_over_gens.shape is:"
#     print conf_intervals_inf_over_gens.shape
#
#
#     percentiles_inf_over_gens = calc_percentiles_inf_over_gens(avg_inf_over_gens_matrix)
#     print ''
#     print ''
#     print "percentiles_inf_over_gens is:"
#     print percentiles_inf_over_gens
#     print "percentiles_inf_over_gens.shape is:"
#     print percentiles_inf_over_gens.shape
#
# inf_over_gens_data_dict = {'raw_data': avg_inf_over_gens_matrix,
#                             'inf_values_last_generation': inf_values_last_generation,
#                            'mean_inf_over_gens': mean_inf_over_gens,
#                            'conf_intervals_inf_over_gens': conf_intervals_inf_over_gens,
#                            'percentiles_inf_over_gens': percentiles_inf_over_gens,
#                            'baseline_inf': baseline_inf,
#                            'max_inf': max_inf}
#
#
#     pickle.dump(inf_over_gens_data_dict, open(output_pickle_file_directory+'lex_inf_data_'+lex_measure+'_'+filename+'.p', "wb"))
#

#
# #####################################################################################
# ## PRIOR 2: NEUTRAL
# ### CONDITION 1: No Selection:
#
#     print ''
#     print ''
#     print 'This is prior 2: Uniform'
#     print ''
#     print 'This is condition 1: No selection'
#
#     perspective_prior_type = 'neutral'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
#     perspective_prior_strength = 0.0  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
#     perspective_prior_strength_string = convert_array_to_string(perspective_prior_strength)
#     selection_type = 'none'
#
#     folder = 'results_iter_'+str(n_meanings)+'M_'+str(n_signals)+'S_select_'+selection_type+'_p_prior_neut_00_l_prior_'+lexicon_prior_type[0:4]+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]
#     print ''
#     print "folder is:"
#     print folder
#
#     output_pickle_file_directory = directory_laptop+folder+'/'
#     print ''
#     print "output_pickle_file_directory is:"
#     print output_pickle_file_directory
#
#     if context_generation == 'random':
#         filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_turn_'+str(turnover_type)+'_select_'+selection_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
#
#     elif context_generation == 'only_helpful' or context_generation == 'optimal':
#         filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_turn_'+str(turnover_type)+'_select_'+selection_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
#     print ''
#     print "filename is:"
#     print filename
#
#
#     multi_run_selected_hyps_per_generation_matrix = unpickle_selected_hyps_matrix(output_pickle_file_directory, filename, n_copies)
#     # print "multi_run_selected_hyps_per_generation_matrix is:"
#     # print multi_run_selected_hyps_per_generation_matrix
#     print "multi_run_selected_hyps_per_generation_matrix.shape is:"
#     print multi_run_selected_hyps_per_generation_matrix.shape
#
#     selected_hyps_new_lex_order_all_runs = get_selected_hyps_ordered(multi_run_selected_hyps_per_generation_matrix, argsort_informativity_per_lexicon)
#     print ''
#     print ''
#     # print "selected_hyps_new_lex_order_all_runs is:"
#     # print selected_hyps_new_lex_order_all_runs
#     print "selected_hyps_new_lex_order_all_runs.shape is:"
#     print selected_hyps_new_lex_order_all_runs.shape
#
#     avg_inf_over_gens_matrix = avg_informativity_over_gens(n_runs, n_copies, n_iterations, inf_per_lex_full_hyp_space, selected_hyps_new_lex_order_all_runs)
#     print "avg_inf_over_gens_matrix is:"
#     print avg_inf_over_gens_matrix
#     print "avg_inf_over_gens_matrix.shape is:"
#     print avg_inf_over_gens_matrix.shape
#
#     mean_inf_over_gens, conf_intervals_inf_over_gens = calc_mean_and_conf_invs_inf_over_gens(avg_inf_over_gens_matrix)
#     print ''
#     print ''
#     print "mean_inf_over_gens is:"
#     print mean_inf_over_gens
#     print "mean_inf_over_gens.shape is:"
#     print mean_inf_over_gens.shape
#     print ''
#     print "conf_intervals_inf_over_gens is:"
#     print conf_intervals_inf_over_gens
#     print "conf_intervals_inf_over_gens.shape is:"
#     print conf_intervals_inf_over_gens.shape
#
#
#     percentiles_inf_over_gens = calc_percentiles_inf_over_gens(avg_inf_over_gens_matrix)
#     print ''
#     print ''
#     print "percentiles_inf_over_gens is:"
#     print percentiles_inf_over_gens
#     print "percentiles_inf_over_gens.shape is:"
#     print percentiles_inf_over_gens.shape
#
#
#     inf_over_gens_data_dict = {'raw_data':avg_inf_over_gens_matrix,
#                                 'inf_values_last_generation': inf_values_last_generation,
#                                'mean_inf_over_gens':mean_inf_over_gens,
#                                'conf_intervals_inf_over_gens':conf_intervals_inf_over_gens,
#                                'percentiles_inf_over_gens':percentiles_inf_over_gens,
#                                'baseline_inf':baseline_inf,
#                                'max_inf':max_inf}
#
#     if context_generation == 'random':
#         filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
#     elif context_generation == 'only_helpful' or context_generation == 'optimal':
#         filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
#
#     pickle.dump(inf_over_gens_data_dict, open(output_pickle_file_directory+'lex_inf_data_'+lex_measure+'_'+filename_short+'.p', "wb"))
#
#
#
#
# #####################################################################################
# ## PRIOR 2: UNIFORM
# ### CONDITION 2: Selection on CS:
#
#     print ''
#     print ''
#     print 'This is prior 2: Uniform'
#     print ''
#     print 'This is condition 2: Selection on CS'
#
#     perspective_prior_type = 'neutral'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
#     perspective_prior_strength = 0.0  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
#     perspective_prior_strength_string = convert_array_to_string(perspective_prior_strength)
#     selection_type = 'ca_with_parent'
#
#     folder = 'results_iter_'+str(n_meanings)+'M_'+str(n_signals)+'S_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+ '_p_prior_neut_00_l_prior_' + lexicon_prior_type[0:4]+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]
#     print ''
#     print "folder is:"
#     print folder
#
#     output_pickle_file_directory = directory_laptop+folder+'/'
#     print ''
#     print "output_pickle_file_directory is:"
#     print output_pickle_file_directory
#
#     if context_generation == 'random':
#         filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_turn_'+str(turnover_type)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
#
#     elif context_generation == 'only_helpful' or context_generation == 'optimal':
#         filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_turn_'+str(turnover_type)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
#     print ''
#     print "filename is:"
#     print filename
#
#     multi_run_selected_hyps_per_generation_matrix = unpickle_selected_hyps_matrix(output_pickle_file_directory, filename, n_copies)
#     # print "multi_run_selected_hyps_per_generation_matrix is:"
#     # print multi_run_selected_hyps_per_generation_matrix
#     print "multi_run_selected_hyps_per_generation_matrix.shape is:"
#     print multi_run_selected_hyps_per_generation_matrix.shape
#
#     selected_hyps_new_lex_order_all_runs = get_selected_hyps_ordered(multi_run_selected_hyps_per_generation_matrix, argsort_informativity_per_lexicon)
#     print ''
#     print ''
#     # print "selected_hyps_new_lex_order_all_runs is:"
#     # print selected_hyps_new_lex_order_all_runs
#     print "selected_hyps_new_lex_order_all_runs.shape is:"
#     print selected_hyps_new_lex_order_all_runs.shape
#
#     avg_inf_over_gens_matrix = avg_informativity_over_gens(n_runs, n_copies, n_iterations, inf_per_lex_full_hyp_space, selected_hyps_new_lex_order_all_runs)
#     print "avg_inf_over_gens_matrix is:"
#     print avg_inf_over_gens_matrix
#     print "avg_inf_over_gens_matrix.shape is:"
#     print avg_inf_over_gens_matrix.shape
#
#
#     mean_inf_over_gens, conf_intervals_inf_over_gens = calc_mean_and_conf_invs_inf_over_gens(avg_inf_over_gens_matrix)
#     print ''
#     print ''
#     print "mean_inf_over_gens is:"
#     print mean_inf_over_gens
#     print "mean_inf_over_gens.shape is:"
#     print mean_inf_over_gens.shape
#     print ''
#     print "conf_intervals_inf_over_gens is:"
#     print conf_intervals_inf_over_gens
#     print "conf_intervals_inf_over_gens.shape is:"
#     print conf_intervals_inf_over_gens.shape
#
#
#     percentiles_inf_over_gens = calc_percentiles_inf_over_gens(avg_inf_over_gens_matrix)
#     print ''
#     print ''
#     print "percentiles_inf_over_gens is:"
#     print percentiles_inf_over_gens
#     print "percentiles_inf_over_gens.shape is:"
#     print percentiles_inf_over_gens.shape
#
#
#     inf_over_gens_data_dict = {'raw_data':avg_inf_over_gens_matrix,
#                                 'inf_values_last_generation': inf_values_last_generation,
#                                'mean_inf_over_gens':mean_inf_over_gens,
#                                'conf_intervals_inf_over_gens':conf_intervals_inf_over_gens,
#                                'percentiles_inf_over_gens':percentiles_inf_over_gens,
#                                'baseline_inf':baseline_inf,
#                                'max_inf':max_inf}
#
#
#
#     if context_generation == 'random':
#         filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
#     elif context_generation == 'only_helpful' or context_generation == 'optimal':
#         filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
#
#
#     pickle.dump(inf_over_gens_data_dict, open(output_pickle_file_directory+'lex_inf_data_'+lex_measure+'_'+filename_short+'.p', "wb"))
#
#
#
#
# #####################################################################################
# ## PRIOR 2: UNIFORM
# ### CONDITION 3: Selection on P-taking:
#
#     print ''
#     print ''
#     print 'This is prior 2: Uniform'
#     print ''
#     print 'This is condition 3: Selection on P-taking'
#
#     perspective_prior_type = 'neutral'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
#     perspective_prior_strength = 0.0  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
#     perspective_prior_strength_string = convert_array_to_string(perspective_prior_strength)
#     selection_type = 'p_taking'
#
#
#     folder = 'results_iter_'+str(n_meanings)+'M_'+str(n_signals)+'S_select_'+selection_type+'_p_prior_neut_00_l_prior_'+lexicon_prior_type[0:4]+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]
#     print ''
#     print "folder is:"
#     print folder
#
#     output_pickle_file_directory = directory_laptop+folder+'/'
#     print ''
#     print "output_pickle_file_directory is:"
#     print output_pickle_file_directory
#
#     if context_generation == 'random':
#         filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_turn_'+str(turnover_type)+'_select_'+selection_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
#
#     elif context_generation == 'only_helpful' or context_generation == 'optimal':
#         filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_turn_'+str(turnover_type)+'_select_'+selection_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
#     print ''
#     print "filename is:"
#     print filename
#
#     multi_run_selected_hyps_per_generation_matrix = unpickle_selected_hyps_matrix(output_pickle_file_directory, filename, n_copies)
#     # print "multi_run_selected_hyps_per_generation_matrix is:"
#     # print multi_run_selected_hyps_per_generation_matrix
#     print "multi_run_selected_hyps_per_generation_matrix.shape is:"
#     print multi_run_selected_hyps_per_generation_matrix.shape
#
#     selected_hyps_new_lex_order_all_runs = get_selected_hyps_ordered(multi_run_selected_hyps_per_generation_matrix, argsort_informativity_per_lexicon)
#     print ''
#     print ''
#     # print "selected_hyps_new_lex_order_all_runs is:"
#     # print selected_hyps_new_lex_order_all_runs
#     print "selected_hyps_new_lex_order_all_runs.shape is:"
#     print selected_hyps_new_lex_order_all_runs.shape
#
#     avg_inf_over_gens_matrix = avg_informativity_over_gens(n_runs, n_copies, n_iterations, inf_per_lex_full_hyp_space, selected_hyps_new_lex_order_all_runs)
#     print "avg_inf_over_gens_matrix is:"
#     print avg_inf_over_gens_matrix
#     print "avg_inf_over_gens_matrix.shape is:"
#     print avg_inf_over_gens_matrix.shape
#
#     mean_inf_over_gens, conf_intervals_inf_over_gens = calc_mean_and_conf_invs_inf_over_gens(avg_inf_over_gens_matrix)
#     print ''
#     print ''
#     print "mean_inf_over_gens is:"
#     print mean_inf_over_gens
#     print "mean_inf_over_gens.shape is:"
#     print mean_inf_over_gens.shape
#     print ''
#     print "conf_intervals_inf_over_gens is:"
#     print conf_intervals_inf_over_gens
#     print "conf_intervals_inf_over_gens.shape is:"
#     print conf_intervals_inf_over_gens.shape
#
#
#     percentiles_inf_over_gens = calc_percentiles_inf_over_gens(avg_inf_over_gens_matrix)
#     print ''
#     print ''
#     print "percentiles_inf_over_gens is:"
#     print percentiles_inf_over_gens
#     print "percentiles_inf_over_gens.shape is:"
#     print percentiles_inf_over_gens.shape
#
#
#     inf_over_gens_data_dict = {'raw_data':avg_inf_over_gens_matrix,
#                                 'inf_values_last_generation': inf_values_last_generation,
#                                'mean_inf_over_gens':mean_inf_over_gens,
#                                'conf_intervals_inf_over_gens':conf_intervals_inf_over_gens,
#                                'percentiles_inf_over_gens':percentiles_inf_over_gens,
#                                'baseline_inf':baseline_inf,
#                                'max_inf':max_inf}
#
#     if context_generation == 'random':
#         filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
#     elif context_generation == 'only_helpful' or context_generation == 'optimal':
#         filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
#
#     pickle.dump(inf_over_gens_data_dict, open(output_pickle_file_directory+'lex_inf_data_'+lex_measure+'_'+filename_short+'.p', "wb"))
#
#
#
#





#####################################################################################
## PRIOR 3: EGOCENTRIC
### CONDITION 1: No Selection:

    print ''
    print ''
    print 'This is prior 3: Egocentric'
    print ''
    print 'This is condition 1: No selection'

    perspective_prior_type = 'egocentric'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
    perspective_prior_strength = 0.9  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
    perspective_prior_strength_string = saveresults.convert_array_to_string(perspective_prior_strength)
    selection_type = 'none'

    folder = 'results_iter_'+str(n_meanings)+'M_'+str(n_signals)+'S_select_'+selection_type+'_p_prior_'+perspective_prior_type[0:4]+'_'+perspective_prior_strength_string+'_l_prior_'+lexicon_prior_type[0:4]+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]
    if pilot == True:
        folder = folder+'_PILOT'
    print ''
    print "folder is:"
    print folder

    directory = directory_laptop+folder+'/'
    print ''
    print "output_pickle_file_directory is:"
    print directory

    if selection_type == 'none' or selection_type == 'p_taking':
        if context_generation == 'random':
            filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_select_'+selection_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_select_'+selection_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    elif selection_type == 'ca_with_parent':
        if context_generation == 'random':
            filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type

        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    print ''
    print "filename is:"
    print filename


    multi_run_selected_hyps_per_generation_matrix = unpickle_selected_hyps_matrix(directory, filename, n_copies)
    # print "multi_run_selected_hyps_per_generation_matrix is:"
    # print multi_run_selected_hyps_per_generation_matrix
    print "multi_run_selected_hyps_per_generation_matrix.shape is:"
    print multi_run_selected_hyps_per_generation_matrix.shape

    selected_hyps_new_lex_order_all_runs = get_selected_hyps_ordered(multi_run_selected_hyps_per_generation_matrix, argsort_informativity_per_lexicon)
    print ''
    print ''
    # print "selected_hyps_new_lex_order_all_runs is:"
    # print selected_hyps_new_lex_order_all_runs
    print "selected_hyps_new_lex_order_all_runs.shape is:"
    print selected_hyps_new_lex_order_all_runs.shape


    inf_values_last_generation = get_inf_values_last_generation(1, 2, n_iterations, inf_per_lex_full_hyp_space, selected_hyps_new_lex_order_all_runs)
    print ''
    print ''
    # print "inf_values_last_generation are:"
    # print inf_values_last_generation
    print "inf_values_last_generation.shape is:"
    print inf_values_last_generation.shape


    avg_inf_over_gens_matrix = avg_informativity_over_gens(n_runs, n_copies, n_iterations, inf_per_lex_full_hyp_space, selected_hyps_new_lex_order_all_runs)
    # print "avg_inf_over_gens_matrix is:"
    # print avg_inf_over_gens_matrix
    print "avg_inf_over_gens_matrix.shape is:"
    print avg_inf_over_gens_matrix.shape



    convergence_generations_per_run_select_none = measur.check_convergence(avg_inf_over_gens_matrix, window_convergence_check, bandwith_convergence_check)
    print "convergence_generations_per_run_select_none is:"
    print convergence_generations_per_run_select_none
    print "convergence_generations_per_run_select_none.shape is:"
    print convergence_generations_per_run_select_none.shape
    print "np.amin(convergence_generations_per_run_select_none) is:"
    print np.amin(convergence_generations_per_run_select_none)
    print "np.mean(convergence_generations_per_run_select_none) is:"
    print np.mean(convergence_generations_per_run_select_none)
    print "np.amax(convergence_generations_per_run_select_none) is:"
    print np.amax(convergence_generations_per_run_select_none)
    print "np.isnan(convergence_generations_per_run_select_none).any() is:"
    print np.isnan(convergence_generations_per_run_select_none).any()





    mean_inf_over_gens, conf_intervals_inf_over_gens = calc_mean_and_conf_invs_inf_over_gens(avg_inf_over_gens_matrix)
    print ''
    print ''
    # print "mean_inf_over_gens is:"
    # print mean_inf_over_gens
    print "mean_inf_over_gens.shape is:"
    print mean_inf_over_gens.shape
    print ''
    # print "conf_intervals_inf_over_gens is:"
    # print conf_intervals_inf_over_gens
    print "conf_intervals_inf_over_gens.shape is:"
    print conf_intervals_inf_over_gens.shape


    percentiles_inf_over_gens = calc_percentiles_inf_over_gens(avg_inf_over_gens_matrix)
    print ''
    print ''
    print "percentiles_inf_over_gens is:"
    print percentiles_inf_over_gens
    print "percentiles_inf_over_gens.shape is:"
    print percentiles_inf_over_gens.shape


    inf_over_gens_data_dict = {'raw_data':avg_inf_over_gens_matrix,
                                'inf_values_last_generation':inf_values_last_generation,
                               'mean_inf_over_gens':mean_inf_over_gens,
                               'convergence_generations_per_run_select_none':convergence_generations_per_run_select_none,
                               'conf_intervals_inf_over_gens':conf_intervals_inf_over_gens,
                               'percentiles_inf_over_gens':percentiles_inf_over_gens,
                               'unique_inf_values':unique_inf_values,
                               'baseline_inf':baseline_inf,
                               'max_inf':max_inf}

    if context_generation == 'random':
        filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    elif context_generation == 'only_helpful' or context_generation == 'optimal':
        filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type

    pickle.dump(inf_over_gens_data_dict, open(results_directory+'lex_inf_data_'+lex_measure+'_'+filename_short+'.p', "wb"))




#####################################################################################
## PRIOR 3: EGOCENTRIC
### CONDITION 2: Selection on CS:

    print ''
    print ''
    print 'This is prior 3: Egocentric'
    print ''
    print 'This is condition 2: Selection on CS'

    perspective_prior_type = 'egocentric'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
    perspective_prior_strength = 0.9  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
    perspective_prior_strength_string = saveresults.convert_array_to_string(perspective_prior_strength)
    selection_type = 'ca_with_parent'
    if pragmatic_level == 'literal' and selection_type == 'ca_with_parent':
        pragmatic_level = 'perspective-taking'

    folder = 'results_iter_'+str(n_meanings)+'M_'+str(n_signals)+'S_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+ '_p_prior_egoc_09_l_prior_' + lexicon_prior_type[0:4]+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]
    if pilot == True:
        folder = folder+'_PILOT'
    print ''
    print "folder is:"
    print folder

    directory = directory_laptop+folder+'/'
    print ''
    print "output_pickle_file_directory is:"
    print directory

    if selection_type == 'none' or selection_type == 'p_taking':
        if context_generation == 'random':
            filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_turn_'+str(turnover_type)+'_select_'+selection_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_turn_'+str(turnover_type)+'_select_'+selection_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    elif selection_type == 'ca_with_parent':
        if context_generation == 'random':
            filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type

        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    print ''
    print "filename is:"
    print filename

    multi_run_selected_hyps_per_generation_matrix = unpickle_selected_hyps_matrix(directory, filename, n_copies)
    # print "multi_run_selected_hyps_per_generation_matrix is:"
    # print multi_run_selected_hyps_per_generation_matrix
    print "multi_run_selected_hyps_per_generation_matrix.shape is:"
    print multi_run_selected_hyps_per_generation_matrix.shape

    selected_hyps_new_lex_order_all_runs = get_selected_hyps_ordered(multi_run_selected_hyps_per_generation_matrix, argsort_informativity_per_lexicon)
    print ''
    print ''
    # print "selected_hyps_new_lex_order_all_runs is:"
    # print selected_hyps_new_lex_order_all_runs
    print "selected_hyps_new_lex_order_all_runs.shape is:"
    print selected_hyps_new_lex_order_all_runs.shape


    inf_values_last_generation = get_inf_values_last_generation(1, 2, n_iterations, inf_per_lex_full_hyp_space, selected_hyps_new_lex_order_all_runs)
    print ''
    print ''
    # print "inf_values_last_generation are:"
    # print inf_values_last_generation
    print "inf_values_last_generation.shape is:"
    print inf_values_last_generation.shape


    avg_inf_over_gens_matrix = avg_informativity_over_gens(n_runs, n_copies, n_iterations, inf_per_lex_full_hyp_space, selected_hyps_new_lex_order_all_runs)
    print ''
    print ''
    # print "avg_inf_over_gens_matrix is:"
    # print avg_inf_over_gens_matrix
    print "avg_inf_over_gens_matrix.shape is:"
    print avg_inf_over_gens_matrix.shape



    convergence_generations_per_run_select_ca = measur.check_convergence(avg_inf_over_gens_matrix, window_convergence_check, bandwith_convergence_check)
    print "convergence_generations_per_run_select_ca is:"
    print convergence_generations_per_run_select_ca
    print "convergence_generations_per_run_select_ca.shape is:"
    print convergence_generations_per_run_select_ca.shape
    print "np.amin(convergence_generations_per_run_select_ca) is:"
    print np.amin(convergence_generations_per_run_select_ca)
    print "np.mean(convergence_generations_per_run_select_ca) is:"
    print np.mean(convergence_generations_per_run_select_ca)
    print "np.amax(convergence_generations_per_run_select_ca) is:"
    print np.amax(convergence_generations_per_run_select_ca)
    print "np.isnan(convergence_generations_per_run_select_ca).any() is:"
    print np.isnan(convergence_generations_per_run_select_ca).any()




    mean_inf_over_gens, conf_intervals_inf_over_gens = calc_mean_and_conf_invs_inf_over_gens(avg_inf_over_gens_matrix)
    print ''
    print ''
    # print "mean_inf_over_gens is:"
    # print mean_inf_over_gens
    print "mean_inf_over_gens.shape is:"
    print mean_inf_over_gens.shape
    print ''
    # print "conf_intervals_inf_over_gens is:"
    # print conf_intervals_inf_over_gens
    print "conf_intervals_inf_over_gens.shape is:"
    print conf_intervals_inf_over_gens.shape


    percentiles_inf_over_gens = calc_percentiles_inf_over_gens(avg_inf_over_gens_matrix)
    print ''
    print ''
    print "percentiles_inf_over_gens is:"
    print percentiles_inf_over_gens
    print "percentiles_inf_over_gens.shape is:"
    print percentiles_inf_over_gens.shape


    inf_over_gens_data_dict = {'raw_data':avg_inf_over_gens_matrix,
                               'inf_values_last_generation':inf_values_last_generation,
                               'convergence_generations_per_run_select_ca':convergence_generations_per_run_select_ca,
                               'mean_inf_over_gens':mean_inf_over_gens,
                               'conf_intervals_inf_over_gens':conf_intervals_inf_over_gens,
                               'percentiles_inf_over_gens':percentiles_inf_over_gens,
                               'unique_inf_values': unique_inf_values,
                               'baseline_inf':baseline_inf,
                               'max_inf':max_inf}

    if context_generation == 'random':
        filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    elif context_generation == 'only_helpful' or context_generation == 'optimal':
        filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type


    pickle.dump(inf_over_gens_data_dict, open(results_directory+'lex_inf_data_'+lex_measure+'_'+filename_short+'.p', "wb"))


    if pragmatic_level == 'perspective-taking':
        pragmatic_level = 'literal'


#####################################################################################
## PRIOR 3: EGOCENTRIC
### CONDITION 3: Selection on P-taking:

    print ''
    print ''
    print 'This is prior 3: Egocentric'
    print ''
    print 'This is condition 3: Selection on P-taking'

    perspective_prior_type = 'egocentric'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
    perspective_prior_strength = 0.9  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
    perspective_prior_strength_string = saveresults.convert_array_to_string(perspective_prior_strength)
    selection_type = 'p_taking'


    folder = 'results_iter_'+str(n_meanings)+'M_'+str(n_signals)+'S_select_'+selection_type+'_p_prior_egoc_09_l_prior_'+lexicon_prior_type[0:4]+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]
    if pilot == True:
        folder = folder+'_PILOT'
    print ''
    print "folder is:"
    print folder

    directory = directory_laptop+folder+'/'
    print ''
    print "output_pickle_file_directory is:"
    print directory

    if selection_type == 'none' or selection_type == 'p_taking':
        if context_generation == 'random':
            filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_select_'+selection_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_select_'+selection_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    elif selection_type == 'ca_with_parent':
        if context_generation == 'random':
            filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type

        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    print ''
    print "filename is:"
    print filename


    multi_run_selected_hyps_per_generation_matrix = unpickle_selected_hyps_matrix(directory, filename, n_copies)
    # print "multi_run_selected_hyps_per_generation_matrix is:"
    # print multi_run_selected_hyps_per_generation_matrix
    print "multi_run_selected_hyps_per_generation_matrix.shape is:"
    print multi_run_selected_hyps_per_generation_matrix.shape

    selected_hyps_new_lex_order_all_runs = get_selected_hyps_ordered(multi_run_selected_hyps_per_generation_matrix, argsort_informativity_per_lexicon)
    print ''
    print ''
    # print "selected_hyps_new_lex_order_all_runs is:"
    # print selected_hyps_new_lex_order_all_runs
    print "selected_hyps_new_lex_order_all_runs.shape is:"
    print selected_hyps_new_lex_order_all_runs.shape


    inf_values_last_generation = get_inf_values_last_generation(1, 2, n_iterations, inf_per_lex_full_hyp_space, selected_hyps_new_lex_order_all_runs)
    print ''
    print ''
    # print "inf_values_last_generation are:"
    # print inf_values_last_generation
    print "inf_values_last_generation.shape is:"
    print inf_values_last_generation.shape


    avg_inf_over_gens_matrix = avg_informativity_over_gens(n_runs, n_copies, n_iterations, inf_per_lex_full_hyp_space, selected_hyps_new_lex_order_all_runs)
    print ''
    print ''
    # print "avg_inf_over_gens_matrix is:"
    # print avg_inf_over_gens_matrix
    print "avg_inf_over_gens_matrix.shape is:"
    print avg_inf_over_gens_matrix.shape




    convergence_generations_per_run_select_p_taking = measur.check_convergence(avg_inf_over_gens_matrix, window_convergence_check, bandwith_convergence_check)
    print "convergence_generations_per_run_select_p_taking is:"
    print convergence_generations_per_run_select_p_taking
    print "convergence_generations_per_run_select_p_taking.shape is:"
    print convergence_generations_per_run_select_p_taking.shape
    print "np.amin(convergence_generations_per_run_select_p_taking) is:"
    print np.amin(convergence_generations_per_run_select_p_taking)
    print "np.mean(convergence_generations_per_run_select_p_taking) is:"
    print np.mean(convergence_generations_per_run_select_p_taking)
    print "np.amax(convergence_generations_per_run_select_p_taking) is:"
    print np.amax(convergence_generations_per_run_select_p_taking)
    print "np.isnan(convergence_generations_per_run_select_p_taking).any() is:"
    print np.isnan(convergence_generations_per_run_select_p_taking).any()





    mean_inf_over_gens, conf_intervals_inf_over_gens = calc_mean_and_conf_invs_inf_over_gens(avg_inf_over_gens_matrix)
    print ''
    print ''
    # print "mean_inf_over_gens is:"
    # print mean_inf_over_gens
    print "mean_inf_over_gens.shape is:"
    print mean_inf_over_gens.shape
    print ''
    # print "conf_intervals_inf_over_gens is:"
    # print conf_intervals_inf_over_gens
    print "conf_intervals_inf_over_gens.shape is:"
    print conf_intervals_inf_over_gens.shape


    percentiles_inf_over_gens = calc_percentiles_inf_over_gens(avg_inf_over_gens_matrix)
    print ''
    print ''
    print "percentiles_inf_over_gens is:"
    print percentiles_inf_over_gens
    print "percentiles_inf_over_gens.shape is:"
    print percentiles_inf_over_gens.shape


    inf_over_gens_data_dict = {'raw_data':avg_inf_over_gens_matrix,
                               'inf_values_last_generation':inf_values_last_generation,
                               'convergence_generations_per_run_select_p_taking':convergence_generations_per_run_select_p_taking,
                               'mean_inf_over_gens':mean_inf_over_gens,
                               'conf_intervals_inf_over_gens':conf_intervals_inf_over_gens,
                               'percentiles_inf_over_gens':percentiles_inf_over_gens,
                               'unique_inf_values': unique_inf_values,
                               'baseline_inf':baseline_inf,
                               'max_inf':max_inf}

    if context_generation == 'random':
        filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    elif context_generation == 'only_helpful' or context_generation == 'optimal':
        filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type

    pickle.dump(inf_over_gens_data_dict, open(results_directory+'lex_inf_data_'+lex_measure+'_'+filename_short+'.p', "wb"))

