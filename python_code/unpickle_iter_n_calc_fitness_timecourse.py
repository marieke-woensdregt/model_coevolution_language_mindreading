__author__ = 'Marieke Woensdregt'


from params_iteration import *
import numpy as np
import pickle
from scipy import stats

# np.set_printoptions(threshold=np.nan)



#######################################################################################################################
# STEP 2: THE PARAMETERS:


# 2.1: The parameters defining the lexicon size (and thus the number of meanings in the world):

n_meanings = 3  # The number of meanings
n_signals = 3  # The number of signals


# 2.2: The parameters defining the contexts and how they map to the agent's saliencies:

context_generation = 'optimal' # This can be set to either 'random', 'only_helpful' or 'optimal'
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


learning_types = ['map', 'sample'] # The types of learning that the learners can do
learning_type_probs = np.array([0., 1.]) # The ratios with which the different learning types will be present in the population
learning_type_probs_string = convert_array_to_string(learning_type_probs) # Turns the learning type probs into a string in order to add it to file names
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

#######################################################################################################################




def get_avg_fitness_matrix(directory, filename, n_copies):
    if n_copies == 1:
        pickle_filename_all_results = 'Results_'+filename+copy_specification
        results_dict = pickle.load(open(directory+pickle_filename_all_results+'.p', 'rb'))
        avg_fitness_matrix_all_runs = results_dict['multi_run_avg_fitness_matrix']
    elif n_copies > 1:
        avg_fitness_matrix_all_runs = np.zeros(((n_copies*n_runs), n_iterations))
        counter = 0
        for c in range(1, n_copies+1):
            pickle_filename_all_results = 'Results_'+filename+'_c'+str(c)
            results_dict = pickle.load(open(directory+pickle_filename_all_results+'.p', 'rb'))
            for r in range(n_runs):
                multi_run_avg_fitness_matrix = results_dict['multi_run_avg_fitness_matrix'][r]
                avg_fitness_matrix_all_runs[counter] = multi_run_avg_fitness_matrix
                counter += 1
    return avg_fitness_matrix_all_runs


def calc_mean_and_conf_invs_fitness_over_gens(avg_fitness_matrix_all_runs):
    mean_avg_fitness_over_gens = np.mean(avg_fitness_matrix_all_runs, axis=0)
    std_avg_fitness_over_gens = np.std(avg_fitness_matrix_all_runs, axis=0)
    conf_intervals_fitness_over_gens = stats.norm.interval(0.95, loc=mean_avg_fitness_over_gens, scale=std_avg_fitness_over_gens / np.sqrt(n_runs*n_copies))
    conf_intervals_fitness_over_gens = np.array(conf_intervals_fitness_over_gens)
    lower_yerr_fitness_over_gens_for_plot = np.subtract(mean_avg_fitness_over_gens, conf_intervals_fitness_over_gens[0])
    upper_yerr_fitness_over_gens_for_plot = np.subtract(conf_intervals_fitness_over_gens[1], mean_avg_fitness_over_gens)
    yerr_fitness_over_gens_for_plot = np.array([lower_yerr_fitness_over_gens_for_plot, upper_yerr_fitness_over_gens_for_plot])
    return mean_avg_fitness_over_gens, yerr_fitness_over_gens_for_plot



def calc_percentiles_fitness_over_gens(avg_fitness_matrix_all_runs):
    percentile_25_fitness_over_gens = np.percentile(avg_fitness_matrix_all_runs, 25, axis=0)
    median_fitness_over_gens = np.percentile(avg_fitness_matrix_all_runs, 50, axis=0)
    percentile_75_fitness_over_gens = np.percentile(avg_fitness_matrix_all_runs, 75, axis=0)
    percentiles_fitness_over_gens = np.array([percentile_25_fitness_over_gens, median_fitness_over_gens, percentile_75_fitness_over_gens])
    return percentiles_fitness_over_gens



def calc_and_pickle_mean_percentiles_avg_fitness(n_runs, n_copies, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type):
    perspective_prior_strength_string = convert_array_to_string(perspective_prior_strength)
    if selection_type == 'none' or selection_type == 'p_taking':
        if context_generation == 'random':
            filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_select_'+selection_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_select_'+selection_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    elif selection_type == 'ca_with_parent':
        if context_generation == 'random':
            filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type

        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    ##
    avg_fitness_matrix_all_runs = get_avg_fitness_matrix(directory, filename, n_copies)
    mean_fitness_over_gens, conf_invs_fitness_over_gens = calc_mean_and_conf_invs_fitness_over_gens(avg_fitness_matrix_all_runs)
    percentiles_fitness_over_gens = calc_percentiles_fitness_over_gens(avg_fitness_matrix_all_runs)
    fitness_timecourse_data = {'raw_data':avg_fitness_matrix_all_runs,
        'mean_fitness_over_gens':mean_fitness_over_gens, 'conf_invs_fitness_over_gens':conf_invs_fitness_over_gens, 'percentiles_fitness_over_gens':percentiles_fitness_over_gens}
    ##
    if selection_type == 'none' or selection_type == 'p_taking':
        if context_generation == 'random':
            filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    elif selection_type == 'ca_with_parent':
        if context_generation == 'random':
            filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    pickle.dump(fitness_timecourse_data, open(results_directory+'fitness_tmcrse_'+filename_short+'.p', "wb"))




#####################################################################################

if __name__ == "__main__":

    directory = '/exports/eddie/scratch/s1370641/'

    directory_laptop = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Eddie_Output/'


    results_directory = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/Iteration/'


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
#     selection_type = 'ca_with_parent'
#
#     folder = 'results_iter_'+str(n_meanings)+'M_'+str(n_signals)+'S_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+ '_p_prior_exo_10_l_prior_' + lexicon_prior_type[0:4]
#     print ''
#     print "folder is:"
#     print folder
#
#     results_directory = directory_laptop+folder+'/'
#     print ''
#     print "results_directory is:"
#     print results_directory
#
#     calc_and_pickle_mean_percentiles_avg_fitness(n_runs, n_copies, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type)
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
#     selection_type = 'p_taking'
#
#     folder = 'results_iter_'+str(n_meanings)+'M_'+str(n_signals)+'S_select_'+selection_type+'_p_prior_exo_10_l_prior_'+lexicon_prior_type[0:4]
#     print ''
#     print "folder is:"
#     print folder
#
#     results_directory = directory_laptop+folder+'/'
#     print ''
#     print "results_directory is:"
#     print results_directory
#
#     calc_and_pickle_mean_percentiles_avg_fitness(n_runs, n_copies, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type)
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
#     selection_type = 'ca_with_parent'
#
#     folder = 'results_iter_'+str(n_meanings)+'M_'+str(n_signals)+'S_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+ '_p_prior_neut_l_prior_' + lexicon_prior_type[0:4]
#     print ''
#     print "folder is:"
#     print folder
#
#     results_directory = directory_laptop+folder+'/'
#     print ''
#     print "results_directory is:"
#     print results_directory
#
#     calc_and_pickle_mean_percentiles_avg_fitness(n_runs, n_copies, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type)
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
#     selection_type = 'p_taking'
#
#     folder = 'results_iter_'+str(n_meanings)+'M_'+str(n_signals)+'S_select_'+selection_type+'_p_prior_neut_l_prior_'+lexicon_prior_type[0:4]
#     print ''
#     print "folder is:"
#     print folder
#
#     results_directory = directory_laptop+folder+'/'
#     print ''
#     print "results_directory is:"
#     print results_directory
#
#     calc_and_pickle_mean_percentiles_avg_fitness(n_runs, n_copies, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type)
#



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
    selection_type = 'ca_with_parent'

    folder = 'results_iter_'+str(n_meanings)+'M_'+str(n_signals)+'S_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+ '_p_prior_egoc_09_l_prior_' + lexicon_prior_type[0:4]+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]
    print ''
    print "folder is:"
    print folder

    directory = directory_laptop+folder+'/'
    print ''
    print "results_directory is:"
    print directory

    calc_and_pickle_mean_percentiles_avg_fitness(n_runs, n_copies, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type)


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
    selection_type = 'p_taking'


    folder = 'results_iter_'+str(n_meanings)+'M_'+str(n_signals)+'S_select_'+selection_type+'_p_prior_egoc_09_l_prior_'+lexicon_prior_type[0:4]+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]
    print ''
    print "folder is:"
    print folder


    directory = directory_laptop+folder+'/'
    print ''
    print "results_directory is:"
    print directory

    calc_and_pickle_mean_percentiles_avg_fitness(n_runs, n_copies, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type)

