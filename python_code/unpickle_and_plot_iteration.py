__author__ = 'Marieke Woensdregt'


import pop
import lex
import time
import saveresults
import plots
import numpy as np
import prior
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import pickle


np.set_printoptions(threshold=np.nan)

#######################################################################################################################
# STEP 2: SOME PARAMETER RESETTING:


# 2.1: The parameters defining the lexicon size (and thus the number of meanings in the world):

n_meanings = 2 # The number of meanings
n_signals = 2 # The number of signals

# 2.2: The parameters defining the contexts and how they map to the agent's saliencies:

context_generation = 'most_optimal' # This can be set to either 'random', 'only_helpful', 'most_optimal'
#helpful_contexts = np.array([[0.1, 0.7], [0.3, 0.9], [0.1, 0.6], [0.4, 0.9], [0.1, 0.8], [0.2, 0.9], [0.1, 0.5], [0.5, 0.9], [0.1, 0.4], [0.6, 0.9], [0.7, 0.1], [0.9, 0.3], [0.6, 0.1], [0.9, 0.4], [0.8, 0.1], [0.9, 0.2], [0.5, 0.1], [0.9, 0.5], [0.4, 0.1], [0.9, 0.6]])  # This is a fixed collection of the 20 most helpful contexts (in which the ratio of meaning probability for the one perspective is maximally different from that for the other perspective).
helpful_contexts = np.array([[0.7, 0.1], [0.9, 0.3], [0.1, 0.7], [0.3, 0.9]])


# 2.3: The parameters that determine the make-up of the population:

pop_size = 10

agent_type = 'no_p_distinction' # This can be set to either 'p_distinction' or 'no_p_distinction'. Determines whether the population is made up of DistinctionAgent objects or Agent objects (DistinctionAgents can learn different perspectives for different agents, Agents can only learn one perspective for all agents they learn from).


lexicon_types = ['optimal_lex', 'half_ambiguous_lex', 'fully_ambiguous_lex'] # The lexicon types that can be present in the population
#FIXME: In the current implementation 'half_ambiguous_lex' can have different instantiations. Therefore, if we run a simulation where there are different lexicon_type_probs, we will want the run_type to be 'population_same_pop' to make sure that all the 'half ambiguous' speakers do have the SAME half ambiguous lexicon.
lexicon_type_probs = np.array([0., 0., 1.]) # The ratios with which the different lexicon types will be present in the population
lexicon_type_probs_string = convert_array_to_string(lexicon_type_probs) # Turns the lexicon type probs into a string in order to add it to file names


perspectives = np.array([0., 1.]) # The different perspectives that agents can have
perspective_probs = np.array([1., 0.]) # The ratios with which the different perspectives will be present in the population
perspective_probs_string = convert_array_to_string(perspective_probs) # Turns the perspective probs into a string in order to add it to file names


teacher_type = 'single_teacher' # This can be set to either 'single_teacher' or 'multi_teacher'



# 2.6: The parameters that determine the learner's hypothesis space:

perspective_hyps = np.array([0., 1.]) # The perspective hypotheses that the learner will consider (1D numpy array)

which_lexicon_hyps = 'all' # Determines the set of lexicon hypotheses that the learner will consider. This can be set to either 'all', 'all_with_full_s_space' or 'only_optimal'
if which_lexicon_hyps == 'all':
    lexicon_hyps = create_all_lexicons(n_meanings, n_signals) # The lexicon hypotheses that the learner will consider (1D numpy array)
elif which_lexicon_hyps == 'all_with_full_s_space':
    all_lexicon_hyps = create_all_lexicons(n_meanings, n_signals)
    lexicon_hyps = remove_subset_of_signals_lexicons(all_lexicon_hyps) # The lexicon hypotheses that the learner will consider (1D numpy array)
elif which_lexicon_hyps == 'only_optimal':
    lexicon_hyps = create_all_optimal_lexicons(n_meanings, n_signals) # The lexicon hypotheses that the learner will consider (1D numpy array)


if agent_type == 'no_p_distinction':
    hypothesis_space = list_hypothesis_space(perspective_hyps, lexicon_hyps) # The full space of composite hypotheses that the learner will consider (2D numpy matrix with composite hypotheses on the rows, perspective hypotheses on column 0 and lexicon hypotheses on column 1)

elif agent_type == 'p_distinction':
    hypothesis_space = list_hypothesis_space_with_speaker_distinction(perspective_hyps, lexicon_hyps, pop_size) # The full space of composite hypotheses that the learner will consider (2D numpy matrix with composite hypotheses on the rows, perspective hypotheses on column 0 and lexicon hypotheses on column 1)


# 2.7: The parameters that determine the learner's prior:


perspective_prior_type = 'egocentric' # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
perspective_prior_strength = 0.9 # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
perspective_prior_strength_string = convert_array_to_string(perspective_prior_strength)



lexicon_prior_type = 'neutral' # This can be set to either 'neutral', 'ambiguous_fixed', 'half_ambiguous_fixed', 'expressivity_bias' or 'compressibility_bias'
lexicon_prior_constant = 0.0 # Determines the strength of the lexicon bias, with small c creating a STRONG prior and large c creating a WEAK prior. (And with c = 1000 creating an almost uniform prior.)
lexicon_prior_constant_string = convert_array_to_string(lexicon_prior_constant)


n_contexts = 60 # The number of contexts that the learner gets to see.


turnover_type = 'whole_pop' # The type of turnover with which the population is replaced (for the iteration simulation). This can be set to either 'chain' (one agent at a time) or 'whole_pop' (entire population at once)

communication_type = 'lex_n_context' # This can be set to either 'lex_only', 'lex_n_context' or 'lex_n_p'
ca_measure_type = "comp_n_prod" # This can be set to either "comp_n_prod" or "comp_only"
n_interactions = 20 # The number of interactions used to calculate communicative accuracy

selection_type = 'ca_with_parent' # This can be set to either 'none', 'p_taking', 'l_learning' or 'ca_with_parent'
selection_weighting = 'none' # This is a factor with which the fitness of the agents (determined as the probability they assign to the correct perspective hypothesis) is multiplied and then exponentiated in order to weight the relative agent fitness (which in turn determines the probability of becoming a teacher for the next generation). A value of 0. implements neutral selection. A value of 1.0 creates weighting where the fitness is pretty much equal to relative posterior probability on correct p hyp), and the higher the value, the more skewed the weighting in favour of agents with better perspective-taking.
if isinstance(selection_weighting, float):
    selection_weight_string = str(np.int(selection_weighting))
else:
    selection_weight_string = selection_weighting


n_iterations = 1000 # The number of iterations (i.e. new agents in the case of 'chain', generations in the case of 'whole_pop')
report_every_i = 100
cut_off_point = 500
n_runs = 10 # The number of runs of the simulation
report_every_r = 1

recording = 'minimal' # This can be set to either 'everything' or 'minimal'

which_hyps_on_graph = 'lex_hyps_only' # This is used in the plot_post_probs_over_lex_hyps() function, and can be set to either 'all_hyps' or 'lex_hyps_only'

posterior_threshold = 0.99 # This threshold determines how much posterior probability a learner needs to have assigned to the (set of) correct hypothesis/hypotheses in order to say they have 'learned' the correct hypothesis.

#######################################################################################################################




run_type_dir = 'Iteration'


if context_generation == 'random':
    file_title = run_type+'_size_'+str(pop_size)+'_turn_'+str(turnover_type)+'_select_'+selection_type+'_weight_'+selection_weight_string+'_'+ca_measure_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_lex_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type

elif context_generation == 'only_helpful' or context_generation == 'most_optimal':
    file_title = run_type+'_size_'+str(pop_size)+'_turn_'+str(turnover_type)+'_select_'+selection_type+'_weight_'+selection_weight_string+'_'+ca_measure_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_lex_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type



pickle_file_title_all_results = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/'+run_type_dir+'/Results_'+file_title

print 
print "pickle_file_title_all_results is:"
print pickle_file_title_all_results
print 

all_results_dict = pickle.load(open(pickle_file_title_all_results + '.p', 'rb'))


multi_run_selected_hyps_per_generation_matrix = all_results_dict['multi_run_selected_hyps_per_generation_matrix']

#########################################################
# BELOW THE PLOT SHOWING THE PROPORTIONS WITH WHICH DIFFERENT HYPOTHESES ARE SELECTED OVER GENERATIONS IS GENERATED:


# TODO: Code up the hypothesis sorting for p_distinction hypothesis space
if agent_type == 'no_p_distinction' and n_meanings == 2 and n_meanings == 2:
    new_hyp_order_handsorted_on_lexicons = np.array([1, 3, 2, 5, 6, 7, 0, 4, 8, 10, 12, 11, 14, 15, 16, 9, 13, 17])
    lexicon_hyps_sorted = lexicon_hyps[new_hyp_order_handsorted_on_lexicons[0:len(lexicon_hyps)]]

    selected_hyps_new_lex_order = np.zeros((n_runs, (n_iterations-cut_off_point), pop_size))
    for r in range(n_runs):
        for i in range(cut_off_point, n_iterations):
            for a in range(pop_size):
                this_agent_hyp = multi_run_selected_hyps_per_generation_matrix[r][i][a]
                new_order_index = np.argwhere(new_hyp_order_handsorted_on_lexicons == this_agent_hyp)
                selected_hyps_new_lex_order[r][(i-cut_off_point)][a] = new_order_index
    selected_hyps_new_lex_order = selected_hyps_new_lex_order.flatten()


    if which_hyps_on_graph == 'lex_hyps_only':
        n_lex_hyps = len(lexicon_hyps)
        for i in range(len(selected_hyps_new_lex_order)):
            hyp_index = selected_hyps_new_lex_order[i]
            if hyp_index > (n_lex_hyps-1):
                selected_hyps_new_lex_order[i] = hyp_index-n_lex_hyps



    bincount_selected_hyps = np.bincount(selected_hyps_new_lex_order.astype(int))


    hypothesis_count_proportions = np.divide(bincount_selected_hyps.astype(float), np.sum(bincount_selected_hyps.astype(float)))


    plot_file_path = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Plots/Iteration/'

    if context_generation == 'random':
        plot_file_title = 'Plot_Prop_Hyps'+'_size_'+str(pop_size)+'_'+agent_type+'_turn_'+str(turnover_type)+'_select_'+selection_type+'_weight_'+selection_weight_string+'_'+ca_measure_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_lex_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type

    elif context_generation == 'only_helpful' or context_generation == 'most_optimal':
        plot_file_title = 'Plot_Prop_Hyps'+'_size_'+str(pop_size)+'_'+agent_type+'_turn_'+str(turnover_type)+'_select_'+selection_type+'_weight_'+selection_weight_string+'_'+ca_measure_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_lex_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type


    plots.plot_one_iteration_run_hist_hypothesis(plot_file_path, plot_file_title, hypothesis_space, perspective_hyps, lexicon_hyps, lexicon_hyps_sorted, hypothesis_count_proportions, conf_intervals, which_hyps_on_graph, cut_off_point, text_size=1.7)

