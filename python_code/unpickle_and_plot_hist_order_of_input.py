__author__ = 'Marieke Woensdregt'


import numpy as np
import pickle

import hypspace
import measur
import plots
import saveresults



# np.set_printoptions(threshold='nan')


#######################################################################################################################
# 1: THE PARAMETERS:


##!!!!!! MAKE SURE TO CHANGE THE PATHS BELOW TO MATCH THE FILE SYSTEM OF YOUR MACHINE:
pickle_file_directory = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/'

plot_file_directory = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Plots/'

run_type_dir = 'Learner_Pop/'



# 1.1: The parameters defining the lexicon size (and thus the number of meanings in the world):

n_meanings = 3  # The number of meanings
n_signals = 3  # The number of signals


# 1.2: The parameters defining the contexts and how they map to the agent's saliencies:

context_generation = 'random'  # This can be set to either 'random', 'only_helpful', 'optimal'
#helpful_contexts = np.array([[0.1, 0.7], [0.3, 0.9], [0.1, 0.6], [0.4, 0.9], [0.1, 0.8], [0.2, 0.9], [0.1, 0.5], [0.5, 0.9], [0.1, 0.4], [0.6, 0.9], [0.7, 0.1], [0.9, 0.3], [0.6, 0.1], [0.9, 0.4], [0.8, 0.1], [0.9, 0.2], [0.5, 0.1], [0.9, 0.5], [0.4, 0.1], [0.9, 0.6]])  # This is a fixed collection of the 20 most helpful contexts (in which the ratio of meaning probability for the one perspective is maximally different from that for the other perspective).
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

sal_alpha = 1.  # The exponent that is used by the saliency function. sal_alpha=1 gives a directly proportional relationship between meaning distances and meaning saliencies, with sal_alpha>1 the saliency of meanings goes down exponentially as their distance from the agent increases (and with sal_alpha<1 the other way around)


error = 0.05  # The error term on production
error_string = saveresults.convert_array_to_string(error)
extra_error = True  # Determines whether the error specified above gets added on top AFTER the pragmatic speaker has calculated its production probabilities by maximising the utility to the listener.


# 1.3: The parameters that determine the make-up of the population:

pop_size = 2


pragmatic_level = 'literal'  # This can be set to either 'literal', 'perspective-taking' or 'prag'
optimality_alpha = 1.0  # Goodman & Stuhlmuller (2013) fitted optimality_alpha = 3.4 to participant data with 4x3 lexicon of three number words ('one', 'two', and 'three') that could be mapped to four different world states (0, 1, 2, and 3)


perspectives = np.array([0., 1.])  # The different perspectives that agents can have
perspective_probs = np.array([.5, .5])  # The proportions with which the different perspectives will be present in the population
perspective_probs_string = saveresults.convert_array_to_string(perspective_probs) # Turns the perspective probs into a string in order to add it to file names


fixed_perspectives = np.array([0., 1.])  # Used to override the perspectives of the population which are otherwise chosen stochastically based on the parameters perspectives and perspective_probs above. This array has to have length equal to pop_size
print ''
print ''
print "fixed_perspectives are:"
print fixed_perspectives

learning_types = ['map', 'sample'] # The types of learning that the learners can do
learning_type_probs = np.array([0., 1.]) # The ratios with which the different learning types will be present in the population
learning_type_probs_string = saveresults.convert_array_to_string(learning_type_probs) # Turns the learning type probs into a string in order to add it to file names
if learning_type_probs[0] == 1.:
    learning_type_string = learning_types[0]
elif learning_type_probs[1] == 1.:
    learning_type_string = learning_types[1]


# 1.4: The parameters that determine the attributes of the learner:

learner_perspective = 0.  # The learner's perspective
learner_lex_type = 'empty_lex'  # The lexicon type of the learner. This will normally be 'empty_lex'
learner_learning_type = 'sample'  # The type of learning that the learner does. This can be set to either 'map' or 'sample'


# 1.5: The parameters that determine the learner's hypothesis space:

perspective_hyps = np.array([0., 1.])  # The perspective hypotheses that the learner will consider (1D numpy array)

which_lexicon_hyps = 'all' # Determines the set of lexicon hypotheses that the learner will consider. This can be set to either 'all', 'all_with_full_s_space' or 'only_optimal'
if which_lexicon_hyps == 'all':
    lexicon_hyps = hypspace.create_all_lexicons(n_meanings, n_signals) # The lexicon hypotheses that the learner will consider (1D numpy array)
elif which_lexicon_hyps == 'all_with_full_s_space':
    all_lexicon_hyps = hypspace.create_all_lexicons(n_meanings, n_signals)
    lexicon_hyps = hypspace.remove_subset_of_signals_lexicons(all_lexicon_hyps) # The lexicon hypotheses that the learner will consider (1D numpy array)
elif which_lexicon_hyps == 'only_optimal':
    lexicon_hyps = hypspace.create_all_optimal_lexicons(n_meanings, n_signals) # The lexicon hypotheses that the learner will consider (1D numpy array)


# 1.6: More parameters that determine the make-up of the population:

lexicon_probs = np.array([0. for x in range(len(lexicon_hyps)-1)]+[1.])

fixed_lexicons = np.array([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]] for x in range(pop_size)])  # Used to override the lexicons of the population which are otherwise chosen stochastically based on the parameters lexicon_hyps and lexicon_probs above. This array has to have length equal to pop_size
print ''
print ''
print "fixed_lexicons are:"
print fixed_lexicons

# 1.7: The parameters that determine the learner's prior:

learner_type = 'both_unknown' # This can be set to either 'perspective_unknown', 'lexicon_unknown' or 'both_unknown'

perspective_prior_type = 'egocentric' # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
perspective_prior_strength = 0.9 # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
perspective_prior_strength_string = saveresults.convert_array_to_string(perspective_prior_strength)


lexicon_prior_type = 'neutral' # This can be set to either 'neutral', 'ambiguous_fixed', 'half_ambiguous_fixed', 'expressivity_bias' or 'compressibility_bias'
lexicon_prior_constant = 0.0 # Determines the strength of the lexicon bias, with small c creating a STRONG prior and large c creating a WEAK prior. (And with c = 1000 creating an almost uniform prior.)
lexicon_prior_constant_string = saveresults.convert_array_to_string(lexicon_prior_constant)



# 1.8: The parameters that determine the amount of data_dict that the learner gets to see, and the amount of runs of the simulation:

n_utterances = 1  # This parameter determines how many signals the learner gets to observe in each context
n_contexts = 800  # The number of contexts that the learner gets to see. If context_generation = 'most_optimal' n_contexts for n_meanings = 2 can be anything in the table of 4 (e.g. 20, 40, 60, 80, 100, etc.); for n_meanings = 3 anything in the table of 12 (e.g. 12, 36, 60, 84, 108, etc.); and for n_meanings = 4 anything in the table of 48 (e.g. 48, 96, 144, etc.).


first_input_stage_ratio = 0.5 # This is the proportion of contexts that will make up the first input stage (see parameter 'speaker_order_type' above)


# 1.9: The parameters that determine how learning is measured:

theta_fixed = 0.01  # The learner is considered to have acquired the correct hypothesis if the posterior probability for that hypothesis exceeds (1.-theta_fixed)
## The parameters below serve to make a 'convergence time over theta' plot, where the amount of learning trials needed to reach a certain level of convergence on the correct hypothesis is plotted against different values of theta.
theta_start = 0.0
theta_stop = 1.0
theta_step = 0.00001
theta_range = np.arange(theta_start, theta_stop, theta_step)
theta_step_string = str(theta_step)
theta_step_string = theta_step_string.replace(".", "")


# 1.10: The parameters that determine the type and number of simulations that are run:

#FIXME: In the current implementation 'half_ambiguous_lex' can have different instantiations. Therefore, if we run a simulation where there are different lexicon_type_probs, we will want the run_type to be 'population_same_pop' to make sure that all the 'half ambiguous' speakers do have the SAME half ambiguous lexicon.
run_type = 'population_same_pop_dist_learner' # This can be set to 'population_diff_pop' if there are no speakers with the 'half_ambiguous' lexicon type, 'population_same_pop' if there are, 'population_same_pop_dist_learner' if the learner can distinguish between different speakers

if run_type == 'population_diff_pop' or run_type == 'population_same_pop':
    agent_type = 'no_p_distinction' # This can be set to either 'p_distinction' or 'no_p_distinction'. Determines whether the population is made up of DistinctionAgent objects or Agent objects (DistinctionAgents can learn different perspectives for different agents, Agents can only learn one perspective for all agents they learn from).
elif run_type == 'population_same_pop_dist_learner':
    agent_type = 'p_distinction' # This can be set to either 'p_distinction' or 'no_p_distinction'. Determines whether the population is made up of DistinctionAgent objects or Agent objects (DistinctionAgents can learn different perspectives for different agents, Agents can only learn one perspective for all agents they learn from).


if agent_type == 'no_p_distinction':
    hypothesis_space = hypspace.list_hypothesis_space(perspective_hyps, lexicon_hyps) # The full space of composite hypotheses that the learner will consider (2D numpy matrix with composite hypotheses on the rows, perspective hypotheses on column 0 and lexicon hypotheses on column 1)

elif agent_type == 'p_distinction':
    hypothesis_space = hypspace.list_hypothesis_space_with_speaker_distinction(perspective_hyps, lexicon_hyps, pop_size) # The full space of composite hypotheses that the learner will consider (2D numpy matrix with composite hypotheses on the rows, perspective hypotheses on column 0 and lexicon hypotheses on column 1)


n_runs = 100  # The number of runs of the simulation
report_every_r = 1

which_hyps_on_graph = 'all_hyps' # This is used in the plot_post_probs_over_lex_hyps() function, and can be set to either 'all_hyps' or 'lex_hyps_only'


ceiling = 0.99
print ''
print ''
print "ceiling is:"
print ceiling

#######################################################################################################################

#
#
#
# def calc_learning_rate(exponentiated_posterior_matrix_for_correct_hyp, context_cap=None):
#     progress_accumulator_over_time_matrix = np.zeros((len(exponentiated_posterior_matrix_for_correct_hyp), len(exponentiated_posterior_matrix_for_correct_hyp[0])))
#     for r in range(len(exponentiated_posterior_matrix_for_correct_hyp)):
#         run = exponentiated_posterior_matrix_for_correct_hyp[r]
#         progress_accumulator = 0.
#         progress_accumulator_over_time = np.zeros(len(exponentiated_posterior_matrix_for_correct_hyp[0]))
#         for c in range(1, len(run)):
#             current_context_index = c
#             posterior_current_context = run[current_context_index]
#             previous_context_index = c-1
#             posterior_previous_context = run[previous_context_index]
#             progress = posterior_current_context - posterior_previous_context
#             progress_accumulator += progress
#             progress_accumulator_divided_by_n_contexts = progress_accumulator/c
#             progress_accumulator_over_time[c] = progress_accumulator_divided_by_n_contexts
#         progress_accumulator_over_time_matrix[r] = progress_accumulator_over_time
#     percentile_25 = np.nanpercentile(progress_accumulator_over_time_matrix, q=25, axis=0)
#     percentile_50 = np.nanpercentile(progress_accumulator_over_time_matrix, q=50, axis=0)
#     percentile_75 = np.nanpercentile(progress_accumulator_over_time_matrix, q=75, axis=0)
#     return [percentile_25, percentile_50, percentile_75]
#
#
#
# #######################################################################################################################
#
#
# correct_composite_hyp_indices_per_speaker_order_type = []
# multi_run_log_posterior_matrix_per_speaker_order_type = []
# for speaker_order_type in ['random_equal', 'same_first_equal', 'opp_first_equal']:
#     print ''
#     print ''
#     print ''
#     print ''
#     print "speaker_order_type is:"
#     print speaker_order_type
#
#     if context_generation == 'random':
#         file_title = 'same_pop_dist_size_'+str(pop_size)+'_'+str(speaker_order_type)+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+str(context_generation)+'_'+str(context_type)+'_contxts_'+'p_probs_'+perspective_probs_string+'_'+learner_type+'_lex_hyps_'+which_lexicon_hyps+'_lex_prior_'+lexicon_prior_type+'_learning_type_probs_'+learning_type_probs_string+'_p_prior_'+perspective_prior_type+'_'+perspective_prior_strength_string+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'
#
#     elif context_generation == 'only_helpful' or context_generation == 'optimal':
#         file_title = 'same_pop_dist_size_'+str(pop_size)+'_'+str(speaker_order_type)+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(context_type)+'_contxts_'+'p_probs_'+perspective_probs_string+'_'+learner_type+'_lex_hyps_'+which_lexicon_hyps+'_lex_prior_'+lexicon_prior_type+'_learning_type_probs_'+learning_type_probs_string+'_p_prior_'+perspective_prior_type+'_'+perspective_prior_strength_string+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'
#
#
#     results_pickle_file_title = pickle_file_directory+run_type_dir+'Results_'+file_title
#
#     results_dict = pickle.load(open(results_pickle_file_title+".p", "rb"))
#
#
#     multi_run_log_posterior_matrix = results_dict['multi_run_log_posterior_matrix']
#
#
#     learner_hypothesis_space = results_dict['learner_hypothesis_space']
#
#     # print
#     # print "learner_hypothesis_space (with indices) is:"
#     # print learner_hypothesis_space
#     #
#     # print "perspective_hyps are:"
#     # print perspective_hyps
#     #
#     # print "lexicon_hyps are:"
#     # print lexicon_hyps
#
#
#     multi_run_perspectives_per_speaker_matrix = results_dict['multi_run_perspectives_per_speaker_matrix']
#
#     # print
#     # print "multi_run_perspectives_per_speaker_matrix is:"
#     # print multi_run_perspectives_per_speaker_matrix
#
#     multi_run_lexicons_per_speaker_matrix = results_dict['multi_run_lexicons_per_speaker_matrix']
#
#     # print
#     # print "multi_run_lexicons_per_speaker_matrix is:"
#     # print multi_run_lexicons_per_speaker_matrix
#
#     real_speaker_perspectives = multi_run_perspectives_per_speaker_matrix[0]
#     print ''
#     print ''
#     print "real_speaker_perspectives are:"
#     print real_speaker_perspectives
#
#     real_lexicon = multi_run_lexicons_per_speaker_matrix[0][0]
#
#     print "real_lexicon is:"
#     print real_lexicon
#
#
#
#
#
#     correct_p_hyp_indices_per_speaker = []
#     for speaker_id in range(pop_size):
#         correct_p_hyp_indices = measur.find_correct_hyp_indices_with_speaker_distinction(learner_hypothesis_space, perspective_hyps, lexicon_hyps, real_speaker_perspectives, speaker_id, real_lexicon, 'perspective')
#         correct_p_hyp_indices_per_speaker.append(correct_p_hyp_indices)
#         np.asarray(correct_p_hyp_indices_per_speaker)
#
#
#     for speaker in correct_p_hyp_indices_per_speaker:
#         print ''
#         print ''
#         print "correct_p_hyp_indices this speaker are:"
#         print speaker
#         print "learner_hypothesis_space[speaker[-1]] is:"
#         print learner_hypothesis_space[speaker[-1]]
#         print "perspective_hyps[learner_hypothesis_space[speaker[-1]][0]] is:"
#         print perspective_hyps[learner_hypothesis_space[speaker[-1]][0]]
#
#
#     correct_lex_hyp_indices = measur.find_correct_hyp_indices(learner_hypothesis_space, perspective_hyps, lexicon_hyps, real_speaker_perspectives, real_lexicon, 'lexicon')
#
#
#     print "correct_lex_hyp_indices are:"
#     print correct_lex_hyp_indices
#
#     print "learner_hypothesis_space[correct_lex_hyp_indices] is:"
#     print learner_hypothesis_space[correct_lex_hyp_indices]
#
#     for hyp in learner_hypothesis_space[correct_lex_hyp_indices]:
#         if hyp[1] <= len(lexicon_hyps):
#             print "lexicon_hyps[hyp[1]] is:"
#             print lexicon_hyps[hyp[1]]
#
#
#     correct_composite_hyp_indices = measur.find_correct_hyp_indices(learner_hypothesis_space, perspective_hyps, lexicon_hyps, real_speaker_perspectives, real_lexicon, 'composite')
#
#
#     print "correct_composite_hyp_indices for"+speaker_order_type+" are:"
#     print correct_composite_hyp_indices
#
#     print "learner_hypothesis_space[correct_composite_hyp_indices] is:"
#     print learner_hypothesis_space[correct_composite_hyp_indices]
#
#     if learner_hypothesis_space[correct_composite_hyp_indices][0][1] <= len(lexicon_hyps):
#         print "lexicon_hyps[learner_hypothesis_space[correct_composite_hyp_indices][0][1]] is:"
#         print lexicon_hyps[learner_hypothesis_space[correct_composite_hyp_indices][0][1]]
#
#
#
#
#
#     print ''
#     print ''
#     print ''
#     print ''
#     print 'TO DO: BELOW IS THE STUFF I NEED TO CHECK!'
#
#
#
#     if speaker_order_type == 'random_equal':
#         print ''
#         print ''
#         print 'AWRIGHT! speaker_order_type == same_first_equal'
#         correct_composite_hyp_indices_same_first_first_half = []
#         print ''
#         print "learner_hypothesis_space[correct_lex_hyp_indices] are:"
#         print learner_hypothesis_space[correct_lex_hyp_indices]
#         for index in correct_lex_hyp_indices:
#             print ''
#             print "index is:"
#             print index
#             comp_hyp = learner_hypothesis_space[index]
#             print "comp_hyp is:"
#             print comp_hyp
#             persp_hyp = comp_hyp[0]
#             print "persp_hyp is:"
#             print persp_hyp
#             if persp_hyp[0] == real_speaker_perspectives[0]:
#                 print 'YAY! persp_hyp[0] == real_speaker_perspectives[0]'
#                 correct_composite_hyp_indices_same_first_first_half.append(index)
#         correct_composite_hyp_indices_same_first_first_half = np.array(correct_composite_hyp_indices_same_first_first_half)
#         print ''
#         print "correct_composite_hyp_indices_same_first_first_half is:"
#         print correct_composite_hyp_indices_same_first_first_half
#
#         correct_composite_hyp_indices_same_first_second_half = correct_composite_hyp_indices
#         print ''
#         print "correct_composite_hyp_indices_same_first_second_half is:"
#         print correct_composite_hyp_indices_same_first_second_half
#
#
#
#
#
#
#
#     elif speaker_order_type == 'same_first_equal':
#         print ''
#         print ''
#         print 'AWRIGHT! speaker_order_type == same_first_equal'
#         correct_composite_hyp_indices_same_first_first_half = []
#         print ''
#         print "learner_hypothesis_space[correct_lex_hyp_indices] are:"
#         print learner_hypothesis_space[correct_lex_hyp_indices]
#         for index in correct_lex_hyp_indices:
#             print ''
#             print "index is:"
#             print index
#             comp_hyp = learner_hypothesis_space[index]
#             print "comp_hyp is:"
#             print comp_hyp
#             persp_hyp = comp_hyp[0]
#             print "persp_hyp is:"
#             print persp_hyp
#             if persp_hyp[0] == real_speaker_perspectives[0]:
#                 print 'YAY! persp_hyp[0] == real_speaker_perspectives[0]'
#                 correct_composite_hyp_indices_same_first_first_half.append(index)
#         correct_composite_hyp_indices_same_first_first_half = np.array(correct_composite_hyp_indices_same_first_first_half)
#         print ''
#         print "correct_composite_hyp_indices_same_first_first_half is:"
#         print correct_composite_hyp_indices_same_first_first_half
#
#         correct_composite_hyp_indices_same_first_second_half = correct_composite_hyp_indices
#         print ''
#         print "correct_composite_hyp_indices_same_first_second_half is:"
#         print correct_composite_hyp_indices_same_first_second_half
#
#
#
#     elif speaker_order_type == 'opp_first_equal':
#         print ''
#         print ''
#         print 'WELL OK THEN! speaker_order_type == opp_first_equal'
#         correct_composite_hyp_indices_opp_first_first_half = []
#         print ''
#         print "learner_hypothesis_space[correct_lex_hyp_indices] are:"
#         print learner_hypothesis_space[correct_lex_hyp_indices]
#         for index in correct_lex_hyp_indices:
#             print ''
#             print "index is:"
#             print index
#             comp_hyp = learner_hypothesis_space[index]
#             print "comp_hyp is:"
#             print comp_hyp
#             persp_hyp = comp_hyp[0]
#             print "persp_hyp is:"
#             print persp_hyp
#             if persp_hyp[1] == real_speaker_perspectives[1]:
#                 print 'YAY! persp_hyp[1] == real_speaker_perspectives[1]'
#                 correct_composite_hyp_indices_opp_first_first_half.append(index)
#         correct_composite_hyp_indices_opp_first_first_half = np.array(correct_composite_hyp_indices_opp_first_first_half)
#         print ''
#         print "correct_composite_hyp_indices_opp_first_first_half is:"
#         print correct_composite_hyp_indices_opp_first_first_half
#
#         correct_composite_hyp_indices_opp_first_second_half = correct_composite_hyp_indices
#         print ''
#         print "correct_composite_hyp_indices_opp_first_second_half is:"
#         print correct_composite_hyp_indices_opp_first_second_half
#
#
#
#
#
#
#
#     correct_composite_hyp_indices_per_speaker_order_type.append(correct_composite_hyp_indices)
#
#
#     print ''
#     print ''
#     print "multi_run_log_posterior_matrix.shape for"+speaker_order_type+" is:"
#     print multi_run_log_posterior_matrix.shape
#
#     multi_run_log_posterior_matrix_per_speaker_order_type.append(multi_run_log_posterior_matrix)
#
#
#
#
#
# correct_composite_hyp_indices_per_speaker_order_type = np.array(correct_composite_hyp_indices_per_speaker_order_type)
#
#
# multi_run_log_posterior_matrix_per_speaker_order_type = np.array(multi_run_log_posterior_matrix_per_speaker_order_type)
#
#
#
# multi_run_log_posterior_matrix_same_first_first_half = multi_run_log_posterior_matrix_per_speaker_order_type[1][:, 0:(n_contexts/2)]
# multi_run_log_posterior_matrix_same_first_second_half = multi_run_log_posterior_matrix_per_speaker_order_type[1][:, (n_contexts/2):n_contexts]
#
# multi_run_log_posterior_matrix_opp_first_first_half = multi_run_log_posterior_matrix_per_speaker_order_type[2][:, 0:(n_contexts/2)]
# multi_run_log_posterior_matrix_opp_first_second_half = multi_run_log_posterior_matrix_per_speaker_order_type[2][:, (n_contexts/2):n_contexts]
#
# print ''
# print ''
# # print "multi_run_log_posterior_matrix_per_speaker_order_type[0] is:"
# # print multi_run_log_posterior_matrix_per_speaker_order_type[0]
# print "multi_run_log_posterior_matrix_per_speaker_order_type[0].shape is:"
# print multi_run_log_posterior_matrix_per_speaker_order_type[0].shape
#
# print ''
# print ''
# # print "np.exp(multi_run_log_posterior_matrix_same_first_first_half) is:"
# # print np.exp(multi_run_log_posterior_matrix_same_first_first_half)
# print "multi_run_log_posterior_matrix_same_first_first_half.shape is:"
# print multi_run_log_posterior_matrix_same_first_first_half.shape
#
# print ''
# print ''
# # print "np.exp(multi_run_log_posterior_matrix_same_first_second_half) is:"
# # print np.exp(multi_run_log_posterior_matrix_same_first_second_half)
# print "multi_run_log_posterior_matrix_same_first_second_half.shape is:"
# print multi_run_log_posterior_matrix_same_first_second_half.shape
#
# print ''
# print ''
# # print "np.exp(multi_run_log_posterior_matrix_opp_first_first_half) is:"
# # print np.exp(multi_run_log_posterior_matrix_opp_first_first_half)
# print "multi_run_log_posterior_matrix_opp_first_first_half.shape is:"
# print multi_run_log_posterior_matrix_opp_first_first_half.shape
#
# print ''
# print ''
# # print "np.exp(multi_run_log_posterior_matrix_opp_first_second_half) is:"
# # print np.exp(multi_run_log_posterior_matrix_opp_first_second_half)
# print "multi_run_log_posterior_matrix_opp_first_second_half.shape is:"
# print multi_run_log_posterior_matrix_opp_first_second_half.shape
#
#
#
#
#
# composite_hyp_posterior_mass_correct_random_same_persp = measur.calc_hyp_correct_posterior_mass(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix_per_speaker_order_type[0], correct_composite_hyp_indices_same_first_first_half)
#
# print ''
# print ''
# # print "composite_hyp_posterior_mass_correct_random_same_persp is:"
# # print composite_hyp_posterior_mass_correct_random_same_persp
# print "composite_hyp_posterior_mass_correct_random_same_persp.shape is:"
# print composite_hyp_posterior_mass_correct_random_same_persp.shape
#
#
#
#
# composite_hyp_posterior_mass_correct_random_opp_persp = measur.calc_hyp_correct_posterior_mass(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix_per_speaker_order_type[0], correct_composite_hyp_indices_opp_first_first_half)
#
# print ''
# print ''
# # print "composite_hyp_posterior_mass_correct_random_opp_persp is:"
# # print composite_hyp_posterior_mass_correct_random_opp_persp
# print "composite_hyp_posterior_mass_correct_random_opp_persp.shape is:"
# print composite_hyp_posterior_mass_correct_random_opp_persp.shape
#
#
#
# composite_hyp_posterior_mass_correct_same_first_first_half = measur.calc_hyp_correct_posterior_mass(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix_same_first_first_half, correct_composite_hyp_indices_same_first_first_half)
#
# print ''
# print ''
# # print "composite_hyp_posterior_mass_correct_same_first_first_half is:"
# # print composite_hyp_posterior_mass_correct_same_first_first_half
# print "composite_hyp_posterior_mass_correct_same_first_first_half.shape is:"
# print composite_hyp_posterior_mass_correct_same_first_first_half.shape
#
#
# composite_hyp_posterior_mass_correct_same_first_second_half = measur.calc_hyp_correct_posterior_mass(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix_same_first_second_half, correct_composite_hyp_indices_same_first_second_half)
#
# print ''
# print ''
# # print "composite_hyp_posterior_mass_correct_same_first_second_half is:"
# # print composite_hyp_posterior_mass_correct_same_first_second_half
# print "composite_hyp_posterior_mass_correct_same_first_second_half.shape is:"
# print composite_hyp_posterior_mass_correct_same_first_second_half.shape
#
#
# composite_hyp_posterior_mass_correct_opp_first_first_half = measur.calc_hyp_correct_posterior_mass(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix_opp_first_first_half, correct_composite_hyp_indices_opp_first_first_half)
#
# print ''
# print ''
# # print "composite_hyp_posterior_mass_correct_opp_first_first_half is:"
# # print composite_hyp_posterior_mass_correct_opp_first_first_half
# print "composite_hyp_posterior_mass_correct_opp_first_first_half.shape is:"
# print composite_hyp_posterior_mass_correct_opp_first_first_half.shape
#
#
# composite_hyp_posterior_mass_correct_opp_first_second_half = measur.calc_hyp_correct_posterior_mass(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix_opp_first_second_half, correct_composite_hyp_indices_opp_first_second_half)
#
# print ''
# print ''
# # print "composite_hyp_posterior_mass_correct_opp_first_second_half is:"
# # print composite_hyp_posterior_mass_correct_opp_first_second_half
# print "composite_hyp_posterior_mass_correct_opp_first_second_half.shape is:"
# print composite_hyp_posterior_mass_correct_opp_first_second_half.shape
#
#
#
#
#
#
#
# no_of_contexts_for_ceiling_random_same_persp = measur.calc_no_of_contexts_for_ceiling(composite_hyp_posterior_mass_correct_random_same_persp, ceiling)
# print ''
# print ''
# print "no_of_contexts_for_ceiling_random_same_persp is:"
# print no_of_contexts_for_ceiling_random_same_persp
# print "no_of_contexts_for_ceiling_random_same_persp.shape is:"
# print no_of_contexts_for_ceiling_random_same_persp.shape
# print "no_of_contexts_for_ceiling_random_same_persp is:"
# no_of_contexts_for_ceiling_random_same_persp_percentile_25 = np.nanpercentile(no_of_contexts_for_ceiling_random_same_persp, q=25)
# no_of_contexts_for_ceiling_random_same_persp_percentile_50 = np.nanpercentile(no_of_contexts_for_ceiling_random_same_persp, q=50)
# no_of_contexts_for_ceiling_random_same_persp_percentile_75 = np.nanpercentile(no_of_contexts_for_ceiling_random_same_persp, q=75)
# no_of_contexts_for_ceiling_random_same_persp_percentiles = [no_of_contexts_for_ceiling_random_same_persp_percentile_25, no_of_contexts_for_ceiling_random_same_persp_percentile_50, no_of_contexts_for_ceiling_random_same_persp_percentile_75]
# print "no_of_contexts_for_ceiling_random_same_persp_percentiles are:"
# print no_of_contexts_for_ceiling_random_same_persp_percentiles
#
#
#
#
#
# no_of_contexts_for_ceiling_random_opp_persp = measur.calc_no_of_contexts_for_ceiling(composite_hyp_posterior_mass_correct_random_opp_persp, ceiling)
# print ''
# print ''
# print "no_of_contexts_for_ceiling_random_opp_persp is:"
# print no_of_contexts_for_ceiling_random_opp_persp
# print "no_of_contexts_for_ceiling_random_opp_persp.shape is:"
# print no_of_contexts_for_ceiling_random_opp_persp.shape
# print "no_of_contexts_for_ceiling_random_opp_persp is:"
# no_of_contexts_for_ceiling_random_opp_persp_percentile_25 = np.nanpercentile(no_of_contexts_for_ceiling_random_opp_persp, q=25)
# no_of_contexts_for_ceiling_random_opp_persp_percentile_50 = np.nanpercentile(no_of_contexts_for_ceiling_random_opp_persp, q=50)
# no_of_contexts_for_ceiling_random_opp_persp_percentile_75 = np.nanpercentile(no_of_contexts_for_ceiling_random_opp_persp, q=75)
# no_of_contexts_for_ceiling_random_opp_persp_percentiles = [no_of_contexts_for_ceiling_random_opp_persp_percentile_25, no_of_contexts_for_ceiling_random_opp_persp_percentile_50, no_of_contexts_for_ceiling_random_opp_persp_percentile_75]
# print "no_of_contexts_for_ceiling_random_opp_persp_percentiles are:"
# print no_of_contexts_for_ceiling_random_opp_persp_percentiles
#
#
#
#
#
# no_of_contexts_for_ceiling_same_first_first_half = measur.calc_no_of_contexts_for_ceiling(composite_hyp_posterior_mass_correct_same_first_first_half, ceiling)
# print ''
# print ''
# print "no_of_contexts_for_ceiling_same_first_first_half is:"
# print no_of_contexts_for_ceiling_same_first_first_half
# print "no_of_contexts_for_ceiling_same_first_first_half.shape is:"
# print no_of_contexts_for_ceiling_same_first_first_half.shape
# print "no_of_contexts_for_ceiling_same_first_first_half is:"
# no_of_contexts_for_ceiling_same_first_first_half_percentile_25 = np.nanpercentile(no_of_contexts_for_ceiling_same_first_first_half, q=25)
# no_of_contexts_for_ceiling_same_first_first_half_percentile_50 = np.nanpercentile(no_of_contexts_for_ceiling_same_first_first_half, q=50)
# no_of_contexts_for_ceiling_same_first_first_half_percentile_75 = np.nanpercentile(no_of_contexts_for_ceiling_same_first_first_half, q=75)
# no_of_contexts_for_ceiling_same_first_first_half_percentiles = [no_of_contexts_for_ceiling_same_first_first_half_percentile_25, no_of_contexts_for_ceiling_same_first_first_half_percentile_50, no_of_contexts_for_ceiling_same_first_first_half_percentile_75]
# print "no_of_contexts_for_ceiling_same_first_first_half_percentiles are:"
# print no_of_contexts_for_ceiling_same_first_first_half_percentiles
#
#
# no_of_contexts_for_ceiling_same_first_second_half = measur.calc_no_of_contexts_for_ceiling(composite_hyp_posterior_mass_correct_same_first_second_half, ceiling)
# print ''
# print ''
# print "no_of_contexts_for_ceiling_same_first_second_half is:"
# print no_of_contexts_for_ceiling_same_first_second_half
# print "no_of_contexts_for_ceiling_same_first_second_half.shape is:"
# print no_of_contexts_for_ceiling_same_first_second_half.shape
# no_of_contexts_for_ceiling_same_first_second_half_percentile_25 = np.nanpercentile(no_of_contexts_for_ceiling_same_first_second_half, q=25)
# no_of_contexts_for_ceiling_same_first_second_half_percentile_50 = np.nanpercentile(no_of_contexts_for_ceiling_same_first_second_half, q=50)
# no_of_contexts_for_ceiling_same_first_second_half_percentile_75 = np.nanpercentile(no_of_contexts_for_ceiling_same_first_second_half, q=75)
# no_of_contexts_for_ceiling_same_first_second_half_percentiles = [no_of_contexts_for_ceiling_same_first_second_half_percentile_25, no_of_contexts_for_ceiling_same_first_second_half_percentile_50, no_of_contexts_for_ceiling_same_first_second_half_percentile_75]
# print "no_of_contexts_for_ceiling_same_first_second_half_percentiles are:"
# print no_of_contexts_for_ceiling_same_first_second_half_percentiles
#
#
#
# no_of_contexts_for_ceiling_opp_first_first_half = measur.calc_no_of_contexts_for_ceiling(composite_hyp_posterior_mass_correct_opp_first_first_half, ceiling)
# print ''
# print ''
# print "no_of_contexts_for_ceiling_opp_first_first_half is:"
# print no_of_contexts_for_ceiling_opp_first_first_half
# print "no_of_contexts_for_ceiling_opp_first_first_half.shape is:"
# print no_of_contexts_for_ceiling_opp_first_first_half.shape
# no_of_contexts_for_ceiling_opp_first_first_half_percentile_25 = np.nanpercentile(no_of_contexts_for_ceiling_opp_first_first_half, q=25)
# no_of_contexts_for_ceiling_opp_first_first_half_percentile_50 = np.nanpercentile(no_of_contexts_for_ceiling_opp_first_first_half, q=50)
# no_of_contexts_for_ceiling_opp_first_first_half_percentile_75 = np.nanpercentile(no_of_contexts_for_ceiling_opp_first_first_half, q=75)
# no_of_contexts_for_ceiling_opp_first_first_half_percentiles = [no_of_contexts_for_ceiling_opp_first_first_half_percentile_25, no_of_contexts_for_ceiling_opp_first_first_half_percentile_50, no_of_contexts_for_ceiling_opp_first_first_half_percentile_75]
# print "no_of_contexts_for_ceiling_opp_first_first_half_percentiles are:"
# print no_of_contexts_for_ceiling_opp_first_first_half_percentiles
#
#
#
# no_of_contexts_for_ceiling_opp_first_second_half = measur.calc_no_of_contexts_for_ceiling(composite_hyp_posterior_mass_correct_opp_first_second_half, ceiling)
# print ''
# print ''
# print "no_of_contexts_for_ceiling_opp_first_second_half is:"
# print no_of_contexts_for_ceiling_opp_first_second_half
# print "no_of_contexts_for_ceiling_opp_first_second_half.shape is:"
# print no_of_contexts_for_ceiling_opp_first_second_half.shape
# no_of_contexts_for_ceiling_opp_first_second_half_percentile_25 = np.nanpercentile(no_of_contexts_for_ceiling_opp_first_second_half, q=25)
# no_of_contexts_for_ceiling_opp_first_second_half_percentile_50 = np.nanpercentile(no_of_contexts_for_ceiling_opp_first_second_half, q=50)
# no_of_contexts_for_ceiling_opp_first_second_half_percentile_75 = np.nanpercentile(no_of_contexts_for_ceiling_opp_first_second_half, q=75)
# no_of_contexts_for_ceiling_opp_first_second_half_percentiles = [no_of_contexts_for_ceiling_opp_first_second_half_percentile_25, no_of_contexts_for_ceiling_opp_first_second_half_percentile_50, no_of_contexts_for_ceiling_opp_first_second_half_percentile_75]
# print "no_of_contexts_for_ceiling_opp_first_second_half_percentiles are:"
# print no_of_contexts_for_ceiling_opp_first_second_half_percentiles
#
#
#
#
# percentiles_same_persp_random_equal = no_of_contexts_for_ceiling_random_same_persp_percentiles
# percentiles_same_persp_same_first = no_of_contexts_for_ceiling_same_first_first_half_percentiles
# percentiles_same_persp_opp_first = no_of_contexts_for_ceiling_opp_first_second_half_percentiles
# percentiles_opp_persp_random_equal = no_of_contexts_for_ceiling_random_opp_persp_percentiles
# percentiles_opp_persp_same_first = no_of_contexts_for_ceiling_same_first_second_half_percentiles
# percentiles_opp_persp_opp_first = no_of_contexts_for_ceiling_opp_first_first_half_percentiles
#
#
# data = [percentiles_same_persp_random_equal, percentiles_same_persp_same_first, percentiles_same_persp_opp_first, percentiles_opp_persp_random_equal, percentiles_opp_persp_same_first, percentiles_opp_persp_opp_first]
#
# data_dict = {}
# data_dict['Random'] = {'same perspective':percentiles_same_persp_random_equal, 'opposite perspective':percentiles_opp_persp_random_equal}
# data_dict['Same First'] = {'same perspective':percentiles_same_persp_same_first, 'opposite perspective':percentiles_opp_persp_same_first}
# data_dict['Opposite First'] = {'same perspective':percentiles_same_persp_opp_first, 'opposite perspective':percentiles_opp_persp_opp_first}
#
#
#
#
#
#
# data_dict_opposite_only = {}
# data_dict_opposite_only['Randomly Interleaved'] = {'opposite perspective':percentiles_opp_persp_random_equal}
# data_dict_opposite_only['Same First'] = {'opposite perspective':percentiles_opp_persp_same_first}
# data_dict_opposite_only['Opposite First'] = {'opposite perspective':percentiles_opp_persp_opp_first}
#




if context_generation == 'random':
    results_file_title = 'same_pop_dist_size_'+str(pop_size)+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+str(context_generation)+'_'+str(context_type)+'_contxts_'+'p_probs_'+perspective_probs_string+'_'+learner_type+'_lex_hyps_'+which_lexicon_hyps+'_lex_prior_'+lexicon_prior_type+'_learning_type_probs_'+learning_type_probs_string+'_p_prior_'+perspective_prior_type+'_'+perspective_prior_strength_string+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'

elif context_generation == 'only_helpful' or context_generation == 'optimal':
    results_file_title = 'same_pop_dist_size_'+str(pop_size)+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(context_type)+'_contxts_'+'p_probs_'+perspective_probs_string+'_'+learner_type+'_lex_hyps_'+which_lexicon_hyps+'_lex_prior_'+lexicon_prior_type+'_learning_type_probs_'+learning_type_probs_string+'_p_prior_'+perspective_prior_type+'_'+perspective_prior_strength_string+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'


data_dict = pickle.load(open(pickle_file_directory+run_type_dir+'data_dict_'+results_file_title+'.p', 'rb'))


data_dict_opposite_only = pickle.load(open(pickle_file_directory+run_type_dir+'data_dict_opp_'+results_file_title+'.p', 'rb'))



print ''
print ''
print ''
print ''
percentiles_same_persp_random_equal = data_dict['Random']['same perspective']
print "percentiles_same_persp_random_equal are:"
print percentiles_same_persp_random_equal

percentiles_opp_persp_random_equal = data_dict['Random']['opposite perspective']
print ''
print "percentiles_opp_persp_random_equal are:"
print percentiles_opp_persp_random_equal


percentiles_same_persp_same_first = data_dict['Same First']['same perspective']
print ''
print ''
print "percentiles_same_persp_same_first are:"
print percentiles_same_persp_same_first

percentiles_opp_persp_same_first = data_dict['Same First']['opposite perspective']
print ''
print "percentiles_opp_persp_same_first are:"
print percentiles_opp_persp_same_first


percentiles_same_persp_opp_first = data_dict['Opposite First']['same perspective']
print ''
print ''
print "percentiles_same_persp_opp_first are:"
print percentiles_same_persp_opp_first

percentiles_opp_persp_opp_first = data_dict['Opposite First']['opposite perspective']
print ''
print "percentiles_opp_persp_opp_first are:"
print percentiles_opp_persp_opp_first




print ''
print ''
print ''
print ''
percentiles_opp_persp_random_equal = data_dict_opposite_only['Randomly Interleaved']['opposite perspective']
print "percentiles_opp_persp_random_equal are:"
print percentiles_opp_persp_random_equal


percentiles_opp_persp_same_first = data_dict_opposite_only['Same First']['opposite perspective']
print ''
print ''
print "percentiles_opp_persp_same_first are:"
print percentiles_opp_persp_same_first


percentiles_opp_persp_opp_first = data_dict_opposite_only['Opposite First']['opposite perspective']
print ''
print ''
print "percentiles_opp_persp_opp_first are:"
print percentiles_opp_persp_opp_first






plot_file_title = plot_file_directory+run_type_dir+'Hist_Input_Order_Helpful_vs_Unhelpful_vs_Random_Separate_3_columns_'+str(n_meanings)+'M_'+str(n_signals)+'S_'+str(n_runs)+'_R.pdf'

plots.boxplot_input_order(plot_file_title, data_dict)


plot_file_title = plot_file_directory+run_type_dir+'Hist_Input_Order_Helpful_vs_Unhelpful_vs_Random_Separate_3_columns_opposite_only_'+str(n_meanings)+'M_'+str(n_signals)+'S_'+str(n_runs)+'_R.pdf'

plots.boxplot_input_order_opposite_only(plot_file_title, data_dict_opposite_only)









