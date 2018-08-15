__author__ = 'Marieke Woensdregt'

from params import *
from lex import Lexicon
import context
from pop import Agent
import measur
import time
import saveresults
import plots
import unpickle_new
import seaborn as sns
import itertools
import matplotlib.pyplot as plt



#######################################################################################################################
# STEP 3: THE PARAMETERS:


# 2.1: The parameters defining the lexicon size (and thus the number of meanings in the world):

n_meanings = 2 # The number of meanings
n_signals = 2 # The number of signals



# 2.2: The parameters defining the contexts and how they map to the agent's saliencies:

context_generation = 'random' # This can be set to either 'random', 'only_helpful', 'most_optimal'
helpful_contexts = np.array([[0.1, 0.7], [0.3, 0.9], [0.1, 0.6], [0.4, 0.9], [0.1, 0.8], [0.2, 0.9], [0.1, 0.5], [0.5, 0.9], [0.1, 0.4], [0.6, 0.9],
[0.7, 0.1], [0.9, 0.3], [0.6, 0.1], [0.9, 0.4], [0.8, 0.1], [0.9, 0.2], [0.5, 0.1], [0.9, 0.5], [0.4, 0.1], [0.9, 0.6]])  # This is a fixed collection of the most helpful contexts (in which the ratio of meaning probability for the one perspective is maximally different from that for the other perspective.
context_type = 'continuous' # This can be set to either 'absolute' for meanings being in/out or 'continuous' for all meanings being present (but in different positions)
context_size = 1 # This parameter is only used if the context_type is 'absolute' and determines the number of meanings present
alpha = 1. # The exponent that is used by the saliency function. sal_alpha=1 gives a directly proportional relationship between meaning distances and meaning saliencies, with sal_alpha>1 the saliency of meanings goes down exponentially as their distance from the agent increases (and with sal_alpha<1 the other way around)



# 2.3: The parameters that determine the make-up of the population:

pop_size = 2
agent_type = 'p_distinction' # Determines whether the population is made up of DistinctionAgent objects or Agent objects (DistinctionAgents can learn different perspectives for different agents, Agents can only learn one perspective for all agents they learn from).


lexicon_types = ['optimal_lex', 'half_ambiguous_lex', 'fully_ambiguous_lex'] # The lexicon types that can be present in the population
#FIXME: In the current implementation 'half_ambiguous_lex' can have different instantiations. Therefore, if we run a simulation where there are different lexicon_type_probs, we will want the run_type to be 'population_same_pop' to make sure that all the 'half ambiguous' speakers do have the SAME half ambiguous lexicon.
lexicon_type_probs = np.array([1., 0., 0.]) # The ratios with which the different lexicon types will be present in the population
lexicon_type_probs_string = convert_array_to_string(lexicon_type_probs) # Turns the lexicon type probs into a string in order to add it to file names





opposite_lexicons = 'no' # Makes sure that 50% of speakers has the one optimal lexicon and 50% have the mirror image of this



error = 0.0 # The error term on production
error_string = str(error)
if len(error_string) == 4:
    error_string = error_string[-2]+error_string[-1]
elif len(error_string) == 3:
    error_string = error_string[-1]+'0'


perspectives = np.array([0., 1.]) # The different perspectives that agents can have
perspective_probs = np.array([0., 1.]) # The ratios with which the different perspectives will be present in the population
perspective_probs_string = convert_array_to_string(perspective_probs) # Turns the perspective probs into a string in order to add it to file names




opposite_perspectives = 'yes' # If this is set to 'yes' the Population.create_pop() method will simply loop through the array of perspectives defined above and will assign them in turn to each agent in the population. If this is set to 'no' the Population.create_pop() method will for each agent simply choose a perspective from the perspectives array with the probabilities determined in perspective_probs above.

alternating_perspectives = 'no' # This is for the iteration condition. If this is set to 'yes', the agent in each generation will have a different perspective from the agent in the previous generation. If it is set to 'no', each new agent will have the same perspective as all previous generations.


learning_types = ['map', 'sample'] # The types of learning that the learners can do
learning_type_probs = np.array([0., 1.]) # The ratios with which the different learning types will be present in the population
learning_type_probs_string = convert_array_to_string(learning_type_probs) # Turns the learning type probs into a string in order to add it to file names
learning_type_string = learning_types[np.where(learning_type_probs==1.)[0]]



# 2.4: The parameters that determine the make-up of an individual speaker (for the dyadic condition):

speaker_perspective = 1. # The speaker's perspective
speaker_lex_type = 'optimal_lex' # The lexicon type of the speaker. This can be set to either 'optimal_lex', 'half_ambiguous_lex', 'fully_ambiguous_lex' or 'specified_lexicon'
speaker_lex_index = 0
speaker_learning_type = 'sample' #FIXME: The speaker has to be initiated with a learning type because I have not yet coded up a subclass of Agent that is only Speaker (for which things like hypothesis space, prior distributiona and learning type would not have to be specified).



# 2.5: The parameters that determine the attributes of the learner:

learner_perspective = 0. # The learner's perspective
learner_lex_type = 'empty_lex' # The lexicon type of the learner. This will normally be 'empty_lex'
learner_learning_type = 'sample' # The type of learning that the learner does. This can be set to either 'map' or 'sample'



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


hypothesis_space = list_hypothesis_space(perspective_hyps, lexicon_hyps) # The full space of composite hypotheses that the learner will consider (2D numpy matrix with composite hypotheses on the rows, perspective hypotheses on column 0 and lexicon hypotheses on column 1)



# 2.7: The parameters that determine the learner's prior:

learner_type = 'both_unknown' # This can be set to either 'perspective_unknown', 'lexicon_unknown' or 'both_unknown'

perspective_prior_type = 'egocentric' # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
perspective_prior_strength = 0.9 # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
perspective_prior_strength_string = convert_array_to_string(perspective_prior_strength)



lexicon_prior_type = 'neutral' # This can be set to either 'neutral', 'ambiguous_fixed', 'half_ambiguous_fixed', 'expressivity_bias' or 'compressibility_bias'.
lexicon_prior_constant = 0.0 # Determines the strength of the lexicon prior, with small c creating a STRONG prior and large c creating a WEAK prior. (And with c = 1000 creating an almost uniform prior.)
lexicon_prior_constant_string = convert_array_to_string(lexicon_prior_constant)






# 2.5: The parameters that determine the amount of data_dict that the learner gets to see, and the amount of runs of the simulation:

n_utterances = 1 # This parameter determines how many signals the learner gets to observe in each context
n_contexts = 500 # The number of contexts that the learner gets to see.
speaker_order_type = 'random_equal' # This can be set to either 'random', 'random_equal' (for random order with making sure that each speaker gets an equal amount of utterances) 'same_first' (first portion of input comes from same perspective, second portion of input from opposite perspective), 'same_first_equal' (where both speakers get to produce the exact same amount of utterances), 'opp_first' (vice versa) or 'opp_first_equal'
first_input_stage_ratio = 0.5 # This is the ratio of contexts that will make up the first input stage (see parameter 'speaker_order_type' above)



# 2.6: The parameters that determine how learning is measured:

theta_fixed = 0.01 # The learner is considered to have acquired the correct hypothesis if the posterior probability for that hypothesis exceeds (1.-theta_fixed)
## The parameters below serve to make a 'convergence time over theta' plot, where the amount of learning trials needed to reach a certain level of convergence on the correct hypothesis is plotted against different values of theta.
theta_start = 0.0
theta_stop = 1.0
theta_step = 0.00001
theta_range = np.arange(theta_start, theta_stop, theta_step)
theta_step_string = str(theta_step)
theta_step_string = theta_step_string.replace(".", "")



# 2.7: The parameters that determine the type and number of simulations that are run:

#FIXME: In the current implementation 'half_ambiguous_lex' can have different instantiations. Therefore, if we run a simulation where there are different lexicon_type_probs, we will want the run_type to be 'population_same_pop' to make sure that all the 'half ambiguous' speakers do have the SAME half ambiguous lexicon.
run_type = 'population_same_pop_dist_learner' # This parameter determines whether the learner communicates with only one speaker ('dyadic') or with a population ('population_diff_pop' if there are no speakers with the 'half_ambiguous' lexicon type, 'population_same_pop' if there are, 'population_same_pop_dist_learner' if the learner can distinguish between different speakers), or whether we do an iterated learning model ('iteration')
turnover_type = 'whole_pop' # The type of turnover with which the population is replaced (for the iteration simulation). This can be set to either 'chain' (one agent at a time) or 'whole_pop' (entire population at once)
n_iterations = 10 # The number of iterations (i.e. new agents in the case of 'chain', generations in the case of 'whole_pop')
report_every_i = 1
cut_off_point = 1
n_runs = 100 # The number of runs of the simulation
report_every_r = 10

which_hyps_on_graph = 'all_hyps' # This is used in the plot_post_probs_over_lex_hyps() function, and can be set to either 'all_hyps' or 'lex_hyps_only'



#######################################################################################################################





def calc_learning_rate(exponentiated_posterior_matrix_for_correct_hyp, context_cap=None):
    progress_accumulator_over_time_matrix = np.zeros((len(exponentiated_posterior_matrix_for_correct_hyp), len(exponentiated_posterior_matrix_for_correct_hyp[0])))
    for r in range(len(exponentiated_posterior_matrix_for_correct_hyp)):
        run = exponentiated_posterior_matrix_for_correct_hyp[r]
        progress_accumulator = 0.
        progress_accumulator_over_time = np.zeros(len(exponentiated_posterior_matrix_for_correct_hyp[0]))
        for c in range(1, len(run)):
            current_context_index = c
            posterior_current_context = run[current_context_index]
            previous_context_index = c-1
            posterior_previous_context = run[previous_context_index]
            progress = posterior_current_context - posterior_previous_context
            progress_accumulator += progress
            progress_accumulator_divided_by_n_contexts = progress_accumulator/c
            progress_accumulator_over_time[c] = progress_accumulator_divided_by_n_contexts
        progress_accumulator_over_time_matrix[r] = progress_accumulator_over_time
    percentile_25 = np.nanpercentile(progress_accumulator_over_time_matrix, q=25, axis=0)
    percentile_50 = np.nanpercentile(progress_accumulator_over_time_matrix, q=50, axis=0)
    percentile_75 = np.nanpercentile(progress_accumulator_over_time_matrix, q=75, axis=0)
    return [percentile_25, percentile_50, percentile_75]





path = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/'

run_type_dir = 'Learner_Pop'

run_type = 'population_same_pop_dist_learner'




#######################################################################################################################
# RANDOM EQUAL:



speaker_order_type = 'random_equal'


if context_generation == 'random':
    file_title = 'same_pop_dist_size_'+str(pop_size)+'_'+str(speaker_order_type)+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+str(context_generation)+'_'+str(context_type)+'_contexts_'+'lex_type_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_'+learner_type+'_lex_hyps_'+which_lexicon_hyps+'_lex_prior_'+lexicon_prior_type+'_opp_L_'+str(opposite_lexicons[0])+'_opp_P_'+str(opposite_perspectives[0])+'_learning_type_probs_'+learning_type_probs_string+'_p_prior_'+perspective_prior_type+'_'+perspective_prior_strength_string+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'

elif context_generation == 'only_helpful':
    file_title = 'same_pop_dist_size_'+str(pop_size)+'_'+str(speaker_order_type)+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(context_type)+'_contexts_'+'lex_type_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_'+learner_type+'_lex_hyps_'+which_lexicon_hyps+'_lex_prior_'+lexicon_prior_type+'_opp_L_'+str(opposite_lexicons[0])+'_opp_P_'+str(opposite_perspectives[0])+'_learning_type_probs_'+learning_type_probs_string+'_p_prior_'+perspective_prior_type+'_'+perspective_prior_strength_string+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'



results_dict = unpickle_new.unpickle(path, run_type_dir, file_title, 'all_results')


multi_run_log_posterior_matrix_random = results_dict['multi_run_log_posterior_matrix']


learner_hypothesis_space = results_dict['learner_hypothesis_space']

# print 
# print "learner_hypothesis_space (with indices) is:"
# print learner_hypothesis_space
#
# print "perspective_hyps are:"
# print perspective_hyps
#
# print "lexicon_hyps are:"
# print lexicon_hyps


multi_run_perspectives_per_speaker_matrix = results_dict['multi_run_perspectives_per_speaker_matrix']

# print 
# print "multi_run_perspectives_per_speaker_matrix is:"
# print multi_run_perspectives_per_speaker_matrix

multi_run_lexicons_per_speaker_matrix = results_dict['multi_run_lexicons_per_speaker_matrix']

# print 
# print "multi_run_lexicons_per_speaker_matrix is:"
# print multi_run_lexicons_per_speaker_matrix

real_speaker_perspectives = multi_run_perspectives_per_speaker_matrix[0]

# print "real_speaker_perspectives are:"
# print real_speaker_perspectives

real_lexicon = multi_run_lexicons_per_speaker_matrix[0][0]

# print "real_lexicon is:"
# print real_lexicon


correct_composite_hyp_indices_random = measur.find_correct_hyp_indices(learner_hypothesis_space, perspective_hyps, lexicon_hyps, real_speaker_perspectives, real_lexicon, 'composite')

print "correct_composite_hyp_indices_random are:"
print correct_composite_hyp_indices_random



print 
print 
print "multi_run_log_posterior_matrix_random.shape for"+speaker_order_type+" is:"
print multi_run_log_posterior_matrix_random.shape



#####################################################################################################
# SAME FIRST EQUAL:


speaker_order_type = 'same_first_equal'


if context_generation == 'random':
    file_title = 'same_pop_dist_size_'+str(pop_size)+'_'+str(speaker_order_type)+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+str(context_generation)+'_'+str(context_type)+'_contexts_'+'lex_type_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_'+learner_type+'_lex_hyps_'+which_lexicon_hyps+'_lex_prior_'+lexicon_prior_type+'_opp_L_'+str(opposite_lexicons[0])+'_opp_P_'+str(opposite_perspectives[0])+'_learning_type_probs_'+learning_type_probs_string+'_p_prior_'+perspective_prior_type+'_'+perspective_prior_strength_string+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'

elif context_generation == 'only_helpful':
    file_title = 'same_pop_dist_size_'+str(pop_size)+'_'+str(speaker_order_type)+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(context_type)+'_contexts_'+'lex_type_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_'+learner_type+'_lex_hyps_'+which_lexicon_hyps+'_lex_prior_'+lexicon_prior_type+'_opp_L_'+str(opposite_lexicons[0])+'_opp_P_'+str(opposite_perspectives[0])+'_learning_type_probs_'+learning_type_probs_string+'_p_prior_'+perspective_prior_type+'_'+perspective_prior_strength_string+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'


results_dict = unpickle_new.unpickle(path, run_type_dir, file_title, 'all_results')


multi_run_log_posterior_matrix_same_first = results_dict['multi_run_log_posterior_matrix']


learner_hypothesis_space = results_dict['learner_hypothesis_space']

# print 
# print "learner_hypothesis_space (with indices) is:"
# print learner_hypothesis_space
#
# print "perspective_hyps are:"
# print perspective_hyps
#
# print "lexicon_hyps are:"
# print lexicon_hyps


multi_run_perspectives_per_speaker_matrix = results_dict['multi_run_perspectives_per_speaker_matrix']

# print 
# print "multi_run_perspectives_per_speaker_matrix is:"
# print multi_run_perspectives_per_speaker_matrix

multi_run_lexicons_per_speaker_matrix = results_dict['multi_run_lexicons_per_speaker_matrix']

# print 
# print "multi_run_lexicons_per_speaker_matrix is:"
# print multi_run_lexicons_per_speaker_matrix

real_speaker_perspectives = multi_run_perspectives_per_speaker_matrix[0]

# print "real_speaker_perspectives are:"
# print real_speaker_perspectives

real_lexicon = multi_run_lexicons_per_speaker_matrix[0][0]

# print "real_lexicon is:"
# print real_lexicon


correct_composite_hyp_indices_same_first = measur.find_correct_hyp_indices(learner_hypothesis_space, perspective_hyps, lexicon_hyps, real_speaker_perspectives, real_lexicon, 'composite')

print "correct_composite_hyp_indices_same_first are:"
print correct_composite_hyp_indices_same_first



print 
print 
print "multi_run_log_posterior_matrix_same_first.shape for"+speaker_order_type+" is:"
print multi_run_log_posterior_matrix_same_first.shape





#
# learning_matrix_correct_composite_hyp_same_first = np.exp(multi_run_log_posterior_matrix_same_first[:,:,correct_composite_hyp_indices_same_first])
#
#
#
# learning_matrix_correct_composite_hyp_same_first_first_half = learning_matrix_correct_composite_hyp_same_first[:, 0:(n_contexts/2)]
#
# print 
# print 
# # print "learning_matrix_correct_composite_hyp_same_first_first_half is:"
# # print learning_matrix_correct_composite_hyp_same_first_first_half
# print "learning_matrix_correct_composite_hyp_same_first_first_half.shape is:"
# print learning_matrix_correct_composite_hyp_same_first_first_half.shape
#
#
# learning_matrix_correct_composite_hyp_same_first_second_half = learning_matrix_correct_composite_hyp_same_first[:, (n_contexts/2):n_contexts]
#
#
#
# print 
# print 
# # print "learning_matrix_correct_composite_hyp_same_first_second_half is:"
# # print learning_matrix_correct_composite_hyp_same_first_second_half
# print "learning_matrix_correct_composite_hyp_same_first_second_half.shape is:"
# print learning_matrix_correct_composite_hyp_same_first_second_half.shape
#
#
#
# learning_rate_same_first_first_half = calc_learning_rate(learning_matrix_correct_composite_hyp_same_first_first_half)
#
#
# print 
# print 
# # print "learning_rate_same_first_first_half is:"
# # print learning_rate_same_first_first_half
# print "len(learning_rate_same_first_first_half) is:"
# print len(learning_rate_same_first_first_half)
#
#
# learning_rate_same_first_second_half = calc_learning_rate(learning_matrix_correct_composite_hyp_same_first_second_half)
#
#
# print 
# print 
# # print "learning_rate_same_first_second_half is:"
# # print learning_rate_same_first_second_half
# print "len(learning_rate_same_first_second_half) is:"
# print len(learning_rate_same_first_second_half)




#####################################################################################################
# SAME FIRST EQUAL:


speaker_order_type = 'opp_first_equal'


if context_generation == 'random':
    file_title = 'same_pop_dist_size_'+str(pop_size)+'_'+str(speaker_order_type)+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+str(context_generation)+'_'+str(context_type)+'_contexts_'+'lex_type_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_'+learner_type+'_lex_hyps_'+which_lexicon_hyps+'_lex_prior_'+lexicon_prior_type+'_opp_L_'+str(opposite_lexicons[0])+'_opp_P_'+str(opposite_perspectives[0])+'_learning_type_probs_'+learning_type_probs_string+'_p_prior_'+perspective_prior_type+'_'+perspective_prior_strength_string+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'

elif context_generation == 'only_helpful':
    file_title = 'same_pop_dist_size_'+str(pop_size)+'_'+str(speaker_order_type)+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(context_type)+'_contexts_'+'lex_type_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_'+learner_type+'_lex_hyps_'+which_lexicon_hyps+'_lex_prior_'+lexicon_prior_type+'_opp_L_'+str(opposite_lexicons[0])+'_opp_P_'+str(opposite_perspectives[0])+'_learning_type_probs_'+learning_type_probs_string+'_p_prior_'+perspective_prior_type+'_'+perspective_prior_strength_string+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'


results_dict = unpickle_new.unpickle(path, run_type_dir, file_title, 'all_results')


multi_run_log_posterior_matrix_opp_first = results_dict['multi_run_log_posterior_matrix']


learner_hypothesis_space = results_dict['learner_hypothesis_space']

# print 
# print "learner_hypothesis_space (with indices) is:"
# print learner_hypothesis_space
#
# print "perspective_hyps are:"
# print perspective_hyps
#
# print "lexicon_hyps are:"
# print lexicon_hyps


multi_run_perspectives_per_speaker_matrix = results_dict['multi_run_perspectives_per_speaker_matrix']

# print 
# print "multi_run_perspectives_per_speaker_matrix is:"
# print multi_run_perspectives_per_speaker_matrix

multi_run_lexicons_per_speaker_matrix = results_dict['multi_run_lexicons_per_speaker_matrix']

# print 
# print "multi_run_lexicons_per_speaker_matrix is:"
# print multi_run_lexicons_per_speaker_matrix

real_speaker_perspectives = multi_run_perspectives_per_speaker_matrix[0]

# print "real_speaker_perspectives are:"
# print real_speaker_perspectives

real_lexicon = multi_run_lexicons_per_speaker_matrix[0][0]

# print "real_lexicon is:"
# print real_lexicon


correct_composite_hyp_indices_opp_first = measur.find_correct_hyp_indices(learner_hypothesis_space, perspective_hyps, lexicon_hyps, real_speaker_perspectives, real_lexicon, 'composite')


print "correct_composite_hyp_indices_opp_first are:"
print correct_composite_hyp_indices_opp_first


print 
print 
print "multi_run_log_posterior_matrix_opp_first.shape for"+speaker_order_type+" is:"
print multi_run_log_posterior_matrix_opp_first.shape



#
# learning_matrix_correct_composite_hyp_opp_first = np.exp(multi_run_log_posterior_matrix_opp_first[:,:,correct_composite_hyp_indices_opp_first])
#
#
# print 
# print 
# print "learning_matrix_correct_composite_hyp_opp_first for"+speaker_order_type+" is:"
# print learning_matrix_correct_composite_hyp_opp_first
#
#
# learning_matrix_correct_composite_hyp_opp_first_first_half = learning_matrix_correct_composite_hyp_opp_first[:, 0:(n_contexts/2)]
#
# print 
# print 
# # print "learning_matrix_correct_composite_hyp_opp_first_first_half is:"
# # print learning_matrix_correct_composite_hyp_opp_first_first_half
# print "learning_matrix_correct_composite_hyp_opp_first_first_half.shape is:"
# print learning_matrix_correct_composite_hyp_opp_first_first_half.shape
#
#
# learning_matrix_correct_composite_hyp_opp_first_second_half = learning_matrix_correct_composite_hyp_opp_first[:, (n_contexts/2):n_contexts]
#
#
#
# print 
# print 
# # print "learning_matrix_correct_composite_hyp_opp_first_second_half is:"
# # print learning_matrix_correct_composite_hyp_opp_first_second_half
# print "learning_matrix_correct_composite_hyp_opp_first_second_half.shape is:"
# print learning_matrix_correct_composite_hyp_opp_first_second_half.shape
#
#
#
# learning_rate_opp_first_first_half = calc_learning_rate(learning_matrix_correct_composite_hyp_opp_first_first_half)
#
#
# print 
# print 
# # print "learning_rate_opp_first_first_half is:"
# # print learning_rate_opp_first_first_half
# print "len(learning_rate_opp_first_first_half) is:"
# print len(learning_rate_opp_first_first_half)
#
#
#
# learning_rate_opp_first_second_half = calc_learning_rate(learning_matrix_correct_composite_hyp_opp_first_second_half)
#
#
# print 
# print 
# # print "learning_rate_opp_first_second_half is:"
# # print learning_rate_opp_first_second_half
# print "len(learning_rate_opp_first_second_half) is:"
# print len(learning_rate_opp_first_second_half)
#


# plt.plot(learning_rate_same_first_first_half[1], label='same_first, first half')
# plt.plot(learning_rate_same_first_second_half[1], label='same_first, second half')
# plt.plot(learning_rate_opp_first_first_half[1], label='opp_first, first half')
# plt.plot(learning_rate_opp_first_second_half[1], label='opp_first, second half')
# plt.legend()
# plt.title('Learning rate for different stages of input in different input orders')
# plt.xlabel('no. of contexts observed')
# plt.ylabel('learning rate (post. prob. gained on correct hyp. per context')
# plt.savefig('Plot_learning_rates_with_different_orders_of_input.png')
# plt.show()







plot_file_path = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Plots/'+run_type_dir
plot_file_title = file_title










# ## Some dummy data:
# percentiles_same_persp_random_equal = np.array([70, 85, 110])
# percentiles_same_persp_same_first = np.array([50, 80, 100])
# percentiles_same_persp_opp_first = np.array([6, 8, 10])
# percentiles_opp_persp_random_equal = np.array([85, 100, 120])
# percentiles_opp_persp_same_first = np.array([10, 12, 15])
# percentiles_opp_persp_opp_first = np.array([95, 120, 180])

#
# data = [percentiles_same_persp_random_equal, percentiles_same_persp_same_first, percentiles_same_persp_opp_first, percentiles_opp_persp_random_equal, percentiles_opp_persp_same_first, percentiles_opp_persp_opp_first]
#
# data_dict = {}
# data_dict['Random'] = {'same perspective':percentiles_same_persp_random_equal, 'opposite perspective':percentiles_opp_persp_random_equal}
# data_dict['Same First'] = {'same perspective':percentiles_same_persp_same_first, 'opposite perspective':percentiles_opp_persp_same_first}
# data_dict['Opposite First'] = {'same perspective':percentiles_same_persp_opp_first, 'opposite perspective':percentiles_opp_persp_opp_first}
#




# AND NOW FOR THE REAL DATA:

multi_run_log_posterior_matrix_same_first_first_half = multi_run_log_posterior_matrix_same_first[:, 0:(n_contexts/2)]
multi_run_log_posterior_matrix_same_first_second_half = multi_run_log_posterior_matrix_same_first[:, (n_contexts/2):n_contexts]

multi_run_log_posterior_matrix_opp_first_first_half = multi_run_log_posterior_matrix_opp_first[:, 0:(n_contexts/2)]
multi_run_log_posterior_matrix_opp_first_second_half = multi_run_log_posterior_matrix_opp_first[:, (n_contexts/2):n_contexts]

print 
print 
# print "multi_run_log_posterior_matrix_random is:"
# print multi_run_log_posterior_matrix_random
print "multi_run_log_posterior_matrix_random.shape is:"
print multi_run_log_posterior_matrix_random.shape


print 
print 
# print "np.exp(multi_run_log_posterior_matrix_same_first_first_half) is:"
# print np.exp(multi_run_log_posterior_matrix_same_first_first_half)
print "multi_run_log_posterior_matrix_same_first_first_half.shape is:"
print multi_run_log_posterior_matrix_same_first_first_half.shape

print 
print 
# print "np.exp(multi_run_log_posterior_matrix_same_first_second_half) is:"
# print np.exp(multi_run_log_posterior_matrix_same_first_second_half)
print "multi_run_log_posterior_matrix_same_first_second_half.shape is:"
print multi_run_log_posterior_matrix_same_first_second_half.shape


print 
print 
# print "np.exp(multi_run_log_posterior_matrix_opp_first_first_half) is:"
# print np.exp(multi_run_log_posterior_matrix_opp_first_first_half)
print "multi_run_log_posterior_matrix_opp_first_first_half.shape is:"
print multi_run_log_posterior_matrix_opp_first_first_half.shape

print 
print 
# print "np.exp(multi_run_log_posterior_matrix_opp_first_second_half) is:"
# print np.exp(multi_run_log_posterior_matrix_opp_first_second_half)
print "multi_run_log_posterior_matrix_opp_first_second_half.shape is:"
print multi_run_log_posterior_matrix_opp_first_second_half.shape



correct_composite_hyp_indices_same_first_first_half = np.array([1, 10])

correct_composite_hyp_indices_same_first_second_half = np.array([10])

correct_composite_hyp_indices_opp_first_first_half = np.array([10, 28])

correct_composite_hyp_indices_opp_first_second_half = np.array([10])




composite_hyp_posterior_mass_correct_random_same_persp = measur.calc_hyp_correct_posterior_mass(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix_random, correct_composite_hyp_indices_same_first_first_half)

print 
print 
# print "composite_hyp_posterior_mass_correct_random_same_persp is:"
# print composite_hyp_posterior_mass_correct_random_same_persp
print "composite_hyp_posterior_mass_correct_random_same_persp.shape is:"
print composite_hyp_posterior_mass_correct_random_same_persp.shape




composite_hyp_posterior_mass_correct_random_opp_persp = measur.calc_hyp_correct_posterior_mass(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix_random, correct_composite_hyp_indices_opp_first_first_half)

print 
print 
# print "composite_hyp_posterior_mass_correct_random_opp_persp is:"
# print composite_hyp_posterior_mass_correct_random_opp_persp
print "composite_hyp_posterior_mass_correct_random_opp_persp.shape is:"
print composite_hyp_posterior_mass_correct_random_opp_persp.shape



composite_hyp_posterior_mass_correct_same_first_first_half = measur.calc_hyp_correct_posterior_mass(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix_same_first_first_half, correct_composite_hyp_indices_same_first_first_half)

print 
print 
# print "composite_hyp_posterior_mass_correct_same_first_first_half is:"
# print composite_hyp_posterior_mass_correct_same_first_first_half
print "composite_hyp_posterior_mass_correct_same_first_first_half.shape is:"
print composite_hyp_posterior_mass_correct_same_first_first_half.shape


composite_hyp_posterior_mass_correct_same_first_second_half = measur.calc_hyp_correct_posterior_mass(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix_same_first_second_half, correct_composite_hyp_indices_same_first_second_half)

print 
print 
# print "composite_hyp_posterior_mass_correct_same_first_second_half is:"
# print composite_hyp_posterior_mass_correct_same_first_second_half
print "composite_hyp_posterior_mass_correct_same_first_second_half.shape is:"
print composite_hyp_posterior_mass_correct_same_first_second_half.shape


composite_hyp_posterior_mass_correct_opp_first_first_half = measur.calc_hyp_correct_posterior_mass(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix_opp_first_first_half, correct_composite_hyp_indices_opp_first_first_half)

print 
print 
# print "composite_hyp_posterior_mass_correct_opp_first_first_half is:"
# print composite_hyp_posterior_mass_correct_opp_first_first_half
print "composite_hyp_posterior_mass_correct_opp_first_first_half.shape is:"
print composite_hyp_posterior_mass_correct_opp_first_first_half.shape


composite_hyp_posterior_mass_correct_opp_first_second_half = measur.calc_hyp_correct_posterior_mass(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix_opp_first_second_half, correct_composite_hyp_indices_opp_first_second_half)

print 
print 
# print "composite_hyp_posterior_mass_correct_opp_first_second_half is:"
# print composite_hyp_posterior_mass_correct_opp_first_second_half
print "composite_hyp_posterior_mass_correct_opp_first_second_half.shape is:"
print composite_hyp_posterior_mass_correct_opp_first_second_half.shape



ceiling = 0.99
print 
print 
print "ceiling is:"
print ceiling



no_of_contexts_for_ceiling_random_same_persp = measur.calc_no_of_contexts_for_ceiling(composite_hyp_posterior_mass_correct_random_same_persp, ceiling)
print 
print 
print "no_of_contexts_for_ceiling_random_same_persp is:"
print no_of_contexts_for_ceiling_random_same_persp
print "no_of_contexts_for_ceiling_random_same_persp.shape is:"
print no_of_contexts_for_ceiling_random_same_persp.shape
print "no_of_contexts_for_ceiling_random_same_persp is:"
no_of_contexts_for_ceiling_random_same_persp_percentile_25 = np.nanpercentile(no_of_contexts_for_ceiling_random_same_persp, q=25)
no_of_contexts_for_ceiling_random_same_persp_percentile_50 = np.nanpercentile(no_of_contexts_for_ceiling_random_same_persp, q=50)
no_of_contexts_for_ceiling_random_same_persp_percentile_75 = np.nanpercentile(no_of_contexts_for_ceiling_random_same_persp, q=75)
no_of_contexts_for_ceiling_random_same_persp_percentiles = [no_of_contexts_for_ceiling_random_same_persp_percentile_25, no_of_contexts_for_ceiling_random_same_persp_percentile_50, no_of_contexts_for_ceiling_random_same_persp_percentile_75]
print "no_of_contexts_for_ceiling_random_same_persp_percentiles are:"
print no_of_contexts_for_ceiling_random_same_persp_percentiles





no_of_contexts_for_ceiling_random_opp_persp = measur.calc_no_of_contexts_for_ceiling(composite_hyp_posterior_mass_correct_random_opp_persp, ceiling)
print 
print 
print "no_of_contexts_for_ceiling_random_opp_persp is:"
print no_of_contexts_for_ceiling_random_opp_persp
print "no_of_contexts_for_ceiling_random_opp_persp.shape is:"
print no_of_contexts_for_ceiling_random_opp_persp.shape
print "no_of_contexts_for_ceiling_random_opp_persp is:"
no_of_contexts_for_ceiling_random_opp_persp_percentile_25 = np.nanpercentile(no_of_contexts_for_ceiling_random_opp_persp, q=25)
no_of_contexts_for_ceiling_random_opp_persp_percentile_50 = np.nanpercentile(no_of_contexts_for_ceiling_random_opp_persp, q=50)
no_of_contexts_for_ceiling_random_opp_persp_percentile_75 = np.nanpercentile(no_of_contexts_for_ceiling_random_opp_persp, q=75)
no_of_contexts_for_ceiling_random_opp_persp_percentiles = [no_of_contexts_for_ceiling_random_opp_persp_percentile_25, no_of_contexts_for_ceiling_random_opp_persp_percentile_50, no_of_contexts_for_ceiling_random_opp_persp_percentile_75]
print "no_of_contexts_for_ceiling_random_opp_persp_percentiles are:"
print no_of_contexts_for_ceiling_random_opp_persp_percentiles





no_of_contexts_for_ceiling_same_first_first_half = measur.calc_no_of_contexts_for_ceiling(composite_hyp_posterior_mass_correct_same_first_first_half, ceiling)
print 
print 
print "no_of_contexts_for_ceiling_same_first_first_half is:"
print no_of_contexts_for_ceiling_same_first_first_half
print "no_of_contexts_for_ceiling_same_first_first_half.shape is:"
print no_of_contexts_for_ceiling_same_first_first_half.shape
print "no_of_contexts_for_ceiling_same_first_first_half is:"
no_of_contexts_for_ceiling_same_first_first_half_percentile_25 = np.nanpercentile(no_of_contexts_for_ceiling_same_first_first_half, q=25)
no_of_contexts_for_ceiling_same_first_first_half_percentile_50 = np.nanpercentile(no_of_contexts_for_ceiling_same_first_first_half, q=50)
no_of_contexts_for_ceiling_same_first_first_half_percentile_75 = np.nanpercentile(no_of_contexts_for_ceiling_same_first_first_half, q=75)
no_of_contexts_for_ceiling_same_first_first_half_percentiles = [no_of_contexts_for_ceiling_same_first_first_half_percentile_25, no_of_contexts_for_ceiling_same_first_first_half_percentile_50, no_of_contexts_for_ceiling_same_first_first_half_percentile_75]
print "no_of_contexts_for_ceiling_same_first_first_half_percentiles are:"
print no_of_contexts_for_ceiling_same_first_first_half_percentiles


no_of_contexts_for_ceiling_same_first_second_half = measur.calc_no_of_contexts_for_ceiling(composite_hyp_posterior_mass_correct_same_first_second_half, ceiling)
print 
print 
print "no_of_contexts_for_ceiling_same_first_second_half is:"
print no_of_contexts_for_ceiling_same_first_second_half
print "no_of_contexts_for_ceiling_same_first_second_half.shape is:"
print no_of_contexts_for_ceiling_same_first_second_half.shape
no_of_contexts_for_ceiling_same_first_second_half_percentile_25 = np.nanpercentile(no_of_contexts_for_ceiling_same_first_second_half, q=25)
no_of_contexts_for_ceiling_same_first_second_half_percentile_50 = np.nanpercentile(no_of_contexts_for_ceiling_same_first_second_half, q=50)
no_of_contexts_for_ceiling_same_first_second_half_percentile_75 = np.nanpercentile(no_of_contexts_for_ceiling_same_first_second_half, q=75)
no_of_contexts_for_ceiling_same_first_second_half_percentiles = [no_of_contexts_for_ceiling_same_first_second_half_percentile_25, no_of_contexts_for_ceiling_same_first_second_half_percentile_50, no_of_contexts_for_ceiling_same_first_second_half_percentile_75]
print "no_of_contexts_for_ceiling_same_first_second_half_percentiles are:"
print no_of_contexts_for_ceiling_same_first_second_half_percentiles



no_of_contexts_for_ceiling_opp_first_first_half = measur.calc_no_of_contexts_for_ceiling(composite_hyp_posterior_mass_correct_opp_first_first_half, ceiling)
print 
print 
print "no_of_contexts_for_ceiling_opp_first_first_half is:"
print no_of_contexts_for_ceiling_opp_first_first_half
print "no_of_contexts_for_ceiling_opp_first_first_half.shape is:"
print no_of_contexts_for_ceiling_opp_first_first_half.shape
no_of_contexts_for_ceiling_opp_first_first_half_percentile_25 = np.nanpercentile(no_of_contexts_for_ceiling_opp_first_first_half, q=25)
no_of_contexts_for_ceiling_opp_first_first_half_percentile_50 = np.nanpercentile(no_of_contexts_for_ceiling_opp_first_first_half, q=50)
no_of_contexts_for_ceiling_opp_first_first_half_percentile_75 = np.nanpercentile(no_of_contexts_for_ceiling_opp_first_first_half, q=75)
no_of_contexts_for_ceiling_opp_first_first_half_percentiles = [no_of_contexts_for_ceiling_opp_first_first_half_percentile_25, no_of_contexts_for_ceiling_opp_first_first_half_percentile_50, no_of_contexts_for_ceiling_opp_first_first_half_percentile_75]
print "no_of_contexts_for_ceiling_opp_first_first_half_percentiles are:"
print no_of_contexts_for_ceiling_opp_first_first_half_percentiles



no_of_contexts_for_ceiling_opp_first_second_half = measur.calc_no_of_contexts_for_ceiling(composite_hyp_posterior_mass_correct_opp_first_second_half, ceiling)
print 
print 
print "no_of_contexts_for_ceiling_opp_first_second_half is:"
print no_of_contexts_for_ceiling_opp_first_second_half
print "no_of_contexts_for_ceiling_opp_first_second_half.shape is:"
print no_of_contexts_for_ceiling_opp_first_second_half.shape
no_of_contexts_for_ceiling_opp_first_second_half_percentile_25 = np.nanpercentile(no_of_contexts_for_ceiling_opp_first_second_half, q=25)
no_of_contexts_for_ceiling_opp_first_second_half_percentile_50 = np.nanpercentile(no_of_contexts_for_ceiling_opp_first_second_half, q=50)
no_of_contexts_for_ceiling_opp_first_second_half_percentile_75 = np.nanpercentile(no_of_contexts_for_ceiling_opp_first_second_half, q=75)
no_of_contexts_for_ceiling_opp_first_second_half_percentiles = [no_of_contexts_for_ceiling_opp_first_second_half_percentile_25, no_of_contexts_for_ceiling_opp_first_second_half_percentile_50, no_of_contexts_for_ceiling_opp_first_second_half_percentile_75]
print "no_of_contexts_for_ceiling_opp_first_second_half_percentiles are:"
print no_of_contexts_for_ceiling_opp_first_second_half_percentiles





## The REAL data:
percentiles_same_persp_random_equal = no_of_contexts_for_ceiling_random_same_persp_percentiles
percentiles_same_persp_same_first = no_of_contexts_for_ceiling_same_first_first_half_percentiles
percentiles_same_persp_opp_first = no_of_contexts_for_ceiling_opp_first_second_half_percentiles
percentiles_opp_persp_random_equal = no_of_contexts_for_ceiling_random_opp_persp_percentiles
percentiles_opp_persp_same_first = no_of_contexts_for_ceiling_same_first_second_half_percentiles
percentiles_opp_persp_opp_first = no_of_contexts_for_ceiling_opp_first_first_half_percentiles


data = [percentiles_same_persp_random_equal, percentiles_same_persp_same_first, percentiles_same_persp_opp_first, percentiles_opp_persp_random_equal, percentiles_opp_persp_same_first, percentiles_opp_persp_opp_first]

data_dict = {}
data_dict['Random'] = {'same perspective':percentiles_same_persp_random_equal, 'opposite perspective':percentiles_opp_persp_random_equal}
data_dict['Same First'] = {'same perspective':percentiles_same_persp_same_first, 'opposite perspective':percentiles_opp_persp_same_first}
data_dict['Opposite First'] = {'same perspective':percentiles_same_persp_opp_first, 'opposite perspective':percentiles_opp_persp_opp_first}






data_dict_opposite_only = {}
data_dict_opposite_only['Randomly Interleaved'] = {'opposite perspective':percentiles_opp_persp_random_equal}
data_dict_opposite_only['Same First'] = {'opposite perspective':percentiles_opp_persp_same_first}
data_dict_opposite_only['Opposite First'] = {'opposite perspective':percentiles_opp_persp_opp_first}




def boxplot_input_order(percentiles_four_conditions_dictionary):
    sns.set_style("whitegrid", {"xtick.major.size":8, "ytick.major.size":8})
    # sns.set_palette("dark")
    sns.set_palette("deep")
    # sns.set_palette("colorblind")
    sns.set_context("poster", font_scale=1.6)
    palette = itertools.cycle(sns.color_palette()[3:])

    fig, axes = plt.subplots(ncols=3, sharey=True)
    fig.subplots_adjust(wspace=0)

    blue = sns.color_palette()[0]
    green = sns.color_palette()[1]
    third_colour = next(palette)
    second_colour = next(palette)

    counter = 0
    for ax, name in zip(axes, ['Random', 'Same First', 'Opposite First']):
        bp = ax.boxplot([data_dict[name][item] for item in ['same perspective', 'opposite perspective']], widths = 0.5, patch_artist=True)
        ## change outline color, fill color and linewidth of the boxes
        for i in range(len(bp['boxes'])):
            if i == 0 or i == 2 or i == 4:
                # change outline color
                bp['boxes'][i].set(color=green, alpha=0.3, linewidth=2)
                # change fill color
                bp['boxes'][i].set(facecolor=green, alpha=0.3)
                ## change color and linewidth of the medians
                bp['medians'][i].set(color=green, linewidth=2)
                # ## change the style of fliers and their fill
                # bp['fliers'][i].set(marker='o', color=odd_boxes_colour, sal_alpha=0.5)
            elif i==1 or i==3 or i==5:
                # change outline color
                bp['boxes'][i].set(color=blue, alpha=0.3, linewidth=2)
                # change fill color
                bp['boxes'][i].set(facecolor=blue, alpha=0.3)
                ## change color and linewidth of the medians))
                bp['medians'][i].set(color=blue, linewidth=2)
                # ## change the style of fliers and their fill
                # bp['fliers'][i].set(marker='o', color=even_boxes_colour, sal_alpha=0.5)
        for j in range(len(bp['whiskers'])):
            if j == 0 or j == 1 or j == 4 or j == 5 or j == 8 or j == 9:
                ## change color and linewidth of the whiskers
                bp['whiskers'][j].set(color=green, linewidth=2)
                ## change color and linewidth of the caps
                bp['caps'][j].set(color=green, linewidth=2)
            elif j == 2 or j == 3 or j == 6 or j == 7 or j == 10 or j == 11:
                ## change color and linewidth of the whiskers
                bp['whiskers'][j].set(color=blue, linewidth=2)
                ## change color and linewidth of the caps
                bp['caps'][j].set(color=blue, linewidth=2)
        # Remove the tick-marks from top and right spines
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        # Set xlabels:
        ax.set(xticklabels=['same p', 'opposite p'], xlabel=name)
        ax.margins(0.1) # Optional
        if counter == 0:
            ax.set_title('Learning different perspectives with different orders of input', x=1.5, y=1.05)
            ax.set_ylabel('No. observations required to learn perspective')
        counter += 1

    fig.subplots_adjust(wspace=0.05)
    plt.gcf().subplots_adjust(bottom=0.15)
    # Save the figure
    plt.savefig(plot_file_path+'/Hist_Input_Order_Helpful_vs_Unhelpful_vs_Random_Separate_3_columns.png')
    plt.show()







def boxplot_input_order_opposite_only(percentiles_four_conditions_dictionary):
    sns.set_style("whitegrid", {"xtick.major.size":8, "ytick.major.size":8})
    # sns.set_palette("dark")
    sns.set_palette("deep")
    # sns.set_palette("colorblind")
    sns.set_context("poster", font_scale=1.6)
    palette = itertools.cycle(sns.color_palette()[3:])

    fig, axes = plt.subplots(ncols=3, sharey=True)
    fig.subplots_adjust(wspace=0)

    blue = sns.color_palette()[0]
    green = sns.color_palette()[1]
    third_colour = next(palette)
    second_colour = next(palette)

    counter = 0
    for ax, name in zip(axes, ['Randomly Interleaved', 'Same First', 'Opposite First']):
        bp = ax.boxplot([percentiles_four_conditions_dictionary[name][item] for item in ['opposite perspective']], widths = 0.5, patch_artist=True)
        ## change outline color, fill color and linewidth of the boxes
        for i in range(len(bp['boxes'])):
            # change outline color
            bp['boxes'][i].set(color=blue, alpha=0.3, linewidth=2)
            # change fill color
            bp['boxes'][i].set(facecolor=blue, alpha=0.3)
            ## change color and linewidth of the medians))
            bp['medians'][i].set(color=blue, linewidth=2)
            # ## change the style of fliers and their fill
            # bp['fliers'][i].set(marker='o', color=even_boxes_colour, sal_alpha=0.5)
        for j in range(len(bp['whiskers'])):
            ## change color and linewidth of the whiskers
            bp['whiskers'][j].set(color=blue, linewidth=2)
            ## change color and linewidth of the caps
            bp['caps'][j].set(color=blue, linewidth=2)
        # Remove the tick-marks from top and right spines
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        # Set xlabels:
        ax.set(xticklabels=['opposite perspective'], xlabel=name)
        ax.margins(0.1) # Optional
        if counter == 0:
            ax.set_title('Learning opposite perspective with different orders of input', x=1.5, y=1.05)
            ax.set_ylabel('No. observations required to learn perspective')
        counter += 1

    fig.subplots_adjust(wspace=0.05)
    plt.gcf().subplots_adjust(bottom=0.15)
    # Save the figure
    plt.savefig(plot_file_path+'/Hist_Input_Order_Helpful_vs_Unhelpful_vs_Random_Separate_3_columns_opposite_only.png')
    plt.show()



boxplot_input_order(data_dict)


boxplot_input_order_opposite_only(data_dict_opposite_only)

