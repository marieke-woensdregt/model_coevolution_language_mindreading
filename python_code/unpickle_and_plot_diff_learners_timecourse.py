__author__ = 'Marieke Woensdregt'


import numpy as np
import hypspace
import prior
from saveresults import convert_array_to_string
import pickle
import measur
import plots
import matplotlib.pyplot as plt
import seaborn as sns



def create_perspective_prior(perspective_hyps, lexicon_hyps, perspective_prior_type, perspective_prior_strength):
    """
    :param perspective_hyps: A 1D numpy array containing the perspective hypotheses.
    :param lexicon_hyps: A 3D numpy array containing the lexicon hypotheses.
    :param perspective_prior_type: The type of perspective prior. This can be set to either 'neutral', 'egocentric' or 'same_as_lexicon' (in which case it is also egocentric, and the prior over the opposite perspective is equal to the prior of a particular lexicon given a neutral lexicon prior).
    :param perspective_prior_strength: The strength of the egocentric prior. (This is only used in case the first parameter is set to 'egocentric'.)
    :return: A 1D numpy array containing the prior probabilities corresponding by index to the perspective hypotheses listed in 'perspective_hyps'.
    """
    if perspective_prior_type == 'neutral':
        perspective_prior = np.full_like(range(len(perspective_hyps)), (1./len(perspective_hyps)), dtype=float) # This creates a neutral prior over all perspective hypotheses (1D numpy array)
    elif perspective_prior_type == 'egocentric':
        perspective_prior = np.array([perspective_prior_strength, (1.-perspective_prior_strength)])
    elif perspective_prior_type == 'same_as_lexicon':
        perspective_prior_same_as_lexicon = 1./len(lexicon_hyps)
        perspective_1_prior = perspective_prior_same_as_lexicon
        perspective_0_prior = 1.-perspective_prior_same_as_lexicon
        perspective_prior = np.array([perspective_0_prior, perspective_1_prior])
    return perspective_prior



def create_lexicon_prior(lexicon_hyps, lexicon_prior_type, lexicon_prior_constant):
    """
    :param lexicon_hyps: The agent's lexicon hypotheses (3D numpy array)
    :param lexicon_prior_type: The type of lexicon prior. This can be set to either 'neutral', 'ambiguous_fixed', 'half_ambiguous_fixed' or 'one_to_one_bias'.
    :param lexicon_prior_constant: A parameter that determines the strength of the one-to-one mapping bias (lexicon_prior_constant can range between 0.0 and 1.0. A bias strength of 0.0 creates a neutral bias, and the favouring of one-to-one lexicons goes up exponentially with the bias strength value as its exponent (i.e. the closer to 1.0, the stronger the bias).
    :return: A 1D numpy array with indices corresponding to lexicons in 'lexicon_hyps' array, with a prior probability value on each index
    """
    if lexicon_prior_type == 'neutral':
        lexicon_prior = np.full_like(range(len(lexicon_hyps)), (1./len(lexicon_hyps)), dtype=float) # This creates a neutral prior over all lexicon hypotheses (1D numpy array)
    elif lexicon_prior_type == 'ambiguous_fixed':
        lexicon_prior = np.zeros(len(lexicon_hyps))
        lexicon_prior[-1] = 1. # This creates a fixed prior where ALL prior probability is fixed upon the fully ambiguous lexicon
    elif lexicon_prior_type == 'half_ambiguous_fixed':
        lexicon_prior = np.zeros(len(lexicon_hyps))
        lexicon_prior[2] = 1. # This creates a fixed prior where ALL prior probability is fixed upon one of the 'half ambiguous' lexicons
    elif lexicon_prior_type == 'one_to_one_bias':
        unambiguous_signals_array = np.zeros(len(lexicon_hyps))
        for i in range(len(lexicon_hyps)):
            lexicon_hyp = lexicon_hyps[i]
            column_sums = np.sum(lexicon_hyp, axis=0)
            unambiguous_signals = 0
            for signal in column_sums:
                if signal == 1:
                    unambiguous_signals += 1
            unambiguous_signals_array[i] = unambiguous_signals
        lexicon_prior = np.power(np.e, np.multiply(lexicon_prior_constant, unambiguous_signals_array))
        lexicon_prior = np.divide(lexicon_prior, np.sum(lexicon_prior))
    return lexicon_prior



def list_composite_log_priors(hypothesis_space, perspective_prior, lexicon_prior):
    """
    :param hypothesis_space: The full space of composite hypotheses (2D numpy matrix)
    :param perspective_prior: The prior probability distribution over perspective hypotheses (1D numpy array)
    :param lexicon_prior: The prior probability distribution over lexicon hypotheses (1D numpy array)
    :return: A 1D numpy array that contains the LOG prior for each composite hypothesis (i.e. log(perspective_prior*lexicon_prior))
    """
    priors = np.zeros(len(params.perspective_hyps)*len(params.lexicon_hyps))
    for i in range(len(hypothesis_space)):
        composite_hypothesis = hypothesis_space[i]
        persp_hyp_index = composite_hypothesis[0]
        lex_hyp_index = composite_hypothesis[1]
        prior = perspective_prior[persp_hyp_index]*lexicon_prior[lex_hyp_index]
        log_prior = np.log(prior)
        priors[i] = log_prior
    return priors




#######################################################################################################################
# STEP 3: THE PARAMETERS:


# 2.1: The parameters defining the lexicon size (and thus the number of meanings in the world):

n_meanings = 2 # The number of meanings
n_signals = 2 # The number of signals



# 2.2: The parameters defining the contexts and how they map to the agent's saliencies:

context_type = 'continuous' # This can be set to either 'absolute' for meanings being in/out or 'continuous' for all meanings being present (but in different positions)
context_size = 1 # This parameter is only used if the context_type is 'absolute' and determines the number of meanings present
alpha = 1. # The exponent that is used by the saliency function. sal_alpha=1 gives a directly proportional relationship between meaning distances and meaning saliencies, with sal_alpha>1 the saliency of meanings goes down exponentially as their distance from the agent increases (and with sal_alpha<1 the other way around)



# 2.3: The parameters that determine the make-up of the population:

pop_size = 1


lexicon_types = ['optimal_lex', 'half_ambiguous_lex', 'fully_ambiguous_lex'] # The lexicon types that can be present in the population
#FIXME: In the current implementation 'half_ambiguous_lex' can have different instantiations. Therefore, if we run a simulation where there are different lexicon_type_probs, we will want the run_type to be 'population_same_pop' to make sure that all the 'half ambiguous' speakers do have the SAME half ambiguous lexicon.
lexicon_type_probs = np.array([1., 0., 0.]) # The ratios with which the different lexicon types will be present in the population
lexicon_type_probs_string = convert_array_to_string(lexicon_type_probs) # Turns the lexicon type probs into a string in order to add it to file names





opposite_lexicons = 'no' # Makes sure that 50% of speakers has the one optimal lexicon and 50% have the mirror image of this






perspectives = np.array([0., 1.]) # The different perspectives that agents can have
perspective_probs = np.array([0., 1.]) # The ratios with which the different perspectives will be present in the population
perspective_probs_string = convert_array_to_string(perspective_probs) # Turns the perspective probs into a string in order to add it to file names




opposite_perspectives = 'no'





learning_types = ['map', 'sample'] # The types of learning that the learners can do
learning_type_probs = np.array([1., 0.]) # The ratios with which the different learning types will be present in the population
learning_type_probs_string = convert_array_to_string(learning_type_probs) # Turns the learning type probs into a string in order to add it to file names



# 2.4: The parameters that determine the make-up of an individual speaker (for the dyadic condition):

speaker_perspective = 1. # The speaker's perspective
speaker_lex_type = 'optimal_lex' # The lexicon type of the speaker. This can be set to either 'optimal_lex', 'half_ambiguous_lex' or 'fully_ambiguous_lex'
speaker_learning_type = 'map' #FIXME: The speaker has to be initiated with a learning type because I have not yet coded up a subclass of Agent that is only Speaker (for which things like hypothesis space, prior distributiona and learning type would not have to be specified).



# 2.5: The parameters that determine the attributes of the learner:

learner_perspective = 0. # The learner's perspective
learner_lex_type = 'empty_lex' # The lexicon type of the learner. This will normally be 'empty_lex'
learner_learning_type = 'map' # The type of learning that the learner does. This can be set to either 'map' or 'sample'



# 2.6: The parameters that determine the learner's hypothesis space:

perspective_hyps = np.array([0., 1.]) # The perspective hypotheses that the learner will consider (1D numpy array)

which_lexicon_hyps = 'all' # Determines the set of lexicon hypotheses that the learner will consider. This can be set to either 'all' or 'only_optimal'
if which_lexicon_hyps == 'all':
    lexicon_hyps = hypspace.create_all_lexicons(n_meanings, n_signals) # The lexicon hypotheses that the learner will consider (1D numpy array)
elif which_lexicon_hyps == 'only_optimal':
    lexicon_hyps = hypspace.create_all_optimal_lexicons(n_meanings, n_signals) # The lexicon hypotheses that the learner will consider (1D numpy array)

hypothesis_space = hypspace.list_hypothesis_space(perspective_hyps, lexicon_hyps) # The full space of composite hypotheses that the learner will consider (2D numpy matrix with composite hypotheses on the rows, perspective hypotheses on column 0 and lexicon hypotheses on column 1)



# 2.7: The parameters that determine the learner's prior:

learner_type = 'both_unknown' # This can be set to either 'perspective_unknown', 'lexicon_unknown' or 'both_unknown'

perspective_prior_type = 'egocentric' # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
perspective_prior_strength = 0.9 # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
perspective_prior_strength_string = convert_array_to_string(perspective_prior_strength)



lexicon_prior_type = 'neutral' # This can be set to either 'neutral', 'ambiguous_fixed', 'half_ambiguous_fixed', 'expressivity_bias' or 'compressibility_bias'.
lexicon_prior_constant = 0.1 # A parameter that determines the strength of the one-to-one mapping bias (lexicon_prior_constant can range between 0.0 and 1.0. A bias strength of 0.0 creates a neutral bias, and the favouring of one-to-one lexicons goes up exponentially with the bias strength value as its exponent (i.e. the closer to 1.0, the stronger the bias).
lexicon_prior_constant_string = convert_array_to_string(lexicon_prior_constant)






# 2.5: The parameters that determine the amount of data_dict that the learner gets to see, and the amount of runs of the simulation:

n_utterances = 1 # This parameter determines how many signals the learner gets to observe in each context
n_contexts = 500 # The number of contexts that the learner gets to see.
speaker_order_type = 'random' # This can be set to either 'random', 'random_equal' (for random order with making sure that each speaker gets an equal amount of utterances) 'same_first' (first portion of input comes from same perspective, second portion of input from opposite perspective) or 'opp_first' (vice versa)
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
turnover_type = 'chain' # The type of turnover with which the population is replaced (for the iteration simulation). This can be set to either 'chain' (one agent at a time) or 'whole_pop' (entire population at once)
n_iterations = 1 # The number of iterations (i.e. new agents in the case of 'chain', generations in the case of 'whole_pop')
n_runs = 1000 # The number of runs of the simulation



#######################################################################################################################







if run_type == 'dyadic':
    run_type_dir = 'Learner_Speaker'

elif run_type == 'population_same_pop' or run_type == 'population_diff_pop' or run_type == 'population_same_pop_dist_learner':
    run_type_dir = 'Learner_Pop'

elif run_type == 'iteration':
    run_type_dir = 'Iteration'



if run_type == 'population_diff_pop':
    file_title = 'diff_pop_size_'+str(pop_size)+'_'+str(speaker_order_type)+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+'context_'+str(context_type)+'_lex_type_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_'+learner_type+'_lex_hyps_'+which_lexicon_hyps+'_lex_prior_'+lexicon_prior_type+'_opp_L_'+str(opposite_lexicons[0])+'_opp_P_'+str(opposite_perspectives[0])+'_learning_type_probs_'+learning_type_probs_string+'_p_prior_'+perspective_prior_type+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'

elif run_type == 'population_same_pop':
    file_title = 'same_pop_size_'+str(pop_size)+'_'+str(speaker_order_type)+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+'context_'+str(context_type)+'_lex_type_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_'+learner_type+'_lex_hyps_'+which_lexicon_hyps+'_lex_prior_'+lexicon_prior_type+'_opp_L_'+str(opposite_lexicons[0])+'_opp_P_'+str(opposite_perspectives[0])+'_learning_type_probs_'+learning_type_probs_string+'_p_prior_'+perspective_prior_type+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'

elif run_type == 'population_same_pop_dist_learner':
    file_title = 'same_pop_dist_learner_size_'+str(pop_size)+'_'+str(speaker_order_type)+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+'context_'+str(context_type)+'_lex_type_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_'+learner_type+'_lex_hyps_'+which_lexicon_hyps+'_lex_prior_'+lexicon_prior_type+'_opp_L_'+str(opposite_lexicons[0])+'_opp_P_'+str(opposite_perspectives[0])+'_learning_type_probs_'+learning_type_probs_string+'_p_prior_'+perspective_prior_type+'_'+perspective_prior_strength_string+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'





pickle_file_title = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/'+run_type_dir+'/Results_'+file_title


result_array_list, result_array_keys = pickle.load(open(pickle_file_title+'.p', 'rb'))


multi_run_log_posterior_matrix = measur.get_result_array_from_key(result_array_list, result_array_keys, 'multi_run_log_posterior_matrix')


learner_hypothesis_space = measur.get_result_array_from_key(result_array_list, result_array_keys, 'learner_hypothesis_space')

# print 
# print "learner_hypothesis_space (with indices) is:"
# print learner_hypothesis_space
#
# print "perspective_hyps are:"
# print perspective_hyps
#
# print "lexicon_hyps are:"
# print lexicon_hyps

multi_run_perspectives_per_speaker_matrix = measur.get_result_array_from_key(result_array_list, result_array_keys, 'multi_run_perspectives_per_speaker_matrix')

# print 
# print "multi_run_perspectives_per_speaker_matrix is:"
# print multi_run_perspectives_per_speaker_matrix

multi_run_lexicons_per_speaker_matrix = measur.get_result_array_from_key(result_array_list, result_array_keys, 'multi_run_lexicons_per_speaker_matrix')

# print 
# print "multi_run_lexicons_per_speaker_matrix is:"
# print multi_run_lexicons_per_speaker_matrix

real_speaker_perspectives = multi_run_perspectives_per_speaker_matrix[0]

# print "real_speaker_perspectives are:"
# print real_speaker_perspectives

real_lexicon = multi_run_lexicons_per_speaker_matrix[0][0]

# print "real_lexicon is:"
# print real_lexicon



correct_composite_hyp_indices = measur.find_correct_hyp_indices(learner_hypothesis_space, perspective_hyps, lexicon_hyps, real_speaker_perspectives, real_lexicon, 'composite')

# print "correct_composite_hyp_indices are:"
# print correct_composite_hyp_indices


percentiles_composite_hyp_posterior_mass_correct = measur.calc_hyp_correct_posterior_mass_percentiles(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix, correct_composite_hyp_indices)

# print "percentiles_composite_hyp_posterior_mass_correct is:"
# print percentiles_composite_hyp_posterior_mass_correct
# print "percentiles_composite_hyp_posterior_mass_correct.shape is:"
# print percentiles_composite_hyp_posterior_mass_correct.shape



percentiles_tom_learner_unambiguous_lexicon = percentiles_composite_hyp_posterior_mass_correct




# run_type = 'population_same_pop'


perspective_prior_type = 'zero_order_tom'


if run_type == 'population_diff_pop':
    file_title = 'diff_pop_size_'+str(pop_size)+'_'+str(speaker_order_type)+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+'context_'+str(context_type)+'_lex_type_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_'+learner_type+'_lex_hyps_'+which_lexicon_hyps+'_lex_prior_'+lexicon_prior_type+'_opp_L_'+str(opposite_lexicons[0])+'_opp_P_'+str(opposite_perspectives[0])+'_learning_type_probs_'+learning_type_probs_string+'_p_prior_'+perspective_prior_type+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'

elif run_type == 'population_same_pop':
    file_title = 'same_pop_size_'+str(pop_size)+'_'+str(speaker_order_type)+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+'context_'+str(context_type)+'_lex_type_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_'+learner_type+'_lex_hyps_'+which_lexicon_hyps+'_lex_prior_'+lexicon_prior_type+'_opp_L_'+str(opposite_lexicons[0])+'_opp_P_'+str(opposite_perspectives[0])+'_learning_type_probs_'+learning_type_probs_string+'_p_prior_'+perspective_prior_type+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'

elif run_type == 'population_same_pop_dist_learner':
    file_title = 'same_pop_dist_learner_size_'+str(pop_size)+'_'+str(speaker_order_type)+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+'context_'+str(context_type)+'_lex_type_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_'+learner_type+'_lex_hyps_'+which_lexicon_hyps+'_lex_prior_'+lexicon_prior_type+'_opp_L_'+str(opposite_lexicons[0])+'_opp_P_'+str(opposite_perspectives[0])+'_learning_type_probs_'+learning_type_probs_string+'_p_prior_'+perspective_prior_type+'_'+perspective_prior_strength_string+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'




pickle_file_title = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/'+run_type_dir+'/Results_'+file_title


result_array_list, result_array_keys = pickle.load(open(pickle_file_title+'.p', 'rb'))


multi_run_log_posterior_matrix = measur.get_result_array_from_key(result_array_list, result_array_keys, 'multi_run_log_posterior_matrix')

#
#
# majority_p_hyp_indices = measur.get_result_array_from_key(result_array_list, result_array_keys, 'majority_p_hyp_indices')
#
# majority_lex_hyp_indices = measur.get_result_array_from_key(result_array_list, result_array_keys, 'majority_lex_hyp_indices')
#
# majority_composite_hyp_indices = measur.get_result_array_from_key(result_array_list, result_array_keys, 'majority_composite_hyp_indices')
#
#
# percentiles_composite_hyp_posterior_mass_correct = measur.calc_hyp_correct_posterior_mass(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix, majority_composite_hyp_indices)
# # print 
# # print 
# # print "percentiles_composite_hyp_posterior_mass_correct is:"
# # print percentiles_composite_hyp_posterior_mass_correct
# # print "percentiles_composite_hyp_posterior_mass_correct[0].shape is:"
# # print percentiles_composite_hyp_posterior_mass_correct[0].shape
#

percentiles_single_perspective_learner_same_lexicon = percentiles_composite_hyp_posterior_mass_correct






learner_hypothesis_space = measur.get_result_array_from_key(result_array_list, result_array_keys, 'learner_hypothesis_space')

# print 
# print "learner_hypothesis_space (with indices) is:"
# print learner_hypothesis_space
#
# print "perspective_hyps are:"
# print perspective_hyps
#
# print "lexicon_hyps are:"
# print lexicon_hyps

multi_run_perspectives_per_speaker_matrix = measur.get_result_array_from_key(result_array_list, result_array_keys, 'multi_run_perspectives_per_speaker_matrix')

# print 
# print "multi_run_perspectives_per_speaker_matrix is:"
# print multi_run_perspectives_per_speaker_matrix

multi_run_lexicons_per_speaker_matrix = measur.get_result_array_from_key(result_array_list, result_array_keys, 'multi_run_lexicons_per_speaker_matrix')

# print 
# print "multi_run_lexicons_per_speaker_matrix is:"
# print multi_run_lexicons_per_speaker_matrix

real_speaker_perspectives = multi_run_perspectives_per_speaker_matrix[0]

# print "real_speaker_perspectives are:"
# print real_speaker_perspectives

real_lexicon = multi_run_lexicons_per_speaker_matrix[0][0]

# print "real_lexicon is:"
# print real_lexicon



correct_composite_hyp_indices = measur.find_correct_hyp_indices(learner_hypothesis_space, perspective_hyps, lexicon_hyps, real_speaker_perspectives, real_lexicon, 'composite')

# print "correct_composite_hyp_indices are:"
# print correct_composite_hyp_indices


percentiles_composite_hyp_posterior_mass_correct = measur.calc_hyp_correct_posterior_mass_percentiles(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix, correct_composite_hyp_indices)

# print "percentiles_composite_hyp_posterior_mass_correct is:"
# print percentiles_composite_hyp_posterior_mass_correct
# print "percentiles_composite_hyp_posterior_mass_correct.shape is:"
# print percentiles_composite_hyp_posterior_mass_correct.shape


percentiles_zero_order_tom_learner_unambiguous_lexicon = percentiles_composite_hyp_posterior_mass_correct









perspective_prior_type = 'egocentric'

lexicon_type_probs = np.array([0., 1., 0.]) # The ratios with which the different lexicon types will be present in the population
lexicon_type_probs_string = convert_array_to_string(lexicon_type_probs) # Turns the lexicon type probs into a string in order to add it to file names



if run_type == 'population_diff_pop':
    file_title = 'diff_pop_size_'+str(pop_size)+'_'+str(speaker_order_type)+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+'context_'+str(context_type)+'_lex_type_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_'+learner_type+'_lex_hyps_'+which_lexicon_hyps+'_lex_prior_'+lexicon_prior_type+'_opp_L_'+str(opposite_lexicons[0])+'_opp_P_'+str(opposite_perspectives[0])+'_learning_type_probs_'+learning_type_probs_string+'_p_prior_'+perspective_prior_type+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'

elif run_type == 'population_same_pop':
    file_title = 'same_pop_size_'+str(pop_size)+'_'+str(speaker_order_type)+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+'context_'+str(context_type)+'_lex_type_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_'+learner_type+'_lex_hyps_'+which_lexicon_hyps+'_lex_prior_'+lexicon_prior_type+'_opp_L_'+str(opposite_lexicons[0])+'_opp_P_'+str(opposite_perspectives[0])+'_learning_type_probs_'+learning_type_probs_string+'_p_prior_'+perspective_prior_type+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'

elif run_type == 'population_same_pop_dist_learner':
    file_title = 'same_pop_dist_learner_size_'+str(pop_size)+'_'+str(speaker_order_type)+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+'context_'+str(context_type)+'_lex_type_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_'+learner_type+'_lex_hyps_'+which_lexicon_hyps+'_lex_prior_'+lexicon_prior_type+'_opp_L_'+str(opposite_lexicons[0])+'_opp_P_'+str(opposite_perspectives[0])+'_learning_type_probs_'+learning_type_probs_string+'_p_prior_'+perspective_prior_type+'_'+perspective_prior_strength_string+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'



pickle_file_title = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/'+run_type_dir+'/Results_'+file_title


result_array_list, result_array_keys = pickle.load(open(pickle_file_title+'.p', 'rb'))


multi_run_log_posterior_matrix = measur.get_result_array_from_key(result_array_list, result_array_keys, 'multi_run_log_posterior_matrix')


learner_hypothesis_space = measur.get_result_array_from_key(result_array_list, result_array_keys, 'learner_hypothesis_space')

# print 
# print "learner_hypothesis_space (with indices) is:"
# print learner_hypothesis_space
#
# print "perspective_hyps are:"
# print perspective_hyps
#
# print "lexicon_hyps are:"
# print lexicon_hyps

multi_run_perspectives_per_speaker_matrix = measur.get_result_array_from_key(result_array_list, result_array_keys, 'multi_run_perspectives_per_speaker_matrix')

# print 
# print "multi_run_perspectives_per_speaker_matrix is:"
# print multi_run_perspectives_per_speaker_matrix

multi_run_lexicons_per_speaker_matrix = measur.get_result_array_from_key(result_array_list, result_array_keys, 'multi_run_lexicons_per_speaker_matrix')

# print 
# print "multi_run_lexicons_per_speaker_matrix is:"
# print multi_run_lexicons_per_speaker_matrix

real_speaker_perspectives = multi_run_perspectives_per_speaker_matrix[0]

# print "real_speaker_perspectives are:"
# print real_speaker_perspectives

real_lexicon = multi_run_lexicons_per_speaker_matrix[0][0]

# print "real_lexicon is:"
# print real_lexicon



correct_composite_hyp_indices = measur.find_correct_hyp_indices(learner_hypothesis_space, perspective_hyps, lexicon_hyps, real_speaker_perspectives, real_lexicon, 'composite')

# print "correct_composite_hyp_indices are:"
# print correct_composite_hyp_indices


percentiles_composite_hyp_posterior_mass_correct = measur.calc_hyp_correct_posterior_mass_percentiles(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix, correct_composite_hyp_indices)

# print "percentiles_composite_hyp_posterior_mass_correct is:"
# print percentiles_composite_hyp_posterior_mass_correct
# print "percentiles_composite_hyp_posterior_mass_correct.shape is:"
# print percentiles_composite_hyp_posterior_mass_correct.shape



percentiles_tom_learner_half_ambiguous_lexicon = percentiles_composite_hyp_posterior_mass_correct








lexicon_type_probs = np.array([0., 0., 1.]) # The ratios with which the different lexicon types will be present in the population
lexicon_type_probs_string = convert_array_to_string(lexicon_type_probs) # Turns the lexicon type probs into a string in order to add it to file names



if run_type == 'population_diff_pop':
    file_title = 'diff_pop_size_'+str(pop_size)+'_'+str(speaker_order_type)+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+'context_'+str(context_type)+'_lex_type_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_'+learner_type+'_lex_hyps_'+which_lexicon_hyps+'_lex_prior_'+lexicon_prior_type+'_opp_L_'+str(opposite_lexicons[0])+'_opp_P_'+str(opposite_perspectives[0])+'_learning_type_probs_'+learning_type_probs_string+'_p_prior_'+perspective_prior_type+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'

elif run_type == 'population_same_pop':
    file_title = 'same_pop_size_'+str(pop_size)+'_'+str(speaker_order_type)+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+'context_'+str(context_type)+'_lex_type_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_'+learner_type+'_lex_hyps_'+which_lexicon_hyps+'_lex_prior_'+lexicon_prior_type+'_opp_L_'+str(opposite_lexicons[0])+'_opp_P_'+str(opposite_perspectives[0])+'_learning_type_probs_'+learning_type_probs_string+'_p_prior_'+perspective_prior_type+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'

elif run_type == 'population_same_pop_dist_learner':
    file_title = 'same_pop_dist_learner_size_'+str(pop_size)+'_'+str(speaker_order_type)+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+'context_'+str(context_type)+'_lex_type_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_'+learner_type+'_lex_hyps_'+which_lexicon_hyps+'_lex_prior_'+lexicon_prior_type+'_opp_L_'+str(opposite_lexicons[0])+'_opp_P_'+str(opposite_perspectives[0])+'_learning_type_probs_'+learning_type_probs_string+'_p_prior_'+perspective_prior_type+'_'+perspective_prior_strength_string+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'



pickle_file_title = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/'+run_type_dir+'/Results_'+file_title


result_array_list, result_array_keys = pickle.load(open(pickle_file_title+'.p', 'rb'))


multi_run_log_posterior_matrix = measur.get_result_array_from_key(result_array_list, result_array_keys, 'multi_run_log_posterior_matrix')


learner_hypothesis_space = measur.get_result_array_from_key(result_array_list, result_array_keys, 'learner_hypothesis_space')

# print 
# print "learner_hypothesis_space (with indices) is:"
# print learner_hypothesis_space
#
# print "perspective_hyps are:"
# print perspective_hyps
#
# print "lexicon_hyps are:"
# print lexicon_hyps

multi_run_perspectives_per_speaker_matrix = measur.get_result_array_from_key(result_array_list, result_array_keys, 'multi_run_perspectives_per_speaker_matrix')

# print 
# print "multi_run_perspectives_per_speaker_matrix is:"
# print multi_run_perspectives_per_speaker_matrix

multi_run_lexicons_per_speaker_matrix = measur.get_result_array_from_key(result_array_list, result_array_keys, 'multi_run_lexicons_per_speaker_matrix')

# print 
# print "multi_run_lexicons_per_speaker_matrix is:"
# print multi_run_lexicons_per_speaker_matrix

real_speaker_perspectives = multi_run_perspectives_per_speaker_matrix[0]

# print "real_speaker_perspectives are:"
# print real_speaker_perspectives

real_lexicon = multi_run_lexicons_per_speaker_matrix[0][0]

# print "real_lexicon is:"
# print real_lexicon



correct_composite_hyp_indices = measur.find_correct_hyp_indices(learner_hypothesis_space, perspective_hyps, lexicon_hyps, real_speaker_perspectives, real_lexicon, 'composite')

# print "correct_composite_hyp_indices are:"
# print correct_composite_hyp_indices


percentiles_composite_hyp_posterior_mass_correct = measur.calc_hyp_correct_posterior_mass_percentiles(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix, correct_composite_hyp_indices)

# print "percentiles_composite_hyp_posterior_mass_correct is:"
# print percentiles_composite_hyp_posterior_mass_correct
# print "percentiles_composite_hyp_posterior_mass_correct.shape is:"
# print percentiles_composite_hyp_posterior_mass_correct.shape



percentiles_tom_learner_fully_ambiguous_lexicon = percentiles_composite_hyp_posterior_mass_correct











n_data_points = (n_contexts*n_utterances)+1

plot_title = 'Different types of learners with different speakers'

plot_file_path = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Plots/'+run_type_dir

plot_file_title = 'diff_learners_with_one_speaker_diff_lexicons_2x2_colors_deep_shorter_legend'


percentiles_of_learners_matrix = [percentiles_zero_order_tom_learner_unambiguous_lexicon, percentiles_tom_learner_half_ambiguous_lexicon, percentiles_tom_learner_fully_ambiguous_lexicon, percentiles_tom_learner_unambiguous_lexicon]

percentiles_of_learners_matrix = np.asarray(percentiles_of_learners_matrix)


learner_labels_list = ['No ToM', 'Partly Ambiguous Lexicon', 'Uninformative Lexicon', 'Typical']

plots.plot_timecourse_scores_percentiles_for_different_learners_cogsci_paper(plot_title, plot_file_path, plot_file_title, n_data_points, percentiles_of_learners_matrix, learner_labels_list)

