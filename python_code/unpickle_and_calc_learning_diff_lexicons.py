__author__ = 'Marieke Woensdregt'


import numpy as np
import pickle
from scipy.misc import logsumexp
import time

import hypspace
import lex
import saveresults


#######################################################################################################################
# 1. THE PARAMETERS:


# 1.1: Set the path to the output_pickle_file_directory on your cluster account where you want the results to be stored if you don't want the results to be stored in the folder from which you're running the array job:
results_directory = '/exports/eddie/scratch/s1370641/'

results_directory = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Eddie_Output/'




# 1.2: The parameters defining the lexicon size (and thus the number of meanings in the world):

n_meanings = 3  # The number of meanings
n_signals = 3  # The number of signals



# 1.3: The parameters defining the contexts and how they map to the agent's saliencies:

context_generation = 'random'  # This can be set to either 'random' or 'optimal'. The optimal contexts are hardcoded below in the variable 'helpful_contexts' (see the context.py module for a function called calc_most_informative_contexts() that calculates which contexts are most helpful to the learner).
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



# 1.4: The parameters that determine the make-up of an individual speaker (for the dyadic condition):

speaker_perspective = 1. # The speaker's perspective
speaker_lex_type = 'specified_lexicon' # The lexicon type of the speaker. This can be set to either 'optimal_lex', 'half_ambiguous_lex', 'fully_ambiguous_lex' or 'specified_lexicon'
speaker_learning_type = 'sample' #FIXME: The speaker has to be initiated with a learning type because I have not yet coded up a subclass of Agent that is only Speaker (for which things like hypothesis space, prior distributiona and learning type would not have to be specified).


# 1.5: The parameters that determine the attributes of the learner:

learner_perspective = 0. # The learner's perspective
learner_lex_type = 'empty_lex' # The lexicon type of the learner. This will normally be 'empty_lex'
learner_learning_type = 'sample' # The type of learning that the learner does. This can be set to either 'map' or 'sample'


pragmatic_level_speaker = 'literal'  # This can be set to either 'literal', 'perspective-taking' or 'prag'
optimality_alpha_speaker = 1.0  # Goodman & Stuhlmuller (2013) fitted sal_alpha = 3.4 to participant data with 4x3 lexicon of three number words ('one', 'two', and 'three') that could be mapped to four different world states (0, 1, 2, and 3)


pragmatic_level_learner = 'perspective-taking'  # This can be set to either 'literal', 'perspective-taking' or 'prag'
optimality_alpha_learner = 1.0  # Goodman & Stuhlmuller (2013) fitted sal_alpha = 3.4 to participant data with 4x3 lexicon of three number words ('one', 'two', and 'three') that could be mapped to four different world states (0, 1, 2, and 3)

pragmatic_level_sp_hyp_lr = 'literal'  # The assumption that the learner has about the speaker's pragmatic level


# 1.6: The parameters that determine the learner's hypothesis space:

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
n_contexts = 1000  # The number of contexts that the learner gets to see. If context_generation = 'optimal' n_contexts for n_meanings = 2 can be anything in the table of 4 (e.g. 20, 40, 60, 80, 100, etc.); for n_meanings = 3 anything in the table of 12 (e.g. 12, 36, 60, 84, 108, etc.); and for n_meanings = 4 anything in the table of 48 (e.g. 48, 96, 144, etc.).


# 1.9: The parameters that determine the type and number of simulations that are run:

#FIXME: In the current implementation 'half_ambiguous_lex' can have different instantiations. Therefore, if we run a simulation where there are different lexicon_type_probs, we will want the run_type to be 'population_same_pop' to make sure that all the 'half ambiguous' speakers do have the SAME half ambiguous lexicon.
run_type = 'dyadic' # This parameter determines whether the learner communicates with only one speaker ('dyadic') or with a population ('population_diff_pop' if there are no speakers with the 'half_ambiguous' lexicon type, 'population_same_pop' if there are, 'population_same_pop_dist_learner' if the learner can distinguish between different speakers), or whether we do an iterated learning model ('iter')
n_runs = 100  # The number of runs of the simulation

lex_measure = 'ca'  # This can be set to either 'mi' for mutual information or 'ca' for communicative accuracy (of the lexicon with itself)


run_type_dir = 'Learner_Speaker'


condition_dir = 'results_learning_diff_lex_types_3M_3S_context_gen_'+context_generation+'_p_prior_'+perspective_prior_type[0:4]+'_sp_'+pragmatic_level_speaker+'_a_'+str(optimality_alpha_speaker)[0]+'_lr_'+pragmatic_level_learner+'_a_'+str(optimality_alpha_learner)[0]+'_lr_sp_hyp_'+pragmatic_level_sp_hyp_lr+'_'+lex_measure+'/'



condition_dir = 'Results_learner_speaker_diff_lex_'+context_generation+'_contxts'+'_sp_'+pragmatic_level_speaker+'_sp_a_'+str(optimality_alpha_speaker)[0]+'_sp_p_'+str(speaker_perspective)[0]+'_lr_'+pragmatic_level_learner+'_lr_a_'+str(optimality_alpha_learner)[0]+'_lr_sp_hyp_'+pragmatic_level_sp_hyp_lr+'_p_prior_'+perspective_prior_type[0:4]+'_'+perspective_prior_strength_string+'/'





#######################################################################################################################



def calc_informativity_timecourse_learner(multi_run_log_posterior_matrix, perspective_hyps, informativity_per_lex_hyp):
    # print ''
    # print ''
    # print ''
    # print ''
    # print 'This is the calc_informativity_timecourse_learner() function:'
    sum_informativity_per_context_per_run = np.zeros((multi_run_log_posterior_matrix.shape[0], multi_run_log_posterior_matrix.shape[1]))
    for r in range(multi_run_log_posterior_matrix.shape[0]):
        for c in range(multi_run_log_posterior_matrix.shape[1]):
            log_posterior = multi_run_log_posterior_matrix[r][c]
            log_posterior_split = np.asarray(np.split(log_posterior, len(perspective_hyps)))
            log_posterior_collapsed_on_p_hyps = np.logaddexp(log_posterior_split[0], log_posterior_split[1])
            log_informativity_per_lex_hyp = np.log(informativity_per_lex_hyp)
            log_posteriors_times_informativity = np.add(log_posterior_collapsed_on_p_hyps, log_informativity_per_lex_hyp)
            log_sum_informativity = logsumexp(log_posteriors_times_informativity)
            # print ''
            # print ''
            # print ''
            # print ''
            # print "logsumexp(log_posteriors_times_informativity) is:"
            # print logsumexp(log_posteriors_times_informativity)
            # print ''
            # print "np.nan_to_num(log_sum_informativity) is:"
            # print np.nan_to_num(log_sum_informativity)
            # print ''
            # print "np.exp(np.nan_to_num(log_sum_informativity)) is:"
            # print np.exp(np.nan_to_num(log_sum_informativity))
            sum_informativity_per_context_per_run[r][c] = np.exp(log_sum_informativity)
    mean_informativity_over_time = np.mean(sum_informativity_per_context_per_run, axis=0)
    return mean_informativity_over_time



#######################################################################################################################



if __name__ == "__main__":


    ### CATEGORISING LEXICONS BY INFORMATIVENESS BELOW:

    #
    #
    # print ''
    # print ''
    # # print "lexicon_hyps are:"
    # # print lexicon_hyps
    # print "lexicon_hyps.shape are:"
    # print lexicon_hyps.shape
    #


    informativity_per_lexicon = lex.calc_ca_all_lexicons(lexicon_hyps, error, lex_measure)
    print ''
    print ''
    # print "informativity_per_lexicon is:"
    # print informativity_per_lexicon
    print "informativity_per_lexicon.shape is:"
    print informativity_per_lexicon.shape
    informativity_per_lexicon_rounded = np.round(informativity_per_lexicon, decimals=2)
    print ''
    print ''
    # print "informativity_per_lexicon_rounded is:"
    # print informativity_per_lexicon_rounded
    print "informativity_per_lexicon_rounded.shape is:"
    print informativity_per_lexicon_rounded.shape

    argsort_informativity_per_lexicon = np.argsort(informativity_per_lexicon_rounded)
    # print ''
    # print ''
    # print "argsort_informativity_per_lexicon is:"
    # print argsort_informativity_per_lexicon
    # print "argsort_informativity_per_lexicon.shape is:"
    # print argsort_informativity_per_lexicon.shape


    informativity_per_lexicon_sorted = informativity_per_lexicon_rounded[argsort_informativity_per_lexicon]
    print ''
    print ''
    # print "informativity_per_lexicon_sorted is:"
    # print informativity_per_lexicon_sorted
    print "informativity_per_lexicon_sorted.shape is:"
    print informativity_per_lexicon_sorted.shape


    lexicon_hyps_sorted_by_informativity = lexicon_hyps[argsort_informativity_per_lexicon]
    print ''
    print ''
    print "lexicon_hyps_sorted_by_informativity is:"
    print lexicon_hyps_sorted_by_informativity
    print "lexicon_hyps_sorted_by_informativity.shape is:"
    print lexicon_hyps_sorted_by_informativity.shape


    unique_informativity_per_lexicon = np.unique(informativity_per_lexicon_sorted)
    print ''
    print ''
    # print "unique_informativity_per_lexicon is:"
    # print unique_informativity_per_lexicon
    print "unique_informativity_per_lexicon.shape is:"
    print unique_informativity_per_lexicon.shape


    inf_level_lex_index_unique_dict = {}
    for i in range(len(unique_informativity_per_lexicon)):
        inf_level = unique_informativity_per_lexicon[i]
        lex_index = np.argwhere(informativity_per_lexicon_rounded==inf_level)
        inf_level_lex_index_unique_dict[str(inf_level)] = lex_index[0]

    print ''
    print ''
    print "inf_level_lex_index_unique_dict is:"
    print inf_level_lex_index_unique_dict





    ### CALCULATING LEARNING MEASURES OVER OBSERVATIONS BELOW:


    speaker_lex_type = 'specified_lexicon'  # The lexicon type of the speaker. This can be set to either 'optimal_lex', 'half_ambiguous_lex', 'fully_ambiguous_lex' or 'specified_lexicon'

    inf_level_per_lex_type = np.zeros(len(unique_informativity_per_lexicon))

    # counter = 0
    # key_list = inf_level_lex_index_unique_dict.keys()
    # print "key_list is:"
    # print key_list
    # print "len(key_list) is:"
    # print len(key_list)
    #
    # starting_point = 0
    # end_point = len(key_list)
    #
    #
    # key_list_sliced = key_list[starting_point:end_point]
    # print "key_list_sliced is;"
    # print key_list_sliced
    # print "len(key_list_sliced) is;"
    # print len(key_list_sliced)
    # for key in key_list_sliced:
    # for key in inf_level_lex_index_unique_dict:



    print "max(inf_level_lex_index_unique_dict.values()) is:"
    print max(inf_level_lex_index_unique_dict.values())



    mean_p_hyp_posterior_mass_correct_per_lex_type = np.array([[[np.nan for x in range(n_contexts+1)] for y in range(max(inf_level_lex_index_unique_dict.values()))] for z in range(len(unique_informativity_per_lexicon))])

    mean_lex_hyp_posterior_mass_correct_per_lex_type = np.array([[[np.nan for x in range(n_contexts+1)] for y in range(max(inf_level_lex_index_unique_dict.values()))] for z in range(len(unique_informativity_per_lexicon))])


    mean_comp_hyp_posterior_mass_correct_per_lex_type = np.array([[[np.nan for x in range(n_contexts+1)] for y in range(max(inf_level_lex_index_unique_dict.values()))] for z in range(len(unique_informativity_per_lexicon))])

    median_p_hyp_posterior_mass_correct_per_lex_type = np.array([[[np.nan for x in range(n_contexts+1)] for y in range(max(inf_level_lex_index_unique_dict.values()))] for z in range(len(unique_informativity_per_lexicon))])


    median_lex_hyp_posterior_mass_correct_per_lex_type = np.array([[[np.nan for x in range(n_contexts+1)] for y in range(max(inf_level_lex_index_unique_dict.values()))] for z in range(len(unique_informativity_per_lexicon))])

    median_comp_hyp_posterior_mass_correct_per_lex_type = np.array([[[np.nan for x in range(n_contexts+1)] for y in range(max(inf_level_lex_index_unique_dict.values()))] for z in range(len(unique_informativity_per_lexicon))])


    mean_informativity_timecourse_per_lex_type = np.array([[[np.nan for x in range(n_contexts+1)] for y in range(max(inf_level_lex_index_unique_dict.values()))] for z in range(len(unique_informativity_per_lexicon))])


    counters = [0 for x in range(len(unique_informativity_per_lexicon))]

    for speaker_lex_index in range(len(lexicon_hyps)):
        print ''
        print ''
        print ''
        print ''
        print "speaker_lex_index is:"
        print speaker_lex_index
        speaker_lex = lexicon_hyps[speaker_lex_index]
        print "speaker_lex is:"
        print speaker_lex

        inf_level = informativity_per_lexicon_rounded[speaker_lex_index]
        print "inf_level is:"
        print inf_level

        print "unique_informativity_per_lexicon is:"
        print unique_informativity_per_lexicon

        lex_type_index = np.argwhere(unique_informativity_per_lexicon==inf_level)[0][0]

        print "lex_type_index is:"
        print lex_type_index
        print ''
        print ''

        inf_level_per_lex_type[lex_type_index] = inf_level

        if context_generation == 'random':
            file_title = run_type+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+str(context_generation)+'_err_'+error_string+'_sp_'+pragmatic_level_speaker+'_sp_a_'+str(optimality_alpha_speaker)[0]+'_sp_lex_'+speaker_lex_type[:-4]+'_index_'+str(speaker_lex_index)+'_sp_p_'+str(speaker_perspective)[0]+'_lr_'+pragmatic_level_learner+'_lr_a_'+str(optimality_alpha_learner)[0]+'_lr_sp_hyp_'+pragmatic_level_sp_hyp_lr+'_'+learner_learning_type+'_lex_prior_'+lexicon_prior_type+'_'+lexicon_prior_constant_string+'_p_prior_'+perspective_prior_type+'_'+perspective_prior_strength_string+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U_'+lex_measure

        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            file_title = run_type+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_C'+'_err_'+error_string+'_sp_'+pragmatic_level_speaker+'_sp_a_'+str(optimality_alpha_speaker)[0]+'_sp_lex_'+speaker_lex_type[:-4]+'_index_'+str(speaker_lex_index)+'_sp_p_'+str(speaker_perspective)[0]+'_lr_'+pragmatic_level_learner+'_lr_a_'+str(optimality_alpha_learner)[0]+'_lr_sp_hyp_'+pragmatic_level_sp_hyp_lr+'_'+learner_learning_type+'_lex_prior_'+lexicon_prior_type+'_'+lexicon_prior_constant_string+'_p_prior_'+perspective_prior_type+'_'+perspective_prior_strength_string+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U_'+lex_measure



        filename_results_dict = results_directory + condition_dir + 'Results_' + file_title


        print ''
        print ''
        print "filename_results_dict is:"
        print filename_results_dict

        t10 = time.clock()


        results_dict = pickle.load(open(filename_results_dict+'.p', 'rb'))


        mean_posteriors_over_observations = results_dict['mean_posteriors_over_observations']
        print ''
        print ''
        print "mean_posteriors_over_observations is:"
        print mean_posteriors_over_observations
        print "mean_posteriors_over_observations.shape is:"
        print mean_posteriors_over_observations.shape

        log_mean_posteriors_over_observations = np.log(mean_posteriors_over_observations)
        print ''
        print ''
        print "log_mean_posteriors_over_observations.shape is:"
        print log_mean_posteriors_over_observations.shape


        std_posteriors_over_observations = results_dict['std_posteriors_over_observations']
        print ''
        print ''
        print "mean_posteriors_over_observations is:"
        print mean_posteriors_over_observations
        print "std_posteriors_over_observations.shape is:"
        print std_posteriors_over_observations.shape


        percentiles_posteriors_over_observations = results_dict['percentiles_posteriors_over_observations']
        print ''
        print ''
        print "percentiles_posteriors_over_observations is:"
        print percentiles_posteriors_over_observations
        print "percentiles_posteriors_over_observations.shape is:"
        print percentiles_posteriors_over_observations.shape

        median_posteriors_over_observations = percentiles_posteriors_over_observations[1]
        print ''
        print ''
        print "median_posteriors_over_observations is:"
        print median_posteriors_over_observations
        print "median_posteriors_over_observations.shape is:"
        print median_posteriors_over_observations.shape

        log_median_posteriors_over_observations = np.log(median_posteriors_over_observations)
        print ''
        print ''
        print "log_median_posteriors_over_observations.shape is:"
        print log_median_posteriors_over_observations.shape


        correct_p_hyp_indices = results_dict['correct_p_hyp_indices']
        print ''
        print ''
        # print "correct_p_hyp_indices is:"
        # print correct_p_hyp_indices
        print "len(correct_p_hyp_indices) is:"
        print len(correct_p_hyp_indices)


        correct_lex_hyp_indices = results_dict['correct_lex_hyp_indices']
        print ''
        print ''
        # print "correct_lex_hyp_indices is:"
        # print correct_lex_hyp_indices
        print "len(correct_lex_hyp_indices) is:"
        print len(correct_lex_hyp_indices)


        correct_composite_hyp_indices = results_dict['correct_composite_hyp_indices']
        print ''
        print ''
        # print "correct_composite_hyp_indices is:"
        # print correct_composite_hyp_indices
        print "len(correct_composite_hyp_indices) is:"
        print len(correct_composite_hyp_indices)

        speaker_lexicon = results_dict['speaker_lexicon']
        print ''
        print ''
        print "speaker_lexicon is:"
        print speaker_lexicon



        print ''
        print ''
        # print "log_mean_posteriors_over_observations[:, correct_p_hyp_indices] is:"
        # print log_mean_posteriors_over_observations[:, correct_p_hyp_indices]
        print "log_mean_posteriors_over_observations[:, correct_p_hyp_indices].shape is:"
        print log_mean_posteriors_over_observations[:, correct_p_hyp_indices].shape

        log_mean_p_hyp_posterior_mass_correct = logsumexp(log_mean_posteriors_over_observations[:, correct_p_hyp_indices], axis=1)
        # print "log_mean_p_hyp_posterior_mass_correct is:"
        # print log_mean_p_hyp_posterior_mass_correct
        print "log_mean_p_hyp_posterior_mass_correct.shape is:"
        print log_mean_p_hyp_posterior_mass_correct.shape

        print "counters[lex_type_index] is:"
        print counters[lex_type_index]

        mean_p_hyp_posterior_mass_correct_per_lex_type[lex_type_index][counters[lex_type_index]] = np.exp(log_mean_p_hyp_posterior_mass_correct)


        print ''
        print ''
        # print "log_median_posteriors_over_observations[:, correct_p_hyp_indices] is:"
        # print log_median_posteriors_over_observations[:, correct_p_hyp_indices]
        print "log_median_posteriors_over_observations[:, correct_p_hyp_indices].shape is:"
        print log_median_posteriors_over_observations[:, correct_p_hyp_indices].shape

        log_median_p_hyp_posterior_mass_correct = logsumexp(log_median_posteriors_over_observations[:, correct_p_hyp_indices], axis=1)
        # print "log_median_p_hyp_posterior_mass_correct is:"
        # print log_median_p_hyp_posterior_mass_correct
        print "log_median_p_hyp_posterior_mass_correct.shape is:"
        print log_median_p_hyp_posterior_mass_correct.shape

        median_p_hyp_posterior_mass_correct_per_lex_type[lex_type_index][counters[lex_type_index]] = np.exp(log_median_p_hyp_posterior_mass_correct)




        print ''
        print ''
        # print "mean_posteriors_over_observations[:, correct_lex_hyp_indices] is:"
        # print mean_posteriors_over_observations[:, correct_lex_hyp_indices]
        print "mean_posteriors_over_observations[:, correct_lex_hyp_indices].shape is:"
        print mean_posteriors_over_observations[:, correct_lex_hyp_indices].shape

        mean_lex_hyp_posterior_mass_correct = np.sum(mean_posteriors_over_observations[:, correct_lex_hyp_indices], axis=1)
        # print "mean_lex_hyp_posterior_mass_correct is:"
        # print mean_lex_hyp_posterior_mass_correct
        print "mean_lex_hyp_posterior_mass_correct.shape is:"
        print mean_lex_hyp_posterior_mass_correct.shape


        mean_lex_hyp_posterior_mass_correct_per_lex_type[lex_type_index][counters[lex_type_index]] = mean_lex_hyp_posterior_mass_correct


        print ''
        print ''
        # print "percentiles_posteriors_over_observations[1][:, correct_lex_hyp_indices] is:"
        # print percentiles_posteriors_over_observations[1][:, correct_lex_hyp_indices]
        print "percentiles_posteriors_over_observations[1][:, correct_lex_hyp_indices].shape is:"
        print percentiles_posteriors_over_observations[1][:, correct_lex_hyp_indices].shape


        median_lex_hyp_posterior_mass_correct = np.sum(percentiles_posteriors_over_observations[1][:, correct_lex_hyp_indices], axis=1)
        # print "median_lex_hyp_posterior_mass_correct is:"
        # print median_lex_hyp_posterior_mass_correct
        print "median_lex_hyp_posterior_mass_correct.shape is:"
        print median_lex_hyp_posterior_mass_correct.shape

        median_lex_hyp_posterior_mass_correct_per_lex_type[lex_type_index][counters[lex_type_index]] = median_lex_hyp_posterior_mass_correct




        print ''
        print ''
        # print "mean_posteriors_over_observations[:, correct_composite_hyp_indices] is:"
        # print mean_posteriors_over_observations[:, correct_composite_hyp_indices]
        print "mean_posteriors_over_observations[:, correct_composite_hyp_indices].shape is:"
        print mean_posteriors_over_observations[:, correct_composite_hyp_indices].shape

        mean_comp_hyp_posterior_mass_correct = np.sum(mean_posteriors_over_observations[:, correct_composite_hyp_indices], axis=1)
        # print "mean_comp_hyp_posterior_mass_correct is:"
        # print mean_comp_hyp_posterior_mass_correct
        print "mean_comp_hyp_posterior_mass_correct.shape is:"
        print mean_comp_hyp_posterior_mass_correct.shape


        mean_comp_hyp_posterior_mass_correct_per_lex_type[lex_type_index][counters[lex_type_index]] = mean_comp_hyp_posterior_mass_correct


        print ''
        print ''
        # print "percentiles_posteriors_over_observations[1][:, correct_composite_hyp_indices] is:"
        # print percentiles_posteriors_over_observations[1][:, correct_composite_hyp_indices]
        print "percentiles_posteriors_over_observations[1][:, correct_composite_hyp_indices].shape is:"
        print percentiles_posteriors_over_observations[1][:, correct_composite_hyp_indices].shape


        median_comp_hyp_posterior_mass_correct = np.sum(percentiles_posteriors_over_observations[1][:, correct_composite_hyp_indices], axis=1)
        # print "median_comp_hyp_posterior_mass_correct is:"
        # print median_comp_hyp_posterior_mass_correct
        print "median_comp_hyp_posterior_mass_correct.shape is:"
        print median_comp_hyp_posterior_mass_correct.shape

        median_comp_hyp_posterior_mass_correct_per_lex_type[lex_type_index][counters[lex_type_index]] = median_comp_hyp_posterior_mass_correct



        mean_informativity_timecourse = calc_informativity_timecourse_learner(np.array([np.log(mean_posteriors_over_observations)]), perspective_hyps, informativity_per_lexicon)
        print ''
        print ''
        print "mean_informativity_timecourse is:"
        print mean_informativity_timecourse
        print "mean_informativity_timecourse.shape is:"
        print mean_informativity_timecourse.shape

        mean_informativity_timecourse_per_lex_type[lex_type_index][counters[lex_type_index]] = mean_informativity_timecourse


        counters[lex_type_index] += 1



    print ''
    print ''
    print ''
    print ''
    print "inf_level_per_lex_type is:"
    print inf_level_per_lex_type
    print "inf_level_per_lex_type.shape is:"
    print inf_level_per_lex_type.shape


    print ''
    print ''
    print ''
    print ''
    print "mean_p_hyp_posterior_mass_correct_per_lex_type is:"
    print mean_p_hyp_posterior_mass_correct_per_lex_type
    print "mean_p_hyp_posterior_mass_correct_per_lex_type.shape is:"
    print mean_p_hyp_posterior_mass_correct_per_lex_type.shape

    print ''
    print ''
    print "median_p_hyp_posterior_mass_correct_per_lex_type is:"
    print median_p_hyp_posterior_mass_correct_per_lex_type
    print "median_p_hyp_posterior_mass_correct_per_lex_type.shape is:"
    print median_p_hyp_posterior_mass_correct_per_lex_type.shape




    print ''
    print ''
    print ''
    print ''
    print "mean_lex_hyp_posterior_mass_correct_per_lex_type is:"
    print mean_lex_hyp_posterior_mass_correct_per_lex_type
    print "mean_lex_hyp_posterior_mass_correct_per_lex_type.shape is:"
    print mean_lex_hyp_posterior_mass_correct_per_lex_type.shape

    print ''
    print ''
    print "median_lex_hyp_posterior_mass_correct_per_lex_type is:"
    print median_lex_hyp_posterior_mass_correct_per_lex_type
    print "median_lex_hyp_posterior_mass_correct_per_lex_type.shape is:"
    print median_lex_hyp_posterior_mass_correct_per_lex_type.shape



    print ''
    print ''
    print ''
    print ''
    print "mean_comp_hyp_posterior_mass_correct_per_lex_type is:"
    print mean_comp_hyp_posterior_mass_correct_per_lex_type
    print "mean_comp_hyp_posterior_mass_correct_per_lex_type.shape is:"
    print mean_comp_hyp_posterior_mass_correct_per_lex_type.shape

    print ''
    print ''
    print "median_comp_hyp_posterior_mass_correct_per_lex_type is:"
    print median_comp_hyp_posterior_mass_correct_per_lex_type
    print "median_comp_hyp_posterior_mass_correct_per_lex_type.shape is:"
    print median_comp_hyp_posterior_mass_correct_per_lex_type.shape



    print ''
    print ''
    print "mean_informativity_timecourse_per_lex_type is:"
    print mean_informativity_timecourse_per_lex_type
    print "mean_informativity_timecourse_per_lex_type.shape is:"
    print mean_informativity_timecourse_per_lex_type.shape



    grand_mean_p_hyp_posterior_mass_correct_per_lex_type = np.nanmean(mean_p_hyp_posterior_mass_correct_per_lex_type, axis=1)
    print "grand_mean_p_hyp_posterior_mass_correct_per_lex_type is"
    print grand_mean_p_hyp_posterior_mass_correct_per_lex_type
    print "grand_mean_p_hyp_posterior_mass_correct_per_lex_type.shape is"
    print grand_mean_p_hyp_posterior_mass_correct_per_lex_type.shape


    # TODO: Does it make sense to take the mean of the median??
    grand_median_p_hyp_posterior_mass_correct_per_lex_type = np.nanmean(median_p_hyp_posterior_mass_correct_per_lex_type, axis=1)
    print "grand_median_p_hyp_posterior_mass_correct_per_lex_type is"
    print grand_median_p_hyp_posterior_mass_correct_per_lex_type
    print "grand_median_p_hyp_posterior_mass_correct_per_lex_type.shape is"
    print grand_median_p_hyp_posterior_mass_correct_per_lex_type.shape



    grand_mean_lex_hyp_posterior_mass_correct_per_lex_type = np.nanmean(mean_lex_hyp_posterior_mass_correct_per_lex_type, axis=1)
    print "grand_mean_lex_hyp_posterior_mass_correct_per_lex_type is"
    print grand_mean_lex_hyp_posterior_mass_correct_per_lex_type
    print "grand_mean_lex_hyp_posterior_mass_correct_per_lex_type.shape is"
    print grand_mean_lex_hyp_posterior_mass_correct_per_lex_type.shape

    # TODO: Does it make sense to take the mean of the median??
    grand_median_lex_hyp_posterior_mass_correct_per_lex_type = np.nanmean(median_lex_hyp_posterior_mass_correct_per_lex_type, axis=1)
    print "grand_median_lex_hyp_posterior_mass_correct_per_lex_type is"
    print grand_median_lex_hyp_posterior_mass_correct_per_lex_type
    print "grand_median_lex_hyp_posterior_mass_correct_per_lex_type.shape is"
    print grand_median_lex_hyp_posterior_mass_correct_per_lex_type.shape



    grand_mean_comp_hyp_posterior_mass_correct_per_lex_type = np.nanmean(mean_comp_hyp_posterior_mass_correct_per_lex_type, axis=1)
    print "grand_mean_comp_hyp_posterior_mass_correct_per_lex_type is"
    print grand_mean_comp_hyp_posterior_mass_correct_per_lex_type
    print "grand_mean_comp_hyp_posterior_mass_correct_per_lex_type.shape is"
    print grand_mean_comp_hyp_posterior_mass_correct_per_lex_type.shape

    # TODO: Does it make sense to take the mean of the median??
    grand_median_comp_hyp_posterior_mass_correct_per_lex_type = np.nanmean(median_comp_hyp_posterior_mass_correct_per_lex_type, axis=1)
    print "grand_median_comp_hyp_posterior_mass_correct_per_lex_type is"
    print grand_median_comp_hyp_posterior_mass_correct_per_lex_type
    print "grand_median_comp_hyp_posterior_mass_correct_per_lex_type.shape is"
    print grand_median_comp_hyp_posterior_mass_correct_per_lex_type.shape



    grand_mean_informativity_timecourse_per_lex_type = np.nanmean(mean_informativity_timecourse_per_lex_type, axis=1)
    print "grand_mean_informativity_timecourse_per_lex_type is"
    print grand_mean_informativity_timecourse_per_lex_type
    print "grand_mean_informativity_timecourse_per_lex_type.shape is"
    print grand_mean_informativity_timecourse_per_lex_type.shape




    grand_means_file_title = str(n_contexts)+'_C_'+str(n_runs)+'_R_'+str(int(n_meanings))+'M_'+str(int(n_signals))+'S_'+str(context_generation)+'_err_'+error_string+'_sp_'+pragmatic_level_speaker+'_a_'+str(optimality_alpha_speaker)[0]+'_lr_'+pragmatic_level_learner+'_a_'+str(optimality_alpha_learner)[0]+'_lr_sp_hyp_'+pragmatic_level_sp_hyp_lr+'_sp_p_'+str(speaker_perspective)[0]+'_p_prior_'+perspective_prior_type[:4]+'_'+perspective_prior_strength_string+'_'+lex_measure

    pickle_file_path = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/Learner_Speaker/'

    pickle.dump(grand_mean_p_hyp_posterior_mass_correct_per_lex_type, open(pickle_file_path+'grand_mean_p_hyp_posterior_mass_correct_diff_lexicons_'+grand_means_file_title+'.p', 'wb'))

    pickle.dump(grand_median_p_hyp_posterior_mass_correct_per_lex_type, open(pickle_file_path+'grand_median_p_hyp_posterior_mass_correct_diff_lexicons_'+grand_means_file_title+'.p', 'wb'))


    pickle.dump(grand_mean_lex_hyp_posterior_mass_correct_per_lex_type, open(pickle_file_path+'grand_mean_lex_hyp_posterior_mass_correct_diff_lexicons_'+grand_means_file_title+'.p', 'wb'))

    pickle.dump(grand_median_lex_hyp_posterior_mass_correct_per_lex_type, open(pickle_file_path+'grand_median_lex_hyp_posterior_mass_correct_diff_lexicons_'+grand_means_file_title+'.p', 'wb'))


    pickle.dump(grand_mean_comp_hyp_posterior_mass_correct_per_lex_type, open(pickle_file_path+'grand_mean_comp_hyp_posterior_mass_correct_diff_lexicons_'+grand_means_file_title+'.p', 'wb'))

    pickle.dump(grand_median_comp_hyp_posterior_mass_correct_per_lex_type, open(pickle_file_path+'grand_median_comp_hyp_posterior_mass_correct_diff_lexicons_'+grand_means_file_title+'.p', 'wb'))



    pickle.dump(grand_mean_informativity_timecourse_per_lex_type, open(pickle_file_path+'grand_mean_informativity_timecourse_diff_lexicons_'+grand_means_file_title+'.p', 'wb'))
