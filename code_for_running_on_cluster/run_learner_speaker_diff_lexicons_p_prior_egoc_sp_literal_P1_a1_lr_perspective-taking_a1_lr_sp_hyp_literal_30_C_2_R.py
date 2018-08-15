__author__ = 'Marieke Woensdregt'

import time
import numpy as np
from run_sim_learner_speaker_for_cluster import multi_runs_dyadic
import hypspace
import saveresults
import measur
import sys



# Setting the speaker_lex_index parameter based on the command-line input:
speaker_lex_index = int(sys.argv[1])-1 #NOTE: first argument in sys.argv list is always the name of the script





#######################################################################################################################
# 1. THE PARAMETERS:

# 1.1: Set the path to the results_directory on your cluster account where you want the results to be stored:
results_directory = '/exports/eddie/scratch/s1370641/'



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
n_contexts = 30  # The number of contexts that the learner gets to see. If context_generation = 'optimal' n_contexts for n_meanings = 2 can be anything in the table of 4 (e.g. 20, 40, 60, 80, 100, etc.); for n_meanings = 3 anything in the table of 12 (e.g. 12, 36, 60, 84, 108, etc.); and for n_meanings = 4 anything in the table of 48 (e.g. 48, 96, 144, etc.).


# 1.9: The parameters that determine the type and number of simulations that are run:

#FIXME: In the current implementation 'half_ambiguous_lex' can have different instantiations. Therefore, if we run a simulation where there are different lexicon_type_probs, we will want the run_type to be 'population_same_pop' to make sure that all the 'half ambiguous' speakers do have the SAME half ambiguous lexicon.
run_type = 'dyadic' # This parameter determines whether the learner communicates with only one speaker ('dyadic') or with a population ('population_diff_pop' if there are no speakers with the 'half_ambiguous' lexicon type, 'population_same_pop' if there are, 'population_same_pop_dist_learner' if the learner can distinguish between different speakers), or whether we do an iterated learning model ('iter')
n_runs = 2  # The number of runs of the simulation
report_every_r = 10


lex_measure = 'ca'  # This can be set to either 'mi' for mutual information or 'ca' for communicative accuracy (of the lexicon with itself)

#######################################################################################################################






if __name__ == "__main__":


    if n_runs > 0 and run_type == 'dyadic':

        t0 = time.clock()

        all_results_dict = multi_runs_dyadic(n_meanings, n_signals, n_runs, n_contexts, n_utterances, context_generation, context_type, context_size, helpful_contexts, speaker_lex_type, speaker_lex_index, error, pragmatic_level_speaker, optimality_alpha_speaker, pragmatic_level_learner, optimality_alpha_learner, speaker_perspective, sal_alpha, speaker_learning_type, learner_perspective, learner_lex_type, learner_learning_type, hypothesis_space, perspective_hyps, lexicon_hyps, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant)


        run_simulation_time = time.clock()-t0
        print 
        print 'run_simulation_time is:'
        print str((run_simulation_time/60))+" m"



        multi_run_log_posterior_matrix = all_results_dict['multi_run_log_posterior_matrix']
        # print 
        # print 
        # print "multi_run_log_posterior_matrix is:"
        # print multi_run_log_posterior_matrix
        # print "multi_run_log_posterior_matrix.shape is:"
        # print multi_run_log_posterior_matrix.shape



        mean_final_posteriors = measur.calc_mean_of_final_posteriors(multi_run_log_posterior_matrix)
        # print 
        # print 
        # print "mean_final_posteriors is:"
        # print mean_final_posteriors


        std_final_posteriors = measur.calc_std_of_final_posteriors(multi_run_log_posterior_matrix)
        # print 
        # print 
        # print "std_final_posteriors is:"
        # print std_final_posteriors


        mean_std_final_posteriors_dict = {'mean_final_posteriors':mean_final_posteriors,
                                          'std_final_posteriors':std_final_posteriors}


    #############################################################################
    # Below the results are written to pickle files:




        if context_generation == 'random':
            file_title = run_type+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+str(context_generation)+'_err_'+error_string+'_sp_'+pragmatic_level_speaker+'_sp_a_'+str(optimality_alpha_speaker)[0]+'_sp_lex_'+speaker_lex_type[:-4]+'_index_'+str(speaker_lex_index)+'_sp_p_'+str(speaker_perspective)[0]+'_lr_'+pragmatic_level_learner+'_lr_a_'+str(optimality_alpha_learner)[0]+'_lr_sp_hyp_'+pragmatic_level_sp_hyp_lr+'_'+learner_learning_type+'_lex_prior_'+lexicon_prior_type+'_'+lexicon_prior_constant_string+'_p_prior_'+perspective_prior_type+'_'+perspective_prior_strength_string+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U_'+lex_measure

        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            file_title = run_type+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_C'+'_err_'+error_string+'_sp_'+pragmatic_level_speaker+'_sp_a_'+str(optimality_alpha_speaker)[0]+'_sp_lex_'+speaker_lex_type[:-4]+'_index_'+str(speaker_lex_index)+'_sp_p_'+str(speaker_perspective)[0]+'_lr_'+pragmatic_level_learner+'_lr_a_'+str(optimality_alpha_learner)[0]+'_lr_sp_hyp_'+pragmatic_level_sp_hyp_lr+'_'+learner_learning_type+'_lex_prior_'+lexicon_prior_type+'_'+lexicon_prior_constant_string+'_p_prior_'+perspective_prior_type+'_'+perspective_prior_strength_string+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U_'+lex_measure


        pickle_file_title_all_results = results_directory + '/Results_' + file_title

        saveresults.write_results_to_pickle_file(pickle_file_title_all_results, all_results_dict)


        pickle_file_title_mean_std_final_posteriors = results_directory + '/Mean_Std_Final_Post_' + file_title

        saveresults.write_results_to_pickle_file(pickle_file_title_mean_std_final_posteriors, mean_std_final_posteriors_dict)



