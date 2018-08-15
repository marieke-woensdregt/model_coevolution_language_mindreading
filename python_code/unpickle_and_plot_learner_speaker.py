__author__ = 'Marieke Woensdregt'


import numpy as np
from hypspace import create_all_lexicons
from hypspace import create_all_optimal_lexicons
from hypspace import remove_subset_of_signals_lexicons
from hypspace import list_hypothesis_space
from saveresults import convert_array_to_string
import pickle
import plots
import measur
import time



#######################################################################################################################
# STEP 3: THE PARAMETERS:


# 2.1: The parameters defining the lexicon size (and thus the number of meanings in the world):

n_meanings = 2 # The number of meanings
n_signals = 2 # The number of signals



# 2.2: The parameters defining the contexts and how they map to the agent's saliencies:

context_generation = 'most_optimal' # This can be set to either 'random', 'only_helpful', 'most_optimal'
#helpful_contexts = np.array([[0.1, 0.7], [0.3, 0.9], [0.1, 0.6], [0.4, 0.9], [0.1, 0.8], [0.2, 0.9], [0.1, 0.5], [0.5, 0.9], [0.1, 0.4], [0.6, 0.9], [0.7, 0.1], [0.9, 0.3], [0.6, 0.1], [0.9, 0.4], [0.8, 0.1], [0.9, 0.2], [0.5, 0.1], [0.9, 0.5], [0.4, 0.1], [0.9, 0.6]])  # This is a fixed collection of the 20 most helpful contexts (in which the ratio of meaning probability for the one perspective is maximally different from that for the other perspective).
helpful_contexts = np.array([[0.7, 0.1], [0.9, 0.3], [0.1, 0.7], [0.3, 0.9]])


context_type = 'continuous' # This can be set to either 'absolute' for meanings being in/out or 'continuous' for all meanings being present (but in different positions)
context_size = 1 # This parameter is only used if the context_type is 'absolute' and determines the number of meanings present

alpha = 1. # The exponent that is used by the saliency function. sal_alpha=1 gives a directly proportional relationship between meaning distances and meaning saliencies, with sal_alpha>1 the saliency of meanings goes down exponentially as their distance from the agent increases (and with sal_alpha<1 the other way around)



error = 0.05 # The error term on production
error_string = convert_array_to_string(error)




# 2.4: The parameters that determine the make-up of an individual speaker (for the dyadic condition):

speaker_perspective = 1. # The speaker's perspective
speaker_lex_type = 'fully_ambiguous_lex' # The lexicon type of the speaker. This can be set to either 'optimal_lex', 'half_ambiguous_lex', 'fully_ambiguous_lex' or 'specified_lexicon'
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



lexicon_prior_type = 'expressivity_bias' # This can be set to either 'neutral', 'ambiguous_fixed', 'half_ambiguous_fixed', 'expressivity_bias' or 'compressibility_bias'.
lexicon_prior_constant = 0.3 # Determines the strength of the lexicon prior, with small c creating a STRONG prior and large c creating a WEAK prior. (And with c = 1000 creating an almost uniform prior.)
lexicon_prior_constant_string = convert_array_to_string(lexicon_prior_constant)






# 2.5: The parameters that determine the amount of data_dict that the learner gets to see, and the amount of runs of the simulation:

n_utterances = 1 # This parameter determines how many signals the learner gets to observe in each context
n_contexts = 500 # The number of contexts that the learner gets to see.



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
run_type = 'dyadic' # This parameter determines whether the learner communicates with only one speaker ('dyadic') or with a population ('population_diff_pop' if there are no speakers with the 'half_ambiguous' lexicon type, 'population_same_pop' if there are, 'population_same_pop_dist_learner' if the learner can distinguish between different speakers), or whether we do an iterated learning model ('iteration')
n_runs = 1000 # The number of runs of the simulation
report_every_r = 100

which_hyps_on_graph = 'all_hyps' # This is used in the plot_post_probs_over_lex_hyps() function, and can be set to either 'all_hyps' or 'lex_hyps_only'
x_axis_steps = 50 # This determines the space between the x-ticks on the graph



#######################################################################################################################



new_hyp_order_handsorted_on_lexicons = np.array([1, 3, 2, 5, 6, 7, 0, 4, 8, 10, 12, 11, 14, 15, 16, 9, 13, 17])


print 
print 
print "lexicon_hyps are:"
print lexicon_hyps


lexicon_hyps_sorted = lexicon_hyps[new_hyp_order_handsorted_on_lexicons[:len(lexicon_hyps)]]

print 
print 
print "lexicon_hyps_sorted are:"
print lexicon_hyps_sorted




run_type_dir = 'Learner_Speaker'



if context_generation == 'random':
    file_title = run_type+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+str(context_generation)+'_err_'+error_string+'_sp_lex_'+speaker_lex_type[:-4]+'_index_'+str(speaker_lex_index)+'_sp_p_'+str(speaker_perspective)[0]+'_'+learner_learning_type+'_lex_prior_'+lexicon_prior_type+'_'+lexicon_prior_constant_string+'_p_prior_'+perspective_prior_type+'_'+perspective_prior_strength_string+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'

elif context_generation == 'only_helpful' or context_generation == 'most_optimal':
    file_title = run_type+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_C'+'_err_'+error_string+'_sp_lex_'+speaker_lex_type[:-4]+'_index_'+str(speaker_lex_index)+'_sp_p_'+str(speaker_perspective)[0]+'_'+learner_learning_type+'_lex_prior_'+lexicon_prior_type+'_'+lexicon_prior_constant_string+'_p_prior_'+perspective_prior_type+'_'+perspective_prior_strength_string+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'



pickle_file_title_all_results = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/'+run_type_dir+'/Results_'+file_title


pickle_file_title_mean_std_final_posteriors = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/'+run_type_dir+'/Mean_Std_Final_Post_'+file_title



pickle_file_title_correct_hyp_posterior_mass_percentiles = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/' + run_type_dir + '/Correct_Hyp_Post_Mass_Percentiles' + file_title


pickle_file_title_hypothesis_percentiles = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/' + run_type_dir + '/Hyp_Percentiles' + file_title


pickle_file_title_convergence_time_min_max = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/'+run_type_dir+'/Convergence_Time_Percentiles'+file_title


pickle_file_title_convergence_time_percentiles = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/'+run_type_dir+'/Convergence_Time_Percentiles'+file_title




all_results_dict = pickle.load(open(pickle_file_title_all_results + '.p', 'rb'))

mean_std_final_posteriors_dict = pickle.load(open(pickle_file_title_mean_std_final_posteriors + '.p', 'rb'))

percentiles_correct_hyp_posterior_mass_dict = pickle.load(open(pickle_file_title_correct_hyp_posterior_mass_percentiles + '.p', 'rb'))

hypotheses_percentiles = pickle.load(open(pickle_file_title_hypothesis_percentiles + '.p', 'rb'))

convergence_time_min_max = pickle.load(open(pickle_file_title_convergence_time_min_max + '.p', 'rb'))

convergence_time_percentiles = pickle.load(open(pickle_file_title_convergence_time_percentiles + '.p', 'rb'))


multi_run_log_posterior_matrix = all_results_dict['multi_run_log_posterior_matrix']



mean_final_posteriors = mean_std_final_posteriors_dict['mean_final_posteriors']

std_final_posteriors = mean_std_final_posteriors_dict['std_final_posteriors']




correct_p_hyp_indices = all_results_dict['correct_p_hyp_indices']
print 
print 
print "correct_p_hyp_indices is:"
print correct_p_hyp_indices



correct_lex_hyp_indices = all_results_dict['correct_lex_hyp_indices']
print 
print 
print "correct_lex_hyp_indices is:"
print correct_lex_hyp_indices


correct_composite_hyp_indices = all_results_dict['correct_composite_hyp_indices']
print 
print 
print "correct_composite_hyp_indices is:"
print correct_composite_hyp_indices


speaker_lexicon = all_results_dict['speaker_lexicon']
# print 
# print 
# print "speaker_lexicon is:"
# print speaker_lexicon



correct_hypothesis_index, mirror_hypothesis_index = measur.find_correct_and_mirror_hyp_index(speaker_lexicon)
# print 
# print 
# print "correct_hypothesis_index is:"
# print correct_hypothesis_index
# print "mirror_hypothesis_index is:"
# print mirror_hypothesis_index





percentiles_p_hyp_posterior_mass_correct = percentiles_correct_hyp_posterior_mass_dict['percentiles_p_hyp_posterior_mass_correct']

percentiles_lex_hyp_posterior_mass_correct = percentiles_correct_hyp_posterior_mass_dict['percentiles_lex_hyp_posterior_mass_correct']

percentiles_composite_hyp_posterior_mass_correct = percentiles_correct_hyp_posterior_mass_dict['percentiles_composite_hyp_posterior_mass_correct']




#
# correct_composite_hyp_indices = unpickle_and_plot_new.unpickle(file_title, run_type_dir, 'correct_composite_hyp_indices')
#
#
# speaker_lexicon = unpickle_and_plot_new.unpickle(file_title, run_type_dir, 'speaker_lexicon')
#
# mirror_hypothesis_index = measur.find_correct_and_mirror_hyp_index(speaker_lexicon)






#############################################################################
# The code below makes the plots and saves them:


if n_runs > 0 and run_type == 'dyadic':
    t3 = time.clock()

    plot_file_path = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Plots/'+run_type_dir

    plot_file_title = file_title


    scores_plot_title = 'Posterior probability assigned to the correct hypothesis over time'


    plots.plot_timecourse_scores_percentiles_med_perc_one_hyp_type(scores_plot_title, plot_file_path, plot_file_title, ((n_contexts * n_utterances) + 1), percentiles_composite_hyp_posterior_mass_correct, percentiles_p_hyp_posterior_mass_correct, percentiles_lex_hyp_posterior_mass_correct, 'composite', x_axis_steps)

    #
    # multi_run_final_log_posteriors_matrix = multi_run_log_posterior_matrix[:,-1,:]
    #
    #
    # p_over_hyps_plot_title = 'Prior distribution for egocentric perspective + expressive lexicon prior'
    #
    # p_over_hyps_plot_x_label = 'Composite hypotheses'
    #
    # p_over_hyps_plot_y_label = 'Prior probability'
    #
    # plots.plot_posterior_over_hyps('multi_run', n_contexts, 0, plot_file_path, '/Posterior_over_l_hyps_'+plot_file_title, p_over_hyps_plot_title, p_over_hyps_plot_x_label, p_over_hyps_plot_y_label, hypothesis_space, perspective_hyps, lexicon_hyps, multi_run_final_log_posteriors_matrix, which_lexicon_hyps, which_hyps_on_graph, std='no')
    #


    scores_plot_title = 'Posterior probability assigned to the correct (part) hypothesis over time'


    plots.plot_timecourse_scores_percentiles(scores_plot_title, plot_file_path, plot_file_title, ((n_contexts*n_utterances)+1), percentiles_composite_hyp_posterior_mass_correct, percentiles_p_hyp_posterior_mass_correct, percentiles_lex_hyp_posterior_mass_correct)


    hypotheses_plot_title = 'Posterior probability assigned to each composite hypothesis over time'

    plots.plot_timecourse_hypotheses_percentiles(hypotheses_plot_title, plot_file_path, plot_file_title, ((n_contexts*n_utterances)+1), hypotheses_percentiles, correct_hypothesis_index, mirror_hypothesis_index)


    #
    # plots.plot_timecourse_scores_percentiles_without_error_median(scores_plot_title, plot_file_path, plot_file_title, ((n_contexts*n_utterances)+1), percentiles_composite_hyp_posterior_mass_correct, percentiles_p_hyp_posterior_mass_correct, percentiles_lex_hyp_posterior_mass_correct)
    #
    #
    # plots.plot_timecourse_scores_percentiles_without_error_mean(scores_plot_title, plot_file_path, plot_file_title, ((n_contexts*n_utterances)+1), percentiles_composite_hyp_posterior_mass_correct, percentiles_p_hyp_posterior_mass_correct, percentiles_lex_hyp_posterior_mass_correct)
    #

    #
    # lex_heatmap_title = 'Heatmap of posterior probability distribution over m-s mappings'
    #
    # plots.plot_lexicon_heatmap(lex_heatmap_title, plot_file_path, plot_file_title, lex_posterior_matrix)
    #
    #
    #
    # convergence_time_plot_title = 'No. of observations required to reach 1.0-theta posterior on correct hypothesis'
    #
    # plots.plot_convergence_time_over_theta_range(convergence_time_plot_title, plot_file_path, plot_file_title, theta_range, percentiles_convergence_time_over_theta_composite, percentiles_convergence_time_over_theta_perspective, percentiles_convergence_time_over_theta_lexicon)
    #




    plotting_time = time.clock()-t3
    print 
    print 'plotting_time is:'
    print str((plotting_time/60))+" m"
