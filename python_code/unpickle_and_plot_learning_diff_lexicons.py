__author__ = 'Marieke Woensdregt'

import numpy as np
from scipy.misc import logsumexp
import itertools
import lex
from hypspace import create_all_lexicons
from hypspace import create_all_optimal_lexicons
from hypspace import remove_subset_of_signals_lexicons
from hypspace import list_hypothesis_space
from saveresults import convert_array_to_string, convert_float_value_to_string
import time
import pickle
import matplotlib.pyplot as plt
import seaborn as sns






#######################################################################################################################
# STEP 3: THE PARAMETERS:


# 2.1: The parameters defining the lexicon size (and thus the number of meanings in the world):

n_meanings = 3  # The number of meanings
n_signals = 3  # The number of signals



# 2.2: The parameters defining the contexts and how they map to the agent's saliencies:

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


context_type = 'continuous' # This can be set to either 'absolute' for meanings being in/out or 'continuous' for all meanings being present (but in different positions)
context_size = 1 # This parameter is only used if the context_type is 'absolute' and determines the number of meanings present

sal_alpha = 1. # The exponent that is used by the saliency function. sal_alpha=1 gives a directly proportional relationship between meaning distances and meaning saliencies, with sal_alpha>1 the saliency of meanings goes down exponentially as their distance from the agent increases (and with sal_alpha<1 the other way around)



error = 0.05 # The error term on production
error_string = convert_array_to_string(error)




# 2.4: The parameters that determine the make-up of an individual speaker (for the dyadic condition):

speaker_perspective = 1. # The speaker's perspective
speaker_lex_type = 'specified_lexicon' # The lexicon type of the speaker. This can be set to either 'optimal_lex', 'half_ambiguous_lex', 'fully_ambiguous_lex' or 'specified_lexicon'
speaker_lex_index = 0
speaker_learning_type = 'sample' #FIXME: The speaker has to be initiated with a learning type because I have not yet coded up a subclass of Agent that is only Speaker (for which things like hypothesis space, prior distributiona and learning type would not have to be specified).



# 2.5: The parameters that determine the attributes of the learner:

learner_perspective = 0.  # The learner's perspective
learner_lex_type = 'empty_lex'  # The lexicon type of the learner. This will normally be 'empty_lex'
learner_learning_type = 'sample'  # The type of learning that the learner does. This can be set to either 'map' or 'sample'



pragmatic_level_speaker = 'prag'  # This can be set to either 'literal', 'perspective-taking' or 'prag'
optimality_alpha_speaker = 3.0  # Goodman & Stuhlmuller (2013) fitted sal_alpha = 3.4 to participant data with 4x3 lexicon of three number words ('one', 'two', and 'three') that could be mapped to four different world states (0, 1, 2, and 3)


pragmatic_level_learner = 'prag'  # This can be set to either 'literal', 'perspective-taking' or 'prag'
optimality_alpha_learner = 3.0  # Goodman & Stuhlmuller (2013) fitted sal_alpha = 3.4 to participant data with 4x3 lexicon of three number words ('one', 'two', and 'three') that could be mapped to four different world states (0, 1, 2, and 3)

pragmatic_level_sp_hyp_lr = 'prag'  # The assumption that the learner has about the speaker's pragmatic level



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

learner_type = 'both_unknown'  # This can be set to either 'perspective_unknown', 'lexicon_unknown' or 'both_unknown'

perspective_prior_type = 'egocentric'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
perspective_prior_strength = 0.9  #  The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
perspective_prior_strength_string = convert_array_to_string(perspective_prior_strength)



lexicon_prior_type = 'neutral'  # This can be set to either 'neutral', 'ambiguous_fixed', 'half_ambiguous_fixed', 'expressivity_bias' or 'compressibility_bias'
lexicon_prior_constant = 0.0 # Determines the strength of the lexicon bias, with small c creating a STRONG prior and large c creating a WEAK prior. (And with c = 1000 creating an almost uniform prior.)
lexicon_prior_constant_string = convert_array_to_string(lexicon_prior_constant)



# 2.5: The parameters that determine the amount of data_dict that the learner gets to see, and the amount of runs of the simulation:

n_utterances = 1  # This parameter determines how many signals the learner gets to observe in each context
n_contexts = 300  # The number of contexts that the learner gets to see. If context_generation = 'optimal' n_contexts for n_meanings = 2 can be anything in the table of 4 (e.g. 20, 40, 60, 80, 100, etc.); for n_meanings = 3 anything in the table of 12 (e.g. 12, 36, 60, 84, 108, etc.); and for n_meanings = 4 anything in the table of 48 (e.g. 48, 96, 144, etc.).

# 2.6: The parameters that determine how learning is measured:

theta_fixed = 0.01  # The learner is considered to have acquired the correct hypothesis if the posterior probability for that hypothesis exceeds (1.-theta_fixed)
## The parameters below serve to make a 'convergence time over theta' plot, where the amount of learning trials needed to reach a certain level of convergence on the correct hypothesis is plotted against different values of theta.
theta_start = 0.0
theta_stop = 1.0
theta_step = 0.00001
theta_range = np.arange(theta_start, theta_stop, theta_step)
theta_step_string = str(theta_step)
theta_step_string = theta_step_string.replace(".", "")



# 2.7: The parameters that determine the type and number of simulations that are run:

#FIXME: In the current implementation 'half_ambiguous_lex' can have different instantiations. Therefore, if we run a simulation where there are different lexicon_type_probs, we will want the run_type to be 'population_same_pop' to make sure that all the 'half ambiguous' speakers do have the SAME half ambiguous lexicon.
run_type = 'dyadic'  # This parameter determines whether the learner communicates with only one speaker ('dyadic') or with a population ('population_diff_pop' if there are no speakers with the 'half_ambiguous' lexicon type, 'population_same_pop' if there are, 'population_same_pop_dist_learner' if the learner can distinguish between different speakers), or whether we do an iterated learning model ('iteration')
n_runs = 100  # The number of runs of the simulation
report_every_r = 10

which_hyps_on_graph = 'all_hyps' # This is used in the plot_post_probs_over_lex_hyps() function, and can be set to either 'all_hyps' or 'lex_hyps_only'
x_axis_start = 0  # This determines where the xticks on the plot start
x_axis_steps = 10  # This determines where the xticks are placed on the plot.

lex_measure = 'ca'  # This can be set to either 'mi' for mutual information or 'ca' for communicative accuracy (of the lexicon with itself)

text_size = 1.7

inf_legend_cutoff = 6
high_cut_off = 120

legend = True

#######################################################################################################################



def calc_informativity_timecourse_learner(multi_run_log_posterior_matrix, perspective_hyps, informativity_per_lex_hyp):
    shape = multi_run_log_posterior_matrix.shape
    sum_informativity_per_context_per_run = np.zeros((shape[0], shape[1]))
    for r in range(shape[0]):
        for c in range(shape[1]):
            log_posterior = multi_run_log_posterior_matrix[r][c]
            log_posterior_split = np.asarray(np.split(log_posterior, len(perspective_hyps)))
            log_posterior_collapsed_on_p_hyps = np.logaddexp(log_posterior_split[0], log_posterior_split[1])
            log_informativity_per_lex_hyp = np.log(informativity_per_lex_hyp)
            log_posteriors_times_informativity = np.add(log_posterior_collapsed_on_p_hyps, log_informativity_per_lex_hyp)
            log_sum_informativity = logsumexp(log_posteriors_times_informativity)
            sum_informativity_per_context_per_run[r][c] = np.exp(log_sum_informativity)
    mean_informativity_over_time = np.mean(sum_informativity_per_context_per_run, axis=0)
    return mean_informativity_over_time





def plot_timecourse_learning_diff_lex_types(plot_file_path, plot_file_title, plot_title, ylabel, inf_level_per_lex_type_sorted, posterior_mass_correct_matrix_sorted, high_cut_off, text_size, lex_measure, baseline, maximum, legend):
    sns.set_style("whitegrid")
    sns.set(font_scale=text_size)
    # palette = itertools.cycle(sns.light_palette("dark lavender", input="xkcd", n_colors=len(mean_composite_hyp_posterior_mass_correct_per_lex_type))) # could also be 'deep'
    # palette = itertools.cycle(sns.diverging_palette(10, 220, sep=80, n=len(unique_informativity_per_lexicon)))
    palette = itertools.cycle(sns.cubehelix_palette(n_colors=len(inf_level_per_lex_type_sorted), reverse=True))
    ## Flip the arrays for plotting and labelling so that legend shows with lowest ca at bottom and highest ca at top:
    inf_level_per_lex_type_sorted = inf_level_per_lex_type_sorted[::-1]
    posterior_mass_correct_matrix_sorted = posterior_mass_correct_matrix_sorted[::-1]
    with sns.axes_style("whitegrid"):
        if legend == True:
            fig, ax = plt.subplots(figsize=(10.5, 6))
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
        for l in range(len(inf_level_per_lex_type_sorted)):
            inf_value = inf_level_per_lex_type_sorted[l]
            # if l < inf_legend_cutoff or l > (len(mean_composite_hyp_posterior_mass_correct_per_lex_type) - inf_legend_cutoff)-1:
            #     ax.plot(mean_composite_hyp_posterior_mass_correct_per_lex_type[l], label='ca = '+str(inf_value), color=next(palette))
            # elif l == inf_legend_cutoff:
            #     ax.plot(mean_composite_hyp_posterior_mass_correct_per_lex_type[l], label='etc.', color=next(palette))
            # else:
            #     ax.plot(mean_composite_hyp_posterior_mass_correct_per_lex_type[l], color=next(palette))
            ax.plot(posterior_mass_correct_matrix_sorted[l][:high_cut_off+1], label='L'+str(l+1)+' ('+lex_measure+' = '+str(abs(inf_value))+')', color=next(palette))
        ax.axhline(baseline, color='0.6', linestyle='--', linewidth=3, label='persp. prior')
        ax.axhline(maximum, color='0.2', linewidth=3)
        ax.set_xlim(0, high_cut_off)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(np.arange(x_axis_start, (high_cut_off+1), x_axis_steps))
        ax.set_yticks(np.arange(0.0, 1.05, 0.1))
        ax.tick_params(labelright=True)
        if legend == True:
            # Shrink current axis by 28%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.77, box.height])
            # Put a legend to the right of the current axis
            legend = ax.legend(loc='center left', bbox_to_anchor=(1.07, 0.5), frameon=True, fontsize=16)
            legend.get_frame().set_linewidth(1.5)
    plt.xlabel('No. of observations', fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.suptitle(plot_title, fontsize=16)
    plt.savefig(plot_file_path+'Plot_Tmcrse_Learning_Diff_Lex_Types_'+plot_file_title+'.png')
    plt.show()




def plot_timecourse_learning_diff_lex_types_avg_two_measures(plot_file_path, plot_file_title, plot_title, ylabel, inf_level_per_lex_type_sorted, posterior_mass_correct_matrix_sorted_msr_1, posterior_mass_correct_matrix_sorted_msr_2, high_cut_off, text_size, lex_measure, baseline, maximum):
    sns.set_style("whitegrid")
    sns.set(font_scale=text_size)
    first_quartile_msr_1 = np.percentile(posterior_mass_correct_matrix_sorted_msr_1, 25, axis=0)
    median_msr_1 = np.percentile(posterior_mass_correct_matrix_sorted_msr_1, 50, axis=0)
    third_quartile_msr_1 = np.percentile(posterior_mass_correct_matrix_sorted_msr_1, 75, axis=0)
    first_quartile_msr_2 = np.percentile(posterior_mass_correct_matrix_sorted_msr_2, 25, axis=0)
    median_msr_2 = np.percentile(posterior_mass_correct_matrix_sorted_msr_2, 50, axis=0)
    third_quartile_msr_2 = np.percentile(posterior_mass_correct_matrix_sorted_msr_2, 75, axis=0)
    color_post_prob = sns.color_palette()[1]
    color_inf = sns.color_palette()[3]
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots()
        ax.axhline(baseline, color='0.6', linestyle='--')
        ax.axhline(maximum, color='0.2')
        ax.plot(np.arange(high_cut_off), median_msr_1[0:high_cut_off], color=color_post_prob, label='post. prob. on correct hyp.')
        ax.fill_between(np.arange(high_cut_off), first_quartile_msr_1[0:high_cut_off], third_quartile_msr_1[0:high_cut_off], color=color_post_prob, alpha=0.5)
        ax.plot(np.arange(high_cut_off), median_msr_2[0:high_cut_off], color=color_inf, label='inferred informativeness')
        ax.fill_between(np.arange(high_cut_off), first_quartile_msr_2[0:high_cut_off], third_quartile_msr_2[0:high_cut_off], color=color_inf, alpha=0.5)
        ax.set_xlim(0, high_cut_off)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(np.arange(x_axis_start, (high_cut_off+1), x_axis_steps))
        ax.set_yticks(np.arange(0.0, 1.05, 0.1))
        ax.tick_params(labelright=True)
        # Shrink current axis by 28%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.84, box.height])
        # Put a legend to the right of the current axis
        legend = ax.legend(loc='center left', bbox_to_anchor=(1.07, 0.5), frameon=True, fontsize=12)
        legend.get_frame().set_linewidth(1.5)
    plt.xlabel('No. of observations')
    plt.ylabel(ylabel)
    plt.title(plot_title)
    plt.savefig(plot_file_path+'Plot_Tmcrse_Learning_Diff_Lex_Types_Average_'+plot_file_title+'.png')
    plt.show()



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

#######################################################################################################################



if __name__ == "__main__":

    speaker_lex_type = 'specified_lexicon'  # The lexicon type of the speaker. This can be set to either 'optimal_lex', 'half_ambiguous_lex', 'fully_ambiguous_lex' or 'specified_lexicon'

    inf_level_per_lex_type = np.zeros(len(unique_informativity_per_lexicon))
    mean_informativity_timecourse_per_lex_type = np.zeros((len(unique_informativity_per_lexicon), (n_contexts+1)))
    mean_lex_hyp_posterior_mass_correct_per_lex_type = np.zeros((len(unique_informativity_per_lexicon), (n_contexts+1)))
    mean_composite_hyp_posterior_mass_correct_per_lex_type = np.zeros((len(unique_informativity_per_lexicon), (n_contexts+1)))

    percentiles_lex_hyp_posterior_mass_correct_per_lex_type = np.zeros((len(unique_informativity_per_lexicon), 4, (n_contexts+1)))
    percentiles_p_hyp_posterior_mass_correct_per_lex_type = np.zeros(
        (len(unique_informativity_per_lexicon), 4, (n_contexts + 1)))
    percentiles_composite_hyp_posterior_mass_correct_per_lex_type = np.zeros(
        (len(unique_informativity_per_lexicon), 4, (n_contexts + 1)))
    counter = 0
    key_list = inf_level_lex_index_unique_dict.keys()
    print "key_list is:"
    print key_list
    print "len(key_list) is:"
    print len(key_list)

    starting_point = 0
    end_point = len(key_list)


    key_list_sliced = key_list[starting_point:end_point]
    print "key_list_sliced is;"
    print key_list_sliced
    print "len(key_list_sliced) is;"
    print len(key_list_sliced)
    for key in key_list_sliced:
    # for key in inf_level_lex_index_unique_dict:
        speaker_lex_index = inf_level_lex_index_unique_dict[key][0]
        print ''
        print ''
        print 'This is key value number:'
        print counter
        print ''
        print "speaker_lex_index is:"
        print speaker_lex_index
        print ''
        inf_level = key
        print "inf_level is:"
        print inf_level
        inf_level_per_lex_type[counter] = inf_level


        run_type_dir = 'Learner_Speaker'


        condition_dir = '/results_learning_diff_lex_types_3M_3S_context_gen_'+context_generation+'_p_prior_'+perspective_prior_type[0:4]+'_sp_'+pragmatic_level_speaker+'_a_'+str(optimality_alpha_speaker)[0]+'_lr_'+pragmatic_level_learner+'_a_'+str(optimality_alpha_learner)[0]+'_lr_sp_hyp_'+pragmatic_level_sp_hyp_lr+'_'+lex_measure



        if context_generation == 'random':
            file_title = run_type+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+str(context_generation)+'_err_'+error_string+'_sp_'+pragmatic_level_speaker+'_sp_a_'+str(optimality_alpha_speaker)[0]+'_sp_lex_'+speaker_lex_type[:-4]+'_index_'+str(speaker_lex_index)+'_inf_'+inf_level+'_sp_p_'+str(speaker_perspective)[0]+'_lr_'+pragmatic_level_learner+'_lr_a_'+str(optimality_alpha_learner)[0]+'_lr_sp_hyp_'+pragmatic_level_sp_hyp_lr+'_'+learner_learning_type+'_lex_prior_'+lexicon_prior_type+'_'+lexicon_prior_constant_string+'_p_prior_'+perspective_prior_type+'_'+perspective_prior_strength_string+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U_'+lex_measure

        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            file_title = run_type+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_C'+'_err_'+error_string+'_sp_'+pragmatic_level_speaker+'_sp_a_'+str(optimality_alpha_speaker)[0]+'_sp_lex_'+speaker_lex_type[:-4]+'_index_'+str(speaker_lex_index)+'_inf_'+inf_level+'_sp_p_'+str(speaker_perspective)[0]+'_lr_'+pragmatic_level_learner+'_lr_a_'+str(optimality_alpha_learner)[0]+'_lr_sp_hyp_'+pragmatic_level_sp_hyp_lr+'_'+learner_learning_type+'_lex_prior_'+lexicon_prior_type+'_'+lexicon_prior_constant_string+'_p_prior_'+perspective_prior_type+'_'+perspective_prior_strength_string+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U_'+lex_measure


        print "file_title is:"
        print file_title


        pickle_file_folder = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/'+run_type_dir+'/timecourse_measures_learning_diff_lex_types_'+str(n_meanings)+'M_'+str(n_signals)+'S_context_gen_'+context_generation+'_p_prior_'+perspective_prior_type[0:4]+'_sp_'+pragmatic_level_speaker+'_a_'+str(optimality_alpha_speaker)[0]+'_lr_'+pragmatic_level_learner+'_a_'+str(optimality_alpha_learner)[0]+'_lr_sp_hyp_'+pragmatic_level_sp_hyp_lr+'_'+lex_measure+'/'



        print ''
        print ''
        print "pickle_file_folder is:"
        print pickle_file_folder

        # all_results_dict = pickle.load(open(pickle_file_title_all_results + '.p', 'rb'))
        # multi_run_log_posterior_matrix = all_results_dict['multi_run_log_posterior_matrix']

        t10 = time.clock()

        # informativity_timecourse = calc_informativity_timecourse_learner(multi_run_log_posterior_matrix, perspective_hyps, informativity_per_lexicon)
        # print ''
        # # print "informativity_timecourse is:"
        # # print informativity_timecourse
        # print "informativity_timecourse.shape is:"
        # print informativity_timecourse.shape
        #
        # pickle_file_title_inf_timecourse = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/' + run_type_dir + condition_dir + '/Inf_Tmcrse_' + file_title
        #
        # pickle.dump(informativity_timecourse, open(pickle_file_title_inf_timecourse + '.p', 'wb'))

        #
        # pickle_file_title_correct_hyp_posterior_mass_mean = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/' + run_type_dir + condition_dir + '/Correct_Hyp_Post_Mass_Mean_' + file_title

        # mean_correct_hyp_posterior_mass_dict = pickle.load(open(pickle_file_title_correct_hyp_posterior_mass_mean + '.p', 'rb'))

        mean_correct_hyp_posterior_mass_dict = pickle.load(open(pickle_file_folder+'Correct_Hyp_Post_Mass_Mean_'+file_title+'.p', 'rb'))


        mean_lex_hyp_posterior_mass_correct = mean_correct_hyp_posterior_mass_dict['mean_lex_hyp_posterior_mass_correct']

        mean_composite_hyp_posterior_mass_correct = mean_correct_hyp_posterior_mass_dict[
            'mean_composite_hyp_posterior_mass_correct']


        percentiles_posterior_mass_dict = pickle.load(open(pickle_file_folder+'Correct_Hyp_Post_Mass_Percentiles_'+file_title+'.p', 'rb'))

        percentiles_lex_hyp_posterior_mass_correct = percentiles_posterior_mass_dict['percentiles_lex_hyp_posterior_mass_correct']

        percentiles_p_hyp_posterior_mass_correct = percentiles_posterior_mass_dict['percentiles_p_hyp_posterior_mass_correct']

        percentiles_composite_hyp_posterior_mass_correct = percentiles_posterior_mass_dict[
            'percentiles_composite_hyp_posterior_mass_correct']



        # informativity_timecourse = pickle.load(open(pickle_file_folder+'Inf_Tmcrse_'+file_title+'.p', 'rb'))
        #
        # mean_informativity_timecourse_per_lex_type[counter] = informativity_timecourse

        mean_lex_hyp_posterior_mass_correct_per_lex_type[counter] = mean_lex_hyp_posterior_mass_correct

        mean_composite_hyp_posterior_mass_correct_per_lex_type[counter] = mean_composite_hyp_posterior_mass_correct



        percentiles_lex_hyp_posterior_mass_correct_per_lex_type[counter] = percentiles_lex_hyp_posterior_mass_correct

        percentiles_p_hyp_posterior_mass_correct_per_lex_type[counter] = percentiles_p_hyp_posterior_mass_correct

        percentiles_composite_hyp_posterior_mass_correct_per_lex_type[counter] = percentiles_composite_hyp_posterior_mass_correct

        counter += 1

    #####################################################################################
    ## GENERATING THE PLOT:

    plot_file_path = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Plots/Learner_Speaker/'


    if context_generation == 'random':
        plot_file_title = run_type+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+str(context_generation)+'_err_'+error_string+'_sp_'+pragmatic_level_speaker+'_a_'+str(optimality_alpha_speaker)[0]+'_lr_'+pragmatic_level_learner+'_a_'+str(optimality_alpha_learner)[0]+'_lr_sp_hyp_'+pragmatic_level_sp_hyp_lr+'_sp_p_'+str(speaker_perspective)[0]+'_'+learner_learning_type+'_lex_prior_'+lexicon_prior_type[:4]+'_'+lexicon_prior_constant_string+'_p_prior_'+perspective_prior_type[:4]+'_'+perspective_prior_strength_string+'_'+str(int(high_cut_off))+'_C_'+str(int(n_utterances))+'_U_'+lex_measure

    elif context_generation == 'only_helpful' or context_generation == 'optimal':
        plot_file_title = run_type+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_C'+'_err_'+error_string+'_sp_'+pragmatic_level_speaker+'_a_'+str(optimality_alpha_speaker)[0]+'_lr_'+pragmatic_level_learner+'_a_'+str(optimality_alpha_learner)[0]+'_lr_sp_hyp_'+pragmatic_level_sp_hyp_lr+'_sp_p_'+str(speaker_perspective)[0]+'_'+learner_learning_type+'_lex_prior_'+lexicon_prior_type[:4]+'_'+lexicon_prior_constant_string+'_p_prior_'+perspective_prior_type[:4]+'_'+perspective_prior_strength_string+'_'+str(int(high_cut_off))+'_C_'+str(int(n_utterances))+'_U_'+lex_measure

    argsort_inf_level_per_lex_type = np.argsort(inf_level_per_lex_type)

    inf_level_per_lex_type_sorted = inf_level_per_lex_type[argsort_inf_level_per_lex_type]

    informativity_timecourse_matrix_sorted = mean_informativity_timecourse_per_lex_type[argsort_inf_level_per_lex_type]

    posterior_mass_correct_lex_hyp_matrix_sorted = mean_lex_hyp_posterior_mass_correct_per_lex_type[argsort_inf_level_per_lex_type]

    posterior_mass_correct_composite_hyp_matrix_sorted = mean_composite_hyp_posterior_mass_correct_per_lex_type[argsort_inf_level_per_lex_type]


    percentiles_posterior_mass_correct_lex_hyp_matrix_sorted = percentiles_lex_hyp_posterior_mass_correct_per_lex_type[argsort_inf_level_per_lex_type]

    percentiles_posterior_mass_correct_p_hyp_matrix_sorted = percentiles_p_hyp_posterior_mass_correct_per_lex_type[argsort_inf_level_per_lex_type]

    percentiles_posterior_mass_correct_composite_hyp_matrix_sorted = percentiles_composite_hyp_posterior_mass_correct_per_lex_type[argsort_inf_level_per_lex_type]



    median_posterior_mass_correct_lex_hyp_matrix_sorted = percentiles_posterior_mass_correct_lex_hyp_matrix_sorted[:,1,:]

    median_posterior_mass_correct_p_hyp_matrix_sorted = percentiles_posterior_mass_correct_p_hyp_matrix_sorted[:,1,:]

    median_posterior_mass_correct_composite_hyp_matrix_sorted = percentiles_posterior_mass_correct_composite_hyp_matrix_sorted[:,1,:]


    maximum = 1.


    if perspective_prior_type == 'neutral':
        plot_title = 'Informativeness, Uniform prior + sp.: '+pragmatic_level_speaker+', a = '+str(optimality_alpha_speaker)[0]+' lr.: '+pragmatic_level_learner+', a = '+str(optimality_alpha_learner)[0]+' sp. hyp.: '+pragmatic_level_sp_hyp_lr
        baseline = 1./len(perspective_hyps)
    elif perspective_prior_type == 'egocentric':
        plot_title = 'Informativeness, Ego. prior + sp.: '+pragmatic_level_speaker+', a = '+str(optimality_alpha_speaker)[0]+' lr.: '+pragmatic_level_learner+', a = '+str(optimality_alpha_learner)[0]+' sp. hyp.: '+pragmatic_level_sp_hyp_lr
        baseline = 1.-perspective_prior_strength

    plot_timecourse_learning_diff_lex_types(plot_file_path, plot_file_title+'_inf_tmcrse', plot_title, 'Mean informativeness of inferred lexicon', unique_informativity_per_lexicon, informativity_timecourse_matrix_sorted, high_cut_off, text_size, lex_measure, baseline=-0.5, maximum=1.5, legend=legend)




    if perspective_prior_type == 'neutral':
        plot_title = 'Lex. hyp., Uniform prior + sp.: '+pragmatic_level_speaker+', a = '+str(optimality_alpha_speaker)[0]+' lr.: '+pragmatic_level_learner+', a = '+str(optimality_alpha_learner)[0]+' sp. hyp.: '+pragmatic_level_sp_hyp_lr
        baseline = 1./len(perspective_hyps)
    elif perspective_prior_type == 'egocentric':
        plot_title = 'Lex. hyp., Ego. prior + sp.: '+pragmatic_level_speaker+', a = '+str(optimality_alpha_speaker)[0]+' lr.: '+pragmatic_level_learner+', a = '+str(optimality_alpha_learner)[0]+' sp. hyp.: '+pragmatic_level_sp_hyp_lr
        baseline = 1.-perspective_prior_strength

    plot_timecourse_learning_diff_lex_types(plot_file_path, plot_file_title+'_mean_lex_hyp_correct', plot_title, 'Mean posterior prob. on correct lex. hyp.', unique_informativity_per_lexicon, posterior_mass_correct_lex_hyp_matrix_sorted, high_cut_off, text_size, lex_measure, baseline=-0.5, maximum=maximum, legend=legend)


    plot_timecourse_learning_diff_lex_types(plot_file_path, plot_file_title+'_median_lex_hyp_correct', plot_title, 'Median posterior prob. on correct lex. hyp.', unique_informativity_per_lexicon, median_posterior_mass_correct_lex_hyp_matrix_sorted, high_cut_off, text_size, lex_measure, baseline=-0.5, maximum=maximum, legend=legend)




    if perspective_prior_type == 'neutral':
        plot_title = 'P. hyp., Uniform prior + sp.: '+pragmatic_level_speaker+', a = '+str(optimality_alpha_speaker)[0]+' lr.: '+pragmatic_level_learner+', a = '+str(optimality_alpha_learner)[0]+' sp. hyp.: '+pragmatic_level_sp_hyp_lr
        baseline = 1./len(perspective_hyps)
    elif perspective_prior_type == 'egocentric':
        plot_title = 'P. hyp., Ego. prior + sp.: '+pragmatic_level_speaker+', a = '+str(optimality_alpha_speaker)[0]+' lr.: '+pragmatic_level_learner+', a = '+str(optimality_alpha_learner)[0]+' sp. hyp.: '+pragmatic_level_sp_hyp_lr
        baseline = 1.-perspective_prior_strength


    plot_timecourse_learning_diff_lex_types(plot_file_path, plot_file_title+'_median_p_hyp_correct', plot_title, 'Median posterior prob. on correct p. hyp.', unique_informativity_per_lexicon, median_posterior_mass_correct_p_hyp_matrix_sorted, high_cut_off, text_size, lex_measure, baseline=-0.5, maximum=maximum, legend=legend)




    if perspective_prior_type == 'neutral':
        plot_title = 'Comp. hyp., Uniform prior + sp.: '+pragmatic_level_speaker+', a = '+str(optimality_alpha_speaker)[0]+' lr.: '+pragmatic_level_learner+', a = '+str(optimality_alpha_learner)[0]+' sp. hyp.: '+pragmatic_level_sp_hyp_lr
        baseline = 1./len(perspective_hyps)
    elif perspective_prior_type == 'egocentric':
        plot_title = 'Comp. hyp., Ego. prior + sp.: '+pragmatic_level_speaker+', a = '+str(optimality_alpha_speaker)[0]+' lr.: '+pragmatic_level_learner+', a = '+str(optimality_alpha_learner)[0]+' sp. hyp.: '+pragmatic_level_sp_hyp_lr
        baseline = 1.-perspective_prior_strength


    plot_timecourse_learning_diff_lex_types(plot_file_path, plot_file_title+'_mean_comp_hyp_correct', plot_title, 'Mean posterior prob. on correct composite hyp.', unique_informativity_per_lexicon, posterior_mass_correct_composite_hyp_matrix_sorted, high_cut_off, text_size, lex_measure, baseline, maximum, legend=legend)


    plot_timecourse_learning_diff_lex_types(plot_file_path, plot_file_title+'_median_comp_hyp_correct', plot_title, 'Median posterior prob. on correct composite hyp.', unique_informativity_per_lexicon, median_posterior_mass_correct_composite_hyp_matrix_sorted, high_cut_off, text_size, lex_measure, baseline, maximum, legend=legend)



    # plot_timecourse_learning_diff_lex_types_avg_two_measures(plot_file_path, plot_file_title+'_avg_two_measures', plot_title, 'Avg. post. prob. // avg. informativeness', inf_level_per_lex_type_sorted, posterior_mass_correct_composite_hyp_matrix_sorted, informativity_timecourse_matrix_sorted, high_cut_off, text_size, lex_measure, baseline, maximum)
    #
