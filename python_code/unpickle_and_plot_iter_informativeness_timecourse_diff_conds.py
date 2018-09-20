__author__ = 'Marieke Woensdregt'


import numpy as np
import pickle
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

import hypspace
import lex
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
n_contexts = 60

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

low_cut_off = 0
high_cut_off = n_iterations
legend = False
text_size = 1.7
#######################################################################################################################



def get_mean_percentiles_avg_informativity(pickle_directory, n_runs, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type):
    perspective_prior_strength_string = saveresults.convert_array_to_string(perspective_prior_strength)
    ##
    if selection_type == 'none' or selection_type == 'p_taking':
        if context_generation == 'random':
            filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    ##
    elif selection_type == 'ca_with_parent':
        if context_generation == 'random':
            filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    ##
    percentiles_n_mean_informativity = pickle.load(open(pickle_directory+'lex_inf_data_'+lex_measure+'_'+filename_short+'.p', 'rb'))
    unique_inf_values = percentiles_n_mean_informativity['unique_inf_values']
    inf_values_last_generation = percentiles_n_mean_informativity['inf_values_last_generation']
    mean_informativity_over_gens = percentiles_n_mean_informativity['mean_inf_over_gens']
    conf_invs_informativity_over_gens = percentiles_n_mean_informativity['conf_intervals_inf_over_gens']
    percentiles_informativity_over_gens = percentiles_n_mean_informativity['percentiles_inf_over_gens']
    baseline_inf = percentiles_n_mean_informativity['baseline_inf']
    max_inf = percentiles_n_mean_informativity['max_inf']
    return inf_values_last_generation, mean_informativity_over_gens, conf_invs_informativity_over_gens, percentiles_informativity_over_gens, unique_inf_values, baseline_inf, max_inf



def plot_percentile_timecourses_three_priors_three_conds(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, high_cut_off, percentiles_exo_select_none, percentiles_exo_select_ca, percentiles_exo_select_p_taking, percentiles_neut_select_none, percentiles_neut_select_ca, percentiles_neut_select_p_taking, percentiles_ego_select_none, percentiles_ego_select_ca, percentiles_ego_select_p_taking):
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    sns.set(font_scale=text_size)
    color_no_selection = sns.color_palette()[0]
    color_select_ca = sns.color_palette()[3]
    color_select_p_taking = sns.color_palette()[4]
    # fig, ax = plt.subplots(nrows=3, ncols=2)
    fig = plt.figure()
    with sns.axes_style("whitegrid"):
        ### TIMECOURSE PRIOR 1: EXOCENTRIC:
        ax1 = fig.add_subplot(311)
        ax1.set_title("Allocentric P Prior")
        ax1.set_ylim(-0.05, (max_inf + 0.05))
        ax1.set_yticks(np.arange(0.0, (max_inf + 0.05), 0.2))
        #### CONDITION 1: No Selection:
        ax1.plot(np.arange(high_cut_off), percentiles_exo_select_none[1][0:high_cut_off], label='No Selection', color=color_no_selection)
        ax1.fill_between(np.arange(high_cut_off), percentiles_exo_select_none[0][0:high_cut_off], percentiles_exo_select_none[2][0:high_cut_off], facecolor=color_no_selection, alpha=0.5)
        #### CONDITION 2: Selection on CS:
        ax1.plot(np.arange(high_cut_off), percentiles_exo_select_ca[1][0:high_cut_off], label='Select CS', color=color_select_ca)
        ax1.fill_between(np.arange(high_cut_off), percentiles_exo_select_ca[0][0:high_cut_off], percentiles_exo_select_ca[2][0:high_cut_off], facecolor=color_select_ca, alpha=0.5)
        #### CONDITION 3: Selection on P-taking:
        ax1.plot(np.arange(high_cut_off), percentiles_exo_select_p_taking[1][0:high_cut_off], label='Select P-taking', color=color_select_p_taking)
        ax1.fill_between(np.arange(high_cut_off), percentiles_exo_select_p_taking[0][0:high_cut_off], percentiles_exo_select_p_taking[2][0:high_cut_off], facecolor=color_select_p_taking, alpha=0.5)

        # Shrink current axis by 28%
        # box = ax1.get_position()
        # ax1.set_position([box.x0, box.y0, box.width * 0.72, box.height])
        # Put a legend to the right of the current axis
        # ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)


        ### TIMECOURSE PRIOR 2: NEUTRAL:
        ax2 = fig.add_subplot(312)
        ax2.set_title("Uniform P Prior")
        ax2.set_ylim(-0.05, (max_inf + 0.05))
        ax2.set_yticks(np.arange(0.0, (max_inf + 0.05), 0.2))
        #### CONDITION 1: No Selection:
        ax2.plot(np.arange(high_cut_off), percentiles_neut_select_none[1][0:high_cut_off], color=color_no_selection)
        ax2.fill_between(np.arange(high_cut_off), percentiles_neut_select_none[0][0:high_cut_off], percentiles_neut_select_none[2][0:high_cut_off], facecolor=color_no_selection, alpha=0.5)
        #### CONDITION 2: Selection on CS:
        ax2.plot(np.arange(high_cut_off), percentiles_neut_select_ca[1][0:high_cut_off], color=color_select_ca)
        ax2.fill_between(np.arange(high_cut_off), percentiles_neut_select_ca[0][0:high_cut_off], percentiles_neut_select_ca[2][0:high_cut_off], facecolor=color_select_ca, alpha=0.5)
        #### CONDITION 3: Selection on P-taking:
        ax2.plot(np.arange(high_cut_off), percentiles_neut_select_p_taking[1][0:high_cut_off], color=color_select_p_taking)
        ax2.fill_between(np.arange(high_cut_off), percentiles_neut_select_p_taking[0][0:high_cut_off], percentiles_neut_select_p_taking[2][0:high_cut_off], facecolor=color_select_p_taking, alpha=0.5)

        ### TIMECOURSE PRIOR 3: EGOCENTRIC:
        ax3 = fig.add_subplot(313)
        ax3.set_title("Egocentric P Prior")
        ax3.set_ylim(-0.05, (max_inf + 0.05))
        ax3.set_yticks(np.arange(0.0, (max_inf + 0.05), 0.2))
        #### CONDITION 1: No Selection:
        ax3.plot(np.arange(high_cut_off), percentiles_ego_select_none[1][0:high_cut_off], color=color_no_selection)
        ax3.fill_between(np.arange(high_cut_off), percentiles_ego_select_none[0][0:high_cut_off], percentiles_ego_select_none[2][0:high_cut_off], facecolor=color_no_selection, alpha=0.5)
        #### CONDITION 2: Selection on CS:
        ax3.plot(np.arange(high_cut_off), percentiles_ego_select_ca[1][0:high_cut_off], color=color_select_ca)
        ax3.fill_between(np.arange(high_cut_off), percentiles_ego_select_ca[0][0:high_cut_off], percentiles_ego_select_ca[2][0:high_cut_off], facecolor=color_select_ca, alpha=0.5)
        #### CONDITION 3: Selection on P-taking:
        ax3.plot(np.arange(high_cut_off), percentiles_ego_select_p_taking[1][0:high_cut_off], color=color_select_p_taking)
        ax3.fill_between(np.arange(high_cut_off), percentiles_ego_select_p_taking[0][0:high_cut_off], percentiles_ego_select_p_taking[2][0:high_cut_off], facecolor=color_select_p_taking, alpha=0.5)
    fig.subplots_adjust(hspace=.8)
    plt.xlabel('Generations')
    plt.title(plot_title)
    plt.savefig(plot_file_path+'Plot_inf_Perc_Diff_Conds_'+lex_measure+'_'+plot_file_title+'_high_bound_'+str(high_cut_off)+'.png')
    plt.show()




def plot_mean_conf_invs_three_conds(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, high_cut_off, mean_select_none, conf_invs_select_none, mean_select_ca, conf_invs_select_ca, mean_select_p_taking, conf_invs_select_p_taking, baseline_inf, max_inf):
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    sns.set(font_scale=text_size)
    color_no_selection = sns.color_palette()[0]
    color_select_ca = sns.color_palette()[3]
    color_select_p_taking = sns.color_palette()[4]
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots()
        ax.set_ylim(-0.05, (max_inf + 0.05))
        ax.set_yticks(np.arange(0.0, (max_inf + 0.05), 0.2))
        ax.axhline(baseline_inf, color='0.6', linestyle='--')
        ax.axhline(max_inf, color='0.2')
        #### CONDITION 1: No Selection:
        ax.plot(np.arange(high_cut_off), mean_select_none[0:high_cut_off], label='No Selection', color=color_no_selection)
        ax.fill_between(np.arange(high_cut_off), conf_invs_select_none[0][0:high_cut_off], conf_invs_select_none[1][0:high_cut_off], facecolor=color_no_selection, alpha=0.5)
        #### CONDITION 2: Selection on CS:
        ax.plot(np.arange(high_cut_off), mean_select_ca[0:high_cut_off], label='Select CS', color=color_select_ca)
        ax.fill_between(np.arange(high_cut_off), conf_invs_select_ca[0][0:high_cut_off], conf_invs_select_ca[1][0:high_cut_off], facecolor=color_select_ca, alpha=0.5)
        #### CONDITION 3: Selection on P-taking:
        ax.plot(np.arange(high_cut_off), mean_select_p_taking[0:high_cut_off], label='Select P-taking', color=color_select_p_taking)
        ax.fill_between(np.arange(high_cut_off), conf_invs_select_p_taking[0][0:high_cut_off], conf_invs_select_p_taking[1][0:high_cut_off], facecolor=color_select_p_taking, alpha=0.5)
        if legend == True:
            # Shrink current axis by 28%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.72, box.height])
            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    plt.xlabel('Generations')
    plt.ylabel(y_axis_label)
    plt.title(plot_title)
    plt.savefig(plot_file_path+'Plot_inf_Mean_CI_'+lex_measure+'_'+plot_file_title+'_high_bound_'+str(high_cut_off)+'.png')
    plt.show()



def plot_percentiles_three_conds(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, high_cut_off, percentiles_select_none, percentiles_select_ca, percentiles_select_p_taking, baseline_inf, max_inf):
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    sns.set(font_scale=text_size)
    color_no_selection = sns.color_palette()[0]
    color_select_ca = sns.color_palette()[3]
    color_select_p_taking = sns.color_palette()[4]
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots()
        ax.set_ylim(-0.05, (max_inf + 0.05))
        ax.set_yticks(np.arange(0.0, (max_inf + 0.05), 0.2))
        ax.axhline(baseline_inf, color='0.6', linestyle='--')
        ax.axhline(max_inf, color='0.2')
        #### CONDITION 1: No Selection:
        ax.plot(np.arange(high_cut_off), percentiles_select_none[1][0:high_cut_off], label='No Selection', color=color_no_selection)
        ax.fill_between(np.arange(high_cut_off), percentiles_select_none[0][0:high_cut_off], percentiles_select_none[2][0:high_cut_off], facecolor=color_no_selection, alpha=0.5)
        #### CONDITION 2: Selection on CS:
        ax.plot(np.arange(high_cut_off), percentiles_select_ca[1][0:high_cut_off], label='Select CS', color=color_select_ca)
        ax.fill_between(np.arange(high_cut_off), percentiles_select_ca[0][0:high_cut_off], percentiles_select_ca[2][0:high_cut_off], facecolor=color_select_ca, alpha=0.5)
        #### CONDITION 3: Selection on P-taking:
        ax.plot(np.arange(high_cut_off), percentiles_select_p_taking[1][0:high_cut_off], label='Select P-taking', color=color_select_p_taking)
        ax.fill_between(np.arange(high_cut_off), percentiles_select_p_taking[0][0:high_cut_off], percentiles_select_p_taking[2][0:high_cut_off], facecolor=color_select_p_taking, alpha=0.5)
        if legend == True:
            # Shrink current axis by 28%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.72, box.height])
            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    plt.xlabel('Generations')
    plt.ylabel(y_axis_label)
    plt.title(plot_title)
    plt.savefig(plot_file_path+'Plot_inf_Perc_'+lex_measure+'_'+plot_file_title+'_high_bound_'+str(high_cut_off)+'.png')
    plt.show()



def plot_percentiles_one_cond(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, low_cut_off, high_cut_off, percentiles, selection_type, baseline_inf, max_inf):
    sns.set_style("ticks")
    sns.set_palette("deep")
    # sns.set(font_scale=text_size)
    # if selection_type == 'none':
    #     color = sns.color_palette()[0]
    # elif selection_type == 'ca_with_parent':
    #     color = sns.color_palette()[3]
    # elif selection_type == 'p_taking':
    #     color = sns.color_palette()[4]
    color = sns.color_palette()[0]
    with sns.axes_style("ticks"):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_ylim(-0.05, max_inf + 0.05)
        ax.set_yticks(np.arange(0.0, (max_inf + 0.05), 0.1))
        ax.set_xlim(0, high_cut_off + 1)
        ax.set_xticks(np.arange(0, high_cut_off+1, 100))
        ax.plot(np.arange(high_cut_off), percentiles[1][0:high_cut_off], color=color, label='median')
        ax.fill_between(np.arange(high_cut_off), percentiles[0][0:high_cut_off], percentiles[2][0:high_cut_off], color=color, alpha=0.5, label='IQR')
        ax.axhline(baseline_inf, color='0.6', linewidth=2, linestyle='--', label='baseline')
        ax.axhline(max_inf, color='0.2', linewidth=2, label='maximum')
        ax.axvline(low_cut_off, color='0.6', linewidth=2, linestyle=':', label='burn-in')
        fig.subplots_adjust(bottom=0.15)
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
        if legend == True:
            handles, labels = ax.get_legend_handles_labels()
            handles = [handles[0], handles[4], handles[1], handles[2], handles[3]]
            labels = [labels[0], labels[4], labels[1], labels[2], labels[3]]
            # Shrink current axis by 28%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
            # Put a legend to the right of the current axis
            ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    sns.despine()
    ax.tick_params(labelsize=14)
    plt.xlabel('Generations', fontsize=20)
    plt.ylabel(y_axis_label, fontsize=20)
    plt.suptitle(plot_title, fontsize=26)
    plt.savefig(plot_file_path+'Plot_Inf_Perc_'+lex_measure+'_'+plot_file_title+'_high_bound_'+str(high_cut_off)+'.pdf')
    plt.show()



def plot_percentiles_three_cond(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, low_cut_off, high_cut_off, percentiles_one, percentiles_two, percentiles_three, baseline_inf, max_inf, legend):
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    sns.set(font_scale=text_size)
    color_none = sns.color_palette()[0]
    color_ca = sns.color_palette()[3]
    color_p_taking = sns.color_palette()[4]
    with sns.axes_style("whitegrid"):
        if legend == True:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_ylim(-0.05, (max_inf + 0.05))
        ax.set_yticks(np.arange(0.0, (max_inf + 0.05), 0.1))
        ax.tick_params(labelright=True)
        ax.set_xlim(0, high_cut_off+1)
        ax.set_xticks(np.arange(0, high_cut_off+1, 100))
        ax.plot(np.arange(high_cut_off), percentiles_one[1][0:high_cut_off], color=color_none, label='no selection')
        ax.fill_between(np.arange(high_cut_off), percentiles_one[0][0:high_cut_off], percentiles_one[2][0:high_cut_off], color=color_none, alpha=0.5)
        ax.plot(np.arange(high_cut_off), percentiles_two[1][0:high_cut_off], color=color_p_taking, label='selection for p-inference')
        ax.fill_between(np.arange(high_cut_off), percentiles_two[0][0:high_cut_off], percentiles_two[2][0:high_cut_off], color=color_p_taking, alpha=0.5)
        ax.plot(np.arange(high_cut_off), percentiles_three[1][0:high_cut_off], color=color_ca, label='selection for communication')
        ax.fill_between(np.arange(high_cut_off), percentiles_three[0][0:high_cut_off], percentiles_three[2][0:high_cut_off], color=color_ca, alpha=0.5)
        ax.axhline(baseline_inf, color='0.6', linewidth=3, linestyle='--', label='baseline')
        ax.axhline(max_inf, color='0.2', linewidth=3, label='maximum')
        # ax.axvline(low_cut_off, color='0.6', linewidth=4, linestyle=':', label='burn-in')
        fig.subplots_adjust(bottom=0.15)
        if legend == True:
            # Shrink current axis by 28%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.65, box.height])
            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    plt.xlabel('Generations', fontsize=16)
    plt.ylabel(y_axis_label, fontsize=16)
    plt.suptitle(plot_title, fontsize=16)
    plt.savefig(plot_file_path+'Plot_Inf_Perc_No_selection_vs_P_taking_'+lex_measure+'_'+plot_file_title+'_high_bound_'+str(high_cut_off)+'.png')
    plt.show()






def plot_histogram_diff_language_samples(plot_file_path, plot_file_title, plot_title, language_samples, pop_size):
    sns.set_style("ticks")
    sns.set_palette("deep")
    colors = [sns.color_palette()[3], sns.color_palette()[1], sns.color_palette()[0]]
    labels = ["no conventions / 'scratch'", "selection for communication", "random sample", ]
    with sns.axes_style("ticks"):
        fig, ax = plt.subplots()
        for i in range(len(language_samples)):
            print "language_samples[i] is:"
            print language_samples[i]
            ax.axvline(0.3333, color='0.6', linewidth=2, linestyle='--', label='baseline')
            ax = sns.distplot(language_samples[i], bins=unique_inf_values, kde=False, color=colors[i], label=labels[i])
            ax.axvline(0.90, color='0.2', linewidth=2, label='maximum')
            # ax = sns.countplot(language_samples[i], color=colors[i])
            # lang_type_counts = np.histogram(language_samples[i], bins=unique_inf_values)
            # print "lang_type_counts are:"
            # print lang_type_counts
            # print "lang_type_counts[0] are:"
            # print lang_type_counts[0]
            # sns.barplot(y=lang_type_counts[0], color=colors[i])
    # plt.xticks(unique_inf_values)
    sns.despine()
    plt.ylim(0, pop_size+1)
    plt.xlim(0.3, 0.93)
    ax.tick_params(labelsize=14)
    plt.ylabel('no. of agents with language type', fontsize=16)
    plt.xlabel('informativeness of languages', fontsize=16)
    plt.suptitle('Distributions of language informativeness', fontsize=20)
    # plt.legend()
    plt.savefig(plot_file_path+'Plot_Hist_over_lang_inf_values_'+lex_measure+'_'+plot_file_title+'_high_bound_'+str(high_cut_off)+'.pdf')
    plt.show()





#####################################################################################

if __name__ == "__main__":


    pickle_directory = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/Iteration/'


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
#     selection_type = 'none'
#
#     mean_informativity_over_gens_exo_select_none, conf_invs_informativity_over_gens_exo_select_none, percentiles_informativity_over_gens_exo_select_none, baseline_inf, max_inf = get_mean_percentiles_avg_informativity(pickle_directory, n_runs, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type)
#
# #####################################################################################
# ## PRIOR 1: EXOCENTRIC
# ### CONDITION 2: Selection on CS:
#
#     print ''
#     print ''
#     print 'This is prior 1: Exocentric'
#     print ''
#     print 'This is condition 1: Selection on CS'
#
#     perspective_prior_type = 'egocentric'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
#     perspective_prior_strength = 0.0  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
#     selection_type = 'ca_with_parent'
#
#     mean_informativity_over_gens_exo_select_ca, conf_invs_informativity_over_gens_exo_select_ca, percentiles_informativity_over_gens_exo_select_ca, baseline_inf, max_inf = get_mean_percentiles_avg_informativity(pickle_directory, n_runs, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type)
#
# #####################################################################################
# ## PRIOR 1: EXOCENTRIC
# ### CONDITION 3: Selection on P-taking:
#
#     print ''
#     print ''
#     print 'This is prior 1: Exocentric'
#     print ''
#     print 'This is condition 1: Selection on P-taking'
#
#     perspective_prior_type = 'egocentric'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
#     perspective_prior_strength = 0.0  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
#     selection_type = 'p_taking'
#
#     mean_informativity_over_gens_exo_select_p_taking, conf_invs_informativity_over_gens_exo_select_p_taking, percentiles_informativity_over_gens_exo_select_p_taking, baseline_inf, max_inf = get_mean_percentiles_avg_informativity(pickle_directory, n_runs, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type)



#
#     #####################################################################################
# ## PRIOR 1: UNIFORM
# ### CONDITION 1: No Selection:
#
#     print ''
#     print ''
#     print 'This is prior 1: Uniform'
#     print ''
#     print 'This is condition 1: No selection'
#
#     perspective_prior_type = 'neutral'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
#     perspective_prior_strength = 0.0  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
#     selection_type = 'none'
#
#     mean_informativity_over_gens_neut_select_none, conf_invs_informativity_over_gens_neut_select_none, percentiles_informativity_over_gens_neut_select_none, baseline_inf, max_inf = get_mean_percentiles_avg_informativity(pickle_directory, n_runs, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type)
#
# #####################################################################################
# ## PRIOR 1: UNIFORM
# ### CONDITION 2: Selection on CS:
#
#     print ''
#     print ''
#     print 'This is prior 1: Uniform'
#     print ''
#     print 'This is condition 1: Selection on CS'
#
#     perspective_prior_type = 'neutral'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
#     perspective_prior_strength = 0.0  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
#     selection_type = 'ca_with_parent'
#
#     mean_informativity_over_gens_neut_select_ca, conf_invs_informativity_over_gens_neut_select_ca, percentiles_informativity_over_gens_neut_select_ca, baseline_inf, max_inf = get_mean_percentiles_avg_informativity(pickle_directory, n_runs, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type)
#
# #####################################################################################
# ## PRIOR 1: UNIFORM
# ### CONDITION 3: Selection on P-taking:
#
#     print ''
#     print ''
#     print 'This is prior 1: Uniform'
#     print ''
#     print 'This is condition 1: Selection on P-taking'
#
#     perspective_prior_type = 'neutral'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
#     perspective_prior_strength = 0.0  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
#     selection_type = 'p_taking'
#
#
#     mean_informativity_over_gens_neut_select_p_taking, conf_invs_informativity_over_gens_neut_select_p_taking, percentiles_informativity_over_gens_neut_select_p_taking, baseline_inf, max_inf = get_mean_percentiles_avg_informativity(pickle_directory, n_runs, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type)
#
#
#


    #####################################################################################
## PRIOR 1: EGOCENTRIC
### CONDITION 1: No Selection:

    print ''
    print ''
    print 'This is prior 1: Egocentric'
    print ''
    print 'This is condition 1: No selection'

    perspective_prior_type = 'egocentric'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
    perspective_prior_strength = 0.9  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
    selection_type = 'none'

    inf_values_last_generation_select_none, mean_informativity_over_gens_ego_select_none, conf_invs_informativity_over_gens_ego_select_none, percentiles_informativity_over_gens_ego_select_none, unique_inf_values, baseline_inf, max_inf = get_mean_percentiles_avg_informativity(pickle_directory, n_runs, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type)


#####################################################################################
## PRIOR 1: EGOCENTRIC
### CONDITION 2: Selection on CS:

    print ''
    print ''
    print 'This is prior 1: Egocentric'
    print ''
    print 'This is condition 1: Selection on CS'

    perspective_prior_type = 'egocentric'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
    perspective_prior_strength = 0.9  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
    selection_type = 'ca_with_parent'
    if pragmatic_level == 'literal' and selection_type == 'ca_with_parent':
        pragmatic_level = 'perspective-taking'

        inf_values_last_generation_select_ca, mean_informativity_over_gens_ego_select_ca, conf_invs_informativity_over_gens_ego_select_ca, percentiles_informativity_over_gens_ego_select_ca, unique_inf_values, baseline_inf, max_inf = get_mean_percentiles_avg_informativity(pickle_directory, n_runs, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type)

    if pragmatic_level == 'perspective-taking':
        pragmatic_level = 'literal'

#####################################################################################
## PRIOR 1: EGOCENTRIC
### CONDITION 3: Selection on P-taking:

    print ''
    print ''
    print 'This is prior 1: Egocentric'
    print ''
    print 'This is condition 1: Selection on P-taking'

    perspective_prior_type = 'egocentric'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
    perspective_prior_strength = 0.9  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
    selection_type = 'p_taking'

    inf_values_last_generation_select_p_taking, mean_informativity_over_gens_ego_select_p_taking, conf_invs_informativity_over_gens_ego_select_p_taking, percentiles_informativity_over_gens_ego_select_p_taking, unique_inf_values, baseline_inf, max_inf = get_mean_percentiles_avg_informativity(pickle_directory, n_runs, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type)



#####################################################################################
## GENERATING THE PLOT:

    plot_file_path = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Plots/Iteration/'


    perspective_prior_type = 'egocentric'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
    perspective_prior_strength = 0.0  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
    perspective_prior_strength_string = saveresults.convert_array_to_string(perspective_prior_strength)
    selection_type = 'ca_with_parent'

    if context_generation == 'random':
        plot_file_title = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_alpha_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    elif context_generation == 'only_helpful' or context_generation == 'optimal':
        plot_file_title = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_alpha_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type


    inf_values_first_generation_all_ambiguous = np.array([0.33 for i in range(pop_size)])



    language_samples = []


    # plot_histogram_diff_language_samples(plot_file_path, 'none_'+plot_file_title, 'Proportions of language types in population', language_samples, pop_size)



    language_samples = [inf_values_first_generation_all_ambiguous]


    # plot_histogram_diff_language_samples(plot_file_path, '_one_'+plot_file_title, 'Proportions of language types in population', language_samples, pop_size)


    language_samples = [inf_values_first_generation_all_ambiguous, inf_values_last_generation_select_ca]

    # plot_histogram_diff_language_samples(plot_file_path, '_two_'+plot_file_title, 'Proportions of language types in population', language_samples, pop_size)


    language_samples = [inf_values_first_generation_all_ambiguous, inf_values_last_generation_select_ca, inf_values_last_generation_select_none]

    # plot_histogram_diff_language_samples(plot_file_path, '_all_'+plot_file_title, 'Proportions of language types in population', language_samples, pop_size)


    plot_title = 'Avg. informativeness of population over generations'

    y_axis_label = 'Avg. informativeness of population'


    # plot_percentile_timecourses_three_priors_three_conds(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, high_cut_off, percentiles_informativity_over_gens_exo_select_none, percentiles_informativity_over_gens_exo_select_ca, percentiles_informativity_over_gens_exo_select_p_taking, percentiles_informativity_over_gens_neut_select_none, percentiles_informativity_over_gens_neut_select_ca, percentiles_informativity_over_gens_neut_select_p_taking, percentiles_informativity_over_gens_ego_select_none, percentiles_informativity_over_gens_ego_select_ca, percentiles_informativity_over_gens_ego_select_p_taking)
    #


    ## Exocentric:

    plot_title = 'Allocentric Perspective Prior'

    # plot_mean_conf_invs_three_conds(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, high_cut_off, mean_informativity_over_gens_exo_select_none, conf_invs_informativity_over_gens_exo_select_none, mean_informativity_over_gens_exo_select_ca, conf_invs_informativity_over_gens_exo_select_ca, mean_informativity_over_gens_exo_select_p_taking, conf_invs_informativity_over_gens_exo_select_p_taking)

    # plot_percentiles_three_conds(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, high_cut_off, percentiles_informativity_over_gens_exo_select_none, percentiles_informativity_over_gens_exo_select_ca, percentiles_informativity_over_gens_exo_select_p_taking)




    ## Uniform:



    ### Select None:


    perspective_prior_type = 'neutral'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
    perspective_prior_strength = 0.0  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
    perspective_prior_strength_string = saveresults.convert_array_to_string(perspective_prior_strength)
    selection_type = 'none'


    if context_generation == 'random':
        plot_file_title = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_alpha_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    elif context_generation == 'only_helpful' or context_generation == 'optimal':
        plot_file_title = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_alpha_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type

    #
    # plot_title = 'No selection'
    plot_title = 'Informativeness over generations'


    # plot_mean_conf_invs_three_conds(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, high_cut_off, mean_informativity_over_gens_neut_select_none, conf_invs_informativity_over_gens_neut_select_none, mean_informativity_over_gens_neut_select_ca, conf_invs_informativity_over_gens_neut_select_ca, mean_informativity_over_gens_neut_select_p_taking, conf_invs_informativity_over_gens_neut_select_p_taking)


    # plot_percentiles_three_conds(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, high_cut_off, percentiles_informativity_over_gens_neut_select_none, percentiles_informativity_over_gens_neut_select_ca, percentiles_informativity_over_gens_neut_select_p_taking)


    # plot_percentiles_one_cond(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, low_cut_off, high_cut_off, percentiles_informativity_over_gens_neut_select_none, selection_type, baseline_inf, max_inf)
    #

    plot_file_title = 'blank_select_'+selection_type

    # percentiles_blank = np.full_like(percentiles_informativity_over_gens_neut_select_none, -1.)
    #
    # plot_percentiles_one_cond(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, low_cut_off, high_cut_off, percentiles_blank, selection_type, baseline_inf, max_inf)
    #



    ### Select CA:

    perspective_prior_type = 'neutral'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
    perspective_prior_strength = 0.0  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
    perspective_prior_strength_string = saveresults.convert_array_to_string(perspective_prior_strength)
    selection_type = 'ca_with_parent'

    if context_generation == 'random':
        plot_file_title = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_alpha_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    elif context_generation == 'only_helpful' or context_generation == 'optimal':
        plot_file_title = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_alpha_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type


    # plot_title = 'Selection for communication'
    plot_title = 'Informativeness over generations'


    # plot_mean_conf_invs_three_conds(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, high_cut_off, mean_informativity_over_gens_neut_select_none, conf_invs_informativity_over_gens_neut_select_none, mean_informativity_over_gens_neut_select_ca, conf_invs_informativity_over_gens_neut_select_ca, mean_informativity_over_gens_neut_select_p_taking, conf_invs_informativity_over_gens_neut_select_p_taking)


    # plot_percentiles_three_conds(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, high_cut_off, percentiles_informativity_over_gens_neut_select_none, percentiles_informativity_over_gens_neut_select_ca, percentiles_informativity_over_gens_neut_select_p_taking)

    #
    # plot_percentiles_one_cond(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, low_cut_off, high_cut_off, percentiles_informativity_over_gens_neut_select_ca, selection_type, baseline_inf, max_inf)
    #

    plot_file_title = 'blank_select_'+selection_type

    # percentiles_blank = np.full_like(percentiles_informativity_over_gens_neut_select_ca, -1.)
    #
    # plot_percentiles_one_cond(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, low_cut_off, high_cut_off, percentiles_blank, selection_type, baseline_inf, max_inf)






    ### Select p-taking:


    perspective_prior_type = 'neutral'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
    perspective_prior_strength = 0.0  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
    perspective_prior_strength_string = saveresults.convert_array_to_string(perspective_prior_strength)
    selection_type = 'p_taking'


    if context_generation == 'random':
        plot_file_title = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_alpha_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    elif context_generation == 'only_helpful' or context_generation == 'optimal':
        plot_file_title = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_alpha_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type

    #
    # plot_title = 'Selection on perspective-taking'
    plot_title = 'Informativeness over generations'


    # plot_mean_conf_invs_three_conds(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, high_cut_off, mean_informativity_over_gens_neut_select_none, conf_invs_informativity_over_gens_neut_select_none, mean_informativity_over_gens_neut_select_ca, conf_invs_informativity_over_gens_neut_select_ca, mean_informativity_over_gens_neut_select_p_taking, conf_invs_informativity_over_gens_neut_select_p_taking)


    # plot_percentiles_three_conds(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, high_cut_off, percentiles_informativity_over_gens_neut_select_none, percentiles_informativity_over_gens_neut_select_ca, percentiles_informativity_over_gens_neut_select_p_taking)


    # plot_percentiles_one_cond(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, low_cut_off, high_cut_off, percentiles_informativity_over_gens_neut_select_p_taking, selection_type, baseline_inf, max_inf)



    plot_file_title = 'blank_select_'+selection_type

    # percentiles_blank = np.full_like(percentiles_informativity_over_gens_neut_select_p_taking, -1.)
    #
    # plot_percentiles_one_cond(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, low_cut_off, high_cut_off, percentiles_blank, selection_type, baseline_inf, max_inf)






    ## Egocentric:




    ### Select None:


    perspective_prior_type = 'egocentric'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
    perspective_prior_strength = 0.9  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
    perspective_prior_strength_string = saveresults.convert_array_to_string(perspective_prior_strength)
    selection_type = 'none'


    if context_generation == 'random':
        plot_file_title = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_alpha_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    elif context_generation == 'only_helpful' or context_generation == 'optimal':
        plot_file_title = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_alpha_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type

    #
    # plot_title = 'No selection'
    plot_title = 'Informativeness over generations'


    # plot_mean_conf_invs_three_conds(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, high_cut_off, mean_informativity_over_gens_ego_select_none, conf_invs_informativity_over_gens_ego_select_none, mean_informativity_over_gens_ego_select_ca, conf_invs_informativity_over_gens_ego_select_ca, mean_informativity_over_gens_ego_select_p_taking, conf_invs_informativity_over_gens_ego_select_p_taking)
    #
    #
    # plot_percentiles_three_conds(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, high_cut_off, percentiles_informativity_over_gens_ego_select_none, percentiles_informativity_over_gens_ego_select_ca, percentiles_informativity_over_gens_ego_select_p_taking)


    plot_percentiles_one_cond(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, low_cut_off, high_cut_off, percentiles_informativity_over_gens_ego_select_none, selection_type, baseline_inf, max_inf)


    plot_file_title = 'blank_select_'+selection_type

    percentiles_blank = np.full_like(percentiles_informativity_over_gens_ego_select_none, -1.)

    # plot_percentiles_one_cond(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, low_cut_off, high_cut_off, percentiles_blank, selection_type, baseline_inf, max_inf)




    ### Select CA:

    perspective_prior_type = 'egocentric'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
    perspective_prior_strength = 0.9  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
    perspective_prior_strength_string = saveresults.convert_array_to_string(perspective_prior_strength)
    selection_type = 'ca_with_parent'

    if context_generation == 'random':
        plot_file_title = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_alpha_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    elif context_generation == 'only_helpful' or context_generation == 'optimal':
        plot_file_title = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_alpha_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type

    #
    # plot_title = 'Selection for communication'
    plot_title = 'Informativeness over generations'


    # plot_mean_conf_invs_three_conds(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, high_cut_off, mean_informativity_over_gens_ego_select_none, conf_invs_informativity_over_gens_ego_select_none, mean_informativity_over_gens_ego_select_ca, conf_invs_informativity_over_gens_ego_select_ca, mean_informativity_over_gens_ego_select_p_taking, conf_invs_informativity_over_gens_ego_select_p_taking)
    #
    #
    # plot_percentiles_three_conds(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, high_cut_off, percentiles_informativity_over_gens_ego_select_none, percentiles_informativity_over_gens_ego_select_ca, percentiles_informativity_over_gens_ego_select_p_taking)


    plot_percentiles_one_cond(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, low_cut_off, high_cut_off, percentiles_informativity_over_gens_ego_select_ca, selection_type, baseline_inf, max_inf)


    plot_file_title = 'blank_select_'+selection_type

    percentiles_blank = np.full_like(percentiles_informativity_over_gens_ego_select_ca, -1.)

    # plot_percentiles_one_cond(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, low_cut_off, high_cut_off, percentiles_blank, selection_type, baseline_inf, max_inf)






    ### Select p-taking:


    perspective_prior_type = 'egocentric'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
    perspective_prior_strength = 0.9  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
    perspective_prior_strength_string = saveresults.convert_array_to_string(perspective_prior_strength)
    selection_type = 'p_taking'


    if context_generation == 'random':
        plot_file_title = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    elif context_generation == 'only_helpful' or context_generation == 'optimal':
        plot_file_title = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type


    # plot_title = 'Selection on perspective-taking'
    plot_title = 'Informativeness over generations'


    # plot_mean_conf_invs_three_conds(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, high_cut_off, mean_informativity_over_gens_ego_select_none, conf_invs_informativity_over_gens_ego_select_none, mean_informativity_over_gens_ego_select_ca, conf_invs_informativity_over_gens_ego_select_ca, mean_informativity_over_gens_ego_select_p_taking, conf_invs_informativity_over_gens_ego_select_p_taking)
    #
    #
    # plot_percentiles_three_conds(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, high_cut_off, percentiles_informativity_over_gens_ego_select_none, percentiles_informativity_over_gens_ego_select_ca, percentiles_informativity_over_gens_ego_select_p_taking)


    plot_percentiles_one_cond(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, low_cut_off, high_cut_off, percentiles_informativity_over_gens_ego_select_p_taking, selection_type, baseline_inf, max_inf)




    plot_file_title = 'blank_select_'+selection_type

    percentiles_blank = np.full_like(percentiles_informativity_over_gens_ego_select_p_taking, -1.)

    # plot_percentiles_one_cond(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, low_cut_off, high_cut_off, percentiles_blank, selection_type, baseline_inf, max_inf)
    #




    perspective_prior_type = 'neutral'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'

    if context_generation == 'random':
        plot_file_title = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    elif context_generation == 'only_helpful' or context_generation == 'optimal':
        plot_file_title = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type


    plot_title = 'Uniform perspective prior'

    # plot_percentiles_three_cond(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, low_cut_off, high_cut_off, percentiles_informativity_over_gens_neut_select_none, percentiles_informativity_over_gens_neut_select_p_taking, percentiles_informativity_over_gens_neut_select_ca, baseline_inf, max_inf, legend=legend)


    perspective_prior_type = 'egocentric'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'

    if context_generation == 'random':
        plot_file_title = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    elif context_generation == 'only_helpful' or context_generation == 'optimal':
        plot_file_title = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type


    plot_title = 'Egocentric perspective prior'

    # plot_percentiles_three_cond(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, low_cut_off, high_cut_off, percentiles_informativity_over_gens_ego_select_none, percentiles_informativity_over_gens_ego_select_p_taking, percentiles_informativity_over_gens_ego_select_ca, baseline_inf, max_inf, legend=legend)

