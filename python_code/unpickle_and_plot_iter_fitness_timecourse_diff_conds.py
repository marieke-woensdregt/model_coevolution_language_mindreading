__author__ = 'Marieke Woensdregt'


import numpy as np
import pickle
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

import hypspace
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
n_contexts = 120  # The number of contexts that the learner gets to see.
speaker_order_type = 'random'  # This can be set to either 'random', 'random_equal' (for random order with making sure that each speaker gets an equal amount of utterances) 'same_first' (first portion of input comes from same perspective, second portion of input from opposite perspective), 'same_first_equal' (where both speakers get to produce the exact same amount of utterances), 'opp_first' (vice versa) or 'opp_first_equal'
first_input_stage_ratio = 0.5  # This is the ratio of contexts that will make up the first input stage (see parameter 'speaker_order_type' above)



# 1.8: The parameters that determine the type and number of simulations that are run:

#FIXME: In the current implementation 'half_ambiguous_lex' can have different instantiations. Therefore, if we run a simulation where there are different lexicon_type_probs, we will want the run_type to be 'population_same_pop' to make sure that all the 'half ambiguous' speakers do have the SAME half ambiguous lexicon.
run_type = 'iter'  # This parameter determines whether the learner communicates with only one speaker ('dyadic') or with a population ('population_diff_pop' if there are no speakers with the 'half_ambiguous' lexicon type, 'population_same_pop' if there are, 'population_same_pop_dist_learner' if the learner can distinguish between different speakers), or whether we do an iterated learning model ('iter')

communication_type = 'lex_n_p'  # This can be set to either 'lex_only', 'lex_n_context', 'lex_n_p' or 'prag'
ca_measure_type = 'comp_only'  # This can be set to either "comp_n_prod" or "comp_only"
n_interactions = 6  # The number of interactions used to calculate communicative accuracy

selection_type = 'ca_with_parent'  # This can be set to either 'none' or 'p_taking'
selection_weighting = 'none'  # This is a factor with which the fitness of the agents (determined as the probability they assign to the correct perspective hypothesis) is multiplied and then exponentiated in order to weight the relative agent fitness (which in turn determines the probability of becoming a teacher for the next generation). A value of 0. implements neutral selection. A value of 1.0 creates weighting where the fitness is pretty much equal to relative posterior probability on correct p hyp), and the higher the value, the more skewed the weighting in favour of agents with better perspective-taking.
if isinstance(selection_weighting, float):
    selection_weight_string = str(np.int(selection_weighting))
else:
    selection_weight_string = selection_weighting


turnover_type = 'whole_pop'  # The type of turnover with which the population is replaced (for the iteration simulation). This can be set to either 'chain' (one agent at a time) or 'whole_pop' (entire population at once)
n_iterations = 500  # The number of iterations (i.e. new agents in the case of 'chain', generations in the case of 'whole_pop')
cut_off_point = 300
n_runs = 1  # The number of runs of the simulation


recording = 'minimal'  # This can be set to either 'everything' or 'minimal'

which_hyps_on_graph = 'all_hyps'  # This is used in the plot_post_probs_over_lex_hyps() function, and can be set to either 'all_hyps' or 'lex_hyps_only'

lex_measure = 'ca'  # This can be set to either 'mi' for mutual information or 'ca' for communicative accuracy (of the lexicon with itself)

posterior_threshold = 0.95  # This parameter is used by the 'measur_n_data_points_for_p_learning' function

decoupling = True  # This can be set to either True or False. It determines whether genetic and cultural inheritance are coupled (i.e. from the same cultural parent) or decoupled.


n_copies = 200
copy_specification = ''

low_cut_off = 0
high_cut_off = n_iterations
#######################################################################################################################



def get_mean_percentiles_avg_fitness(pickle_directory, n_runs, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type, measure_type):
    perspective_prior_strength_string = saveresults.convert_array_to_string(perspective_prior_strength)
    ##
    if selection_type == 'none' or selection_type == 'p_taking':
        if context_generation == 'random':
            filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    elif selection_type == 'ca_with_parent':
        if context_generation == 'random':
            filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    ##
    if measure_type == 'fitness' and selection_type != 'none':
        percentiles_n_mean = pickle.load(open(pickle_directory+'fitness_tmcrse_'+filename_short+'.p', 'rb'))
    elif measure_type == 'communication_success':
        percentiles_n_mean = pickle.load(
            open(pickle_directory + 'comm_success_tmcrse_' + filename_short + '.p', 'rb'))
    elif measure_type == 'p_taking_success':
        percentiles_n_mean = pickle.load(
            open(pickle_directory + 'p_taking_success_tmcrse_' + filename_short + '.p', 'rb'))
    mean_over_gens = percentiles_n_mean['mean_fitness_over_gens']
    conf_invs_over_gens = percentiles_n_mean['conf_invs_fitness_over_gens']
    percentiles_over_gens = percentiles_n_mean['percentiles_fitness_over_gens']
    return mean_over_gens, conf_invs_over_gens, percentiles_over_gens




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
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_yticks(np.arange(0.0, 1.05, 0.5))
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
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_yticks(np.arange(0.0, 1.05, 0.5))
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
        ax3.set_ylim(-0.05, 1.05)
        ax3.set_yticks(np.arange(0.0, 1.05, 0.5))
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
    plt.savefig(plot_file_path+'Plot_Fitness_Perc_Diff_Conds_'+plot_file_title+'_hb_'+str(high_cut_off)+'.png')
    plt.show()




def plot_mean_conf_invs_three_conds(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, high_cut_off, mean_select_none, conf_invs_select_none, mean_select_ca, conf_invs_select_ca, mean_select_p_taking, conf_invs_select_p_taking, max_inf='None'):
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    sns.set(font_scale=text_size)
    color_no_selection = sns.color_palette()[0]
    color_select_ca = sns.color_palette()[3]
    color_select_p_taking = sns.color_palette()[4]
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots()
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks(np.arange(0.0, 1.05, 0.2))
        if max_inf != 'None':
            ax.axhline(max_inf, label='Max. Informativeness', color='black', linestyle='--')
        #### CONDITION 1: No Selection:
        ax.plot(np.arange(high_cut_off), mean_select_none[0:high_cut_off], label='No Selection', color=color_no_selection)
        ax.fill_between(np.arange(high_cut_off), conf_invs_select_none[0][0:high_cut_off], conf_invs_select_none[1][0:high_cut_off], facecolor=color_no_selection, alpha=0.5)
        #### CONDITION 2: Selection on CS:
        ax.plot(np.arange(high_cut_off), mean_select_ca[0:high_cut_off], label='Select CS', color=color_select_ca)
        ax.fill_between(np.arange(high_cut_off), conf_invs_select_ca[0][0:high_cut_off], conf_invs_select_ca[1][0:high_cut_off], facecolor=color_select_ca, alpha=0.5)
        #### CONDITION 3: Selection on P-taking:
        ax.plot(np.arange(high_cut_off), mean_select_p_taking[0:high_cut_off], label='Select P-taking', color=color_select_p_taking)
        ax.fill_between(np.arange(high_cut_off), conf_invs_select_p_taking[0][0:high_cut_off], conf_invs_select_p_taking[1][0:high_cut_off], facecolor=color_select_p_taking, alpha=0.5)
        # Shrink current axis by 28%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.72, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    plt.xlabel('Generations')
    plt.ylabel(y_axis_label)
    plt.title(plot_title)
    plt.savefig(plot_file_path+'Plot_Fitness_Mean_CI_'+plot_file_title+'_hb_'+str(high_cut_off)+'.png')
    plt.show()



def plot_percentiles_three_conds(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, high_cut_off, percentiles_select_none, percentiles_select_ca, percentiles_select_p_taking, max_inf='None'):
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    sns.set(font_scale=text_size)
    color_no_selection = sns.color_palette()[0]
    color_select_ca = sns.color_palette()[3]
    color_select_p_taking = sns.color_palette()[4]
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots()
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks(np.arange(0.0, 1.05, 0.2))
        if max_inf != 'None':
            ax.axhline(max_inf, label='Max. Informativeness', color='black', linestyle='--')
        #### CONDITION 1: No Selection:
        ax.plot(np.arange(high_cut_off), percentiles_select_none[1][0:high_cut_off], label='No Selection', color=color_no_selection)
        ax.fill_between(np.arange(high_cut_off), percentiles_select_none[0][0:high_cut_off], percentiles_select_none[2][0:high_cut_off], facecolor=color_no_selection, alpha=0.5)
        #### CONDITION 2: Selection on CS:
        ax.plot(np.arange(high_cut_off), percentiles_select_ca[1][0:high_cut_off], label='Select CS', color=color_select_ca)
        ax.fill_between(np.arange(high_cut_off), percentiles_select_ca[0][0:high_cut_off], percentiles_select_ca[2][0:high_cut_off], facecolor=color_select_ca, alpha=0.5)
        #### CONDITION 3: Selection on P-taking:
        ax.plot(np.arange(high_cut_off), percentiles_select_p_taking[1][0:high_cut_off], label='Select P-taking', color=color_select_p_taking)
        ax.fill_between(np.arange(high_cut_off), percentiles_select_p_taking[0][0:high_cut_off], percentiles_select_p_taking[2][0:high_cut_off], facecolor=color_select_p_taking, alpha=0.5)
        # Shrink current axis by 28%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.72, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    plt.xlabel('Generations')
    plt.ylabel(y_axis_label)
    plt.title(plot_title)
    plt.savefig(plot_file_path+'Plot_Fitness_Perc_'+plot_file_title+'_hb_'+str(high_cut_off)+'.png')
    plt.show()



def plot_percentiles_one_cond(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, low_cut_off, high_cut_off, percentiles, selection_type, max_fitness='None'):
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    sns.set(font_scale=text_size)
    if selection_type == 'none':
        color = sns.color_palette()[0]
    elif selection_type == 'ca_with_parent':
        color = sns.color_palette()[3]
    elif selection_type == 'p_taking':
        color = sns.color_palette()[4]
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots()
        ax.set_ylim(-0.05, (max_fitness+0.05))
        ax.set_yticks(np.arange(0.0, (max_fitness+0.05), 0.1))
        ax.set_xlim(0, high_cut_off + 1)
        ax.set_xticks(np.arange(0, high_cut_off+1, high_cut_off/10))
        ax.axvline(low_cut_off, color='0.6', linestyle=':', label='burn-in')
        if max_fitness != 'None':
            ax.axhline(max_fitness, color='0.2', label='maximum')
        ax.plot(np.arange(high_cut_off), percentiles[1][0:high_cut_off], color=color, label='median')
        ax.fill_between(np.arange(high_cut_off), percentiles[0][0:high_cut_off], percentiles[2][0:high_cut_off], color=color, alpha=0.5, label='IQR')
        # Shrink current axis by 28%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    plt.xlabel('Generations', fontsize=18)
    plt.ylabel(y_axis_label, fontsize=18)
    plt.suptitle(plot_title, fontsize=20)
    plt.savefig(plot_file_path+'Plot_Fitness_Perc_'+plot_file_title+'_hb_'+str(high_cut_off)+'.png')
    plt.show()



#####################################################################################

if __name__ == "__main__":


    pickle_directory = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/Iteration/'

    max_fitness_select_ca = (np.power(0.95, 2)+np.power(0.05, 2))
    print ''
    print ''
    print "max_fitness_select_ca is:"
    print max_fitness_select_ca

    max_fitness_select_p_taking = 1.0
    print ''
    print ''
    print "max_fitness_select_p_taking is:"
    print max_fitness_select_p_taking


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
#     mean_fitness_over_gens_exo_select_ca, conf_invs_fitness_over_gens_exo_select_ca, percentiles_fitness_over_gens_exo_select_ca = get_mean_percentiles_avg_fitness(pickle_directory, n_runs, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type)
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
#     mean_fitness_over_gens_exo_select_p_taking, conf_invs_fitness_over_gens_exo_select_p_taking, percentiles_fitness_over_gens_exo_select_p_taking = get_mean_percentiles_avg_fitness(pickle_directory, n_runs, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type)
#
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
#     mean_fitness_over_gens_neut_select_ca, conf_invs_fitness_over_gens_neut_select_ca, percentiles_fitness_over_gens_neut_select_ca = get_mean_percentiles_avg_fitness(pickle_directory, n_runs, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type)
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
#     mean_fitness_over_gens_neut_select_p_taking, conf_invs_fitness_over_gens_neut_select_p_taking, percentiles_fitness_over_gens_neut_select_p_taking = get_mean_percentiles_avg_fitness(pickle_directory, n_runs, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type)
#
#
#


#
# #####################################################################################
# ## PRIOR 1: EGOCENTRIC
# ### CONDITION 2: NO selection
#
#     print ''
#     print ''
#     print 'This is prior 1: Egocentric'
#     print ''
#     print 'This is condition 1: NO selection'
#
#     perspective_prior_type = 'egocentric'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
#     perspective_prior_strength = 0.9  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
#     selection_type = 'none'
#
#
#     mean_communication_success_over_gens_ego_select_none, conf_invs_communication_success_over_gens_ego_select_none, percentiles_communication_success_over_gens_ego_select_none = get_mean_percentiles_avg_fitness(pickle_directory, n_runs, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type, 'communication_success')
#
#
#     mean_p_taking_success_over_gens_ego_select_none, conf_invs_p_taking_success_over_gens_ego_select_none, percentiles_p_taking_success_over_gens_ego_select_none = get_mean_percentiles_avg_fitness(pickle_directory, n_runs, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type, 'p_taking_success')
#
#
#
#

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
    if pragmatic_level == 'literal':
        pragmatic_level = 'perspective-taking'

    mean_fitness_over_gens_ego_select_ca, conf_invs_fitness_over_gens_ego_select_ca, percentiles_fitness_over_gens_ego_select_ca = get_mean_percentiles_avg_fitness(pickle_directory, n_runs, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type, 'fitness')

    # mean_communication_success_over_gens_ego_select_ca, conf_invs_communication_success_over_gens_ego_select_ca, percentiles_communication_success_over_gens_ego_select_ca = get_mean_percentiles_avg_fitness(pickle_directory, n_runs, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type, 'communication_success')
    #
    # mean_p_taking_success_over_gens_ego_select_ca, conf_invs_p_taking_success_over_gens_ego_select_ca, percentiles_p_taking_success_over_gens_ego_select_ca = get_mean_percentiles_avg_fitness(pickle_directory, n_runs, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type, 'p_taking_success')



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
    if pragmatic_level == 'perspective-taking':
        pragmatic_level = 'literal'

    mean_fitness_over_gens_ego_select_p_taking, conf_invs_fitness_over_gens_ego_select_p_taking, percentiles_fitness_over_gens_ego_select_p_taking = get_mean_percentiles_avg_fitness(pickle_directory, n_runs, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type, 'fitness')

    # mean_communication_success_over_gens_ego_select_p_taking, conf_invs_communication_success_over_gens_ego_select_p_taking, percentiles_communication_success_over_gens_ego_select_p_taking = get_mean_percentiles_avg_fitness(pickle_directory, n_runs, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type, 'communication_success')
    #
    # mean_p_taking_success_over_gens_ego_select_p_taking, conf_invs_p_taking_success_over_gens_ego_select_p_taking, percentiles_p_taking_success_over_gens_ego_select_p_taking = get_mean_percentiles_avg_fitness(pickle_directory, n_runs, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type, 'p_taking_success')
    #


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



    plot_title = 'Avg. fitness of population over generations'

    y_axis_label = 'Avg. fitness of population'

    text_size = 1.6

    # plot_percentile_timecourses_three_priors_three_conds(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, high_cut_off, percentiles_fitness_over_gens_exo_select_none, percentiles_fitness_over_gens_exo_select_ca, percentiles_fitness_over_gens_exo_select_p_taking, percentiles_fitness_over_gens_neut_select_none, percentiles_fitness_over_gens_neut_select_ca, percentiles_fitness_over_gens_neut_select_p_taking, percentiles_fitness_over_gens_ego_select_none, percentiles_fitness_over_gens_ego_select_ca, percentiles_fitness_over_gens_ego_select_p_taking)
    #


    ## Exocentric:

    plot_title = 'Allocentric Perspective Prior'

    # plot_mean_conf_invs_three_conds(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, high_cut_off, mean_fitness_over_gens_exo_select_none, conf_invs_fitness_over_gens_exo_select_none, mean_fitness_over_gens_exo_select_ca, conf_invs_fitness_over_gens_exo_select_ca, mean_fitness_over_gens_exo_select_p_taking, conf_invs_fitness_over_gens_exo_select_p_taking)

    # plot_percentiles_three_conds(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, high_cut_off, percentiles_fitness_over_gens_exo_select_none, percentiles_fitness_over_gens_exo_select_ca, percentiles_fitness_over_gens_exo_select_p_taking)





    ### Select CA:

    perspective_prior_type = 'neutral'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
    perspective_prior_strength = 0.0  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
    perspective_prior_strength_string = saveresults.convert_array_to_string(perspective_prior_strength)
    selection_type = 'ca_with_parent'

    if context_generation == 'random':
        plot_file_title = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    elif context_generation == 'only_helpful' or context_generation == 'optimal':
        plot_file_title = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type



    # plot_title = 'Selection for communication'
    plot_title = 'Fitness over generations'

    # plot_mean_conf_invs_three_conds(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, high_cut_off, mean_fitness_over_gens_neut_select_none, conf_invs_fitness_over_gens_neut_select_none, mean_fitness_over_gens_neut_select_ca, conf_invs_fitness_over_gens_neut_select_ca, mean_fitness_over_gens_neut_select_p_taking, conf_invs_fitness_over_gens_neut_select_p_taking)


    # plot_percentiles_three_conds(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, high_cut_off, percentiles_fitness_over_gens_neut_select_none, percentiles_fitness_over_gens_neut_select_ca, percentiles_fitness_over_gens_neut_select_p_taking)


    # plot_percentiles_one_cond(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, low_cut_off, high_cut_off, percentiles_fitness_over_gens_neut_select_ca, selection_type, max_fitness_select_ca)
    #



    # plot_file_title = 'blank_select_'+selection_type
    #
    # percentiles_blank = np.full_like(percentiles_fitness_over_gens_neut_select_ca, -1.)
    #
    # plot_percentiles_one_cond(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, low_cut_off, high_cut_off, percentiles_blank, selection_type, max_fitness_select_ca)




    ### Select p-taking:


    perspective_prior_type = 'neutral'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
    perspective_prior_strength = 0.0  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
    perspective_prior_strength_string = saveresults.convert_array_to_string(perspective_prior_strength)
    selection_type = 'p_taking'


    if context_generation == 'random':
        plot_file_title = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    elif context_generation == 'only_helpful' or context_generation == 'optimal':
        plot_file_title = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type


    # plot_title = 'Selection on perspective-taking'
    plot_title = 'Fitness over generations'

    # plot_mean_conf_invs_three_conds(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, high_cut_off, mean_fitness_over_gens_neut_select_none, conf_invs_fitness_over_gens_neut_select_none, mean_fitness_over_gens_neut_select_ca, conf_invs_fitness_over_gens_neut_select_ca, mean_fitness_over_gens_neut_select_p_taking, conf_invs_fitness_over_gens_neut_select_p_taking)


    # plot_percentiles_three_conds(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, high_cut_off, percentiles_fitness_over_gens_neut_select_none, percentiles_fitness_over_gens_neut_select_ca, percentiles_fitness_over_gens_neut_select_p_taking)
    #

    # plot_percentiles_one_cond(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, low_cut_off, high_cut_off, percentiles_fitness_over_gens_neut_select_p_taking, selection_type, max_fitness_select_p_taking)




    # plot_file_title = 'blank_select_'+selection_type
    #
    # percentiles_blank = np.full_like(percentiles_fitness_over_gens_neut_select_p_taking, -1.)
    #
    # plot_percentiles_one_cond(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, low_cut_off, high_cut_off, percentiles_blank, selection_type, max_fitness_select_p_taking)






    ## Egocentric:




    ### Select none:
    #
    #
    # perspective_prior_type = 'egocentric'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
    # perspective_prior_strength = 0.9  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
    # perspective_prior_strength_string = saveresults.convert_array_to_string(perspective_prior_strength)
    # selection_type = 'none'
    #
    #
    # if context_generation == 'random':
    #     plot_file_title = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    # elif context_generation == 'only_helpful' or context_generation == 'optimal':
    #     plot_file_title = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    #
    #
    #
    # # plot_title = 'Selection on perspective-taking'
    # plot_title = 'Fitness over generations'
    #
    # # plot_mean_conf_invs_three_conds(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, high_cut_off, mean_fitness_over_gens_ego_select_none, conf_invs_fitness_over_gens_ego_select_none, mean_fitness_over_gens_ego_select_ca, conf_invs_fitness_over_gens_ego_select_ca, mean_fitness_over_gens_ego_select_p_taking, conf_invs_fitness_over_gens_ego_select_p_taking)
    #
    #
    # # plot_percentiles_three_conds(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, high_cut_off, percentiles_fitness_over_gens_ego_select_none, percentiles_fitness_over_gens_ego_select_ca, percentiles_fitness_over_gens_ego_select_p_taking)
    # #
    #
    #
    # plot_file_title_full = plot_file_title + '_comm_success'
    # selection_type == 'ca_with_parent'
    #
    # plot_percentiles_one_cond(plot_file_path, plot_file_title_full, plot_title, y_axis_label, text_size, low_cut_off, high_cut_off, percentiles_communication_success_over_gens_ego_select_none, selection_type, max_fitness_select_ca)
    #
    #
    # plot_file_title_full = plot_file_title + '_p_taking_success'
    # selection_type == 'p_taking'
    #
    # plot_percentiles_one_cond(plot_file_path, plot_file_title_full, plot_title, y_axis_label, text_size, low_cut_off, high_cut_off, percentiles_p_taking_success_over_gens_ego_select_none, selection_type, max_fitness_select_p_taking)
    #
    #
    # # plot_file_title = 'blank_select_'+selection_type
    # #
    # # percentiles_blank = np.full_like(percentiles_fitness_over_gens_ego_select_p_taking, -1.)
    # #
    # # plot_percentiles_one_cond(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, low_cut_off, high_cut_off, percentiles_blank, selection_type, max_fitness_select_p_taking)
    #


    ### Select CA:

    perspective_prior_type = 'egocentric'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
    perspective_prior_strength = 0.9  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
    perspective_prior_strength_string = saveresults.convert_array_to_string(perspective_prior_strength)
    selection_type = 'ca_with_parent'

    if context_generation == 'random':
        plot_file_title = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    elif context_generation == 'only_helpful' or context_generation == 'optimal':
        plot_file_title = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type


    #
    # plot_title = 'Selection for communication'
    plot_title = 'Fitness over generations'

    # # plot_mean_conf_invs_three_conds(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, high_cut_off, mean_fitness_over_gens_ego_select_none, conf_invs_fitness_over_gens_ego_select_none, mean_fitness_over_gens_ego_select_ca, conf_invs_fitness_over_gens_ego_select_ca, mean_fitness_over_gens_ego_select_p_taking, conf_invs_fitness_over_gens_ego_select_p_taking)
    #
    #
    # # plot_percentiles_three_conds(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, high_cut_off, percentiles_fitness_over_gens_ego_select_none, percentiles_fitness_over_gens_ego_select_ca, percentiles_fitness_over_gens_ego_select_p_taking)
    # #
    #
    plot_file_title_full = plot_file_title

    plot_percentiles_one_cond(plot_file_path, plot_file_title_full, plot_title, y_axis_label, text_size, low_cut_off, high_cut_off, percentiles_fitness_over_gens_ego_select_ca, selection_type, max_fitness_select_ca)


    plot_file_title_full = plot_file_title+'_comm_success'

    plot_title = 'Communication success over generations'


    # plot_percentiles_one_cond(plot_file_path, plot_file_title_full, plot_title, y_axis_label, text_size, low_cut_off, high_cut_off, percentiles_communication_success_over_gens_ego_select_ca, selection_type, max_fitness_select_ca)



    selection_type = 'p_taking'

    plot_file_title_full = plot_file_title+'_p_taking_success'

    plot_title = 'Perspective-inference success over generations'

    # plot_percentiles_one_cond(plot_file_path, plot_file_title_full, plot_title, y_axis_label, text_size, low_cut_off, high_cut_off, percentiles_p_taking_success_over_gens_ego_select_ca, selection_type, max_fitness_select_p_taking)




    #
    # # plot_file_title = 'blank_select_'+selection_type
    # #
    # # percentiles_blank = np.full_like(percentiles_fitness_over_gens_ego_select_ca, -1.)
    # #
    # # plot_percentiles_one_cond(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, low_cut_off, high_cut_off, percentiles_blank, selection_type, max_fitness_select_ca)
    # #




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
    plot_title = 'Fitness over generations'

    # plot_mean_conf_invs_three_conds(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, high_cut_off, mean_fitness_over_gens_ego_select_none, conf_invs_fitness_over_gens_ego_select_none, mean_fitness_over_gens_ego_select_ca, conf_invs_fitness_over_gens_ego_select_ca, mean_fitness_over_gens_ego_select_p_taking, conf_invs_fitness_over_gens_ego_select_p_taking)


    # plot_percentiles_three_conds(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, high_cut_off, percentiles_fitness_over_gens_ego_select_none, percentiles_fitness_over_gens_ego_select_ca, percentiles_fitness_over_gens_ego_select_p_taking)
    #

    plot_file_title_full = plot_file_title

    plot_percentiles_one_cond(plot_file_path, plot_file_title_full, plot_title, y_axis_label, text_size, low_cut_off, high_cut_off, percentiles_fitness_over_gens_ego_select_p_taking, selection_type, max_fitness_select_p_taking)


    plot_file_title_full = plot_file_title + '_comm_success'
    selection_type = 'ca_with_parent'

    plot_title = 'Communication success over generations'

    # plot_percentiles_one_cond(plot_file_path, plot_file_title_full, plot_title, y_axis_label, text_size, low_cut_off, high_cut_off, percentiles_communication_success_over_gens_ego_select_p_taking, selection_type, max_fitness_select_ca)


    plot_file_title_full = plot_file_title + '_p_taking_success'
    selection_type = 'p_taking'

    plot_title = 'Perspective-inference success over generations'

    # plot_percentiles_one_cond(plot_file_path, plot_file_title_full, plot_title, y_axis_label, text_size, low_cut_off, high_cut_off, percentiles_p_taking_success_over_gens_ego_select_p_taking, selection_type, max_fitness_select_p_taking)


    # plot_file_title = 'blank_select_'+selection_type
    #
    # percentiles_blank = np.full_like(percentiles_fitness_over_gens_ego_select_p_taking, -1.)
    #
    # plot_percentiles_one_cond(plot_file_path, plot_file_title, plot_title, y_axis_label, text_size, low_cut_off, high_cut_off, percentiles_blank, selection_type, max_fitness_select_p_taking)
    #

