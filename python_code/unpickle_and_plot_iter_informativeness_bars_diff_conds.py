__author__ = 'Marieke Woensdregt'


import lex
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plots
import pandas as pd

# np.set_printoptions(threshold=np.nan)


#######################################################################################################################
# STEP 2: THE PARAMETERS:


# 2.1: The parameters defining the lexicon size (and thus the number of meanings in the world):

n_meanings = 3  # The number of meanings
n_signals = 3  # The number of signals


# 2.2: The parameters defining the contexts and how they map to the agent's saliencies:


context_generation = 'optimal' # This can be set to either 'random', 'only_helpful' or 'optimal'
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



# 2.3: The parameters that determine the make-up of the population:

pop_size = 10
teacher_type = 'sng_teacher'  # This can be set to either 'sng_teacher' or 'multi_teacher'
agent_type = 'no_p_distinction'  # This can be set to either 'p_distinction' or 'no_p_distinction'. Determines whether the population is made up of DistinctionAgent objects or Agent objects (DistinctionAgents can learn different perspectives for different agents, Agents can only learn one perspective for all agents they learn from).


lexicon_types = ['optimal_lex', 'half_ambiguous_lex', 'fully_ambiguous_lex']  # The lexicon types that can be present in the population
#FIXME: In the current implementation 'half_ambiguous_lex' can have different instantiations. Therefore, if we run a simulation where there are different lexicon_type_probs, we will want the run_type to be 'population_same_pop' to make sure that all the 'half ambiguous' speakers do have the SAME half ambiguous lexicon.
lexicon_type_probs = np.array([0., 0., 1.])  # The ratios with which the different lexicon types will be present in the population
lexicon_type_probs_string = convert_array_to_string(lexicon_type_probs)  # Turns the lexicon type probs into a string in order to add it to file names


perspectives = np.array([0., 1.])  # The different perspectives that agents can have
perspective_probs = np.array([0., 1.])  # The ratios with which the different perspectives will be present in the population
perspective_probs_string = convert_array_to_string(perspective_probs)  # Turns the perspective probs into a string in order to add it to file names


learning_types = ['map', 'sample']  # The types of learning that the learners can do
learning_type_probs = np.array([0., 1.])  # The ratios with which the different learning types will be present in the population
learning_type_probs_string = convert_array_to_string(learning_type_probs)  # Turns the learning type probs into a string in order to add it to file names
learning_type_string = learning_types[np.where(learning_type_probs == 1.)[0]]


# 2.6: The parameters that determine the learner's hypothesis space:

perspective_hyps = np.array([0., 1.])  # The perspective hypotheses that the learner will consider (1D numpy array)

which_lexicon_hyps = 'all'  # Determines the set of lexicon hypotheses that the learner will consider. This can be set to either 'all', 'all_with_full_s_space' or 'only_optimal'
if which_lexicon_hyps == 'all':
    lexicon_hyps = create_all_lexicons(n_meanings, n_signals)  # The lexicon hypotheses that the learner will consider (1D numpy array)
elif which_lexicon_hyps == 'all_with_full_s_space':
    all_lexicon_hyps = create_all_lexicons(n_meanings, n_signals)
    lexicon_hyps = remove_subset_of_signals_lexicons(all_lexicon_hyps)  # The lexicon hypotheses that the learner will consider (1D numpy array)
elif which_lexicon_hyps == 'only_optimal':
    lexicon_hyps = create_all_optimal_lexicons(n_meanings, n_signals)  # The lexicon hypotheses that the learner will consider (1D numpy array)


if agent_type == 'no_p_distinction':
    hypothesis_space = list_hypothesis_space(perspective_hyps, lexicon_hyps)  # The full space of composite hypotheses that the learner will consider (2D numpy matrix with composite hypotheses on the rows, perspective hypotheses on column 0 and lexicon hypotheses on column 1)

elif agent_type == 'p_distinction':
    hypothesis_space = list_hypothesis_space_with_speaker_distinction(perspective_hyps, lexicon_hyps, pop_size)  # The full space of composite hypotheses that the learner will consider (2D numpy matrix with composite hypotheses on the rows, perspective hypotheses on column 0 and lexicon hypotheses on column 1)


# 2.7: The parameters that determine the learner's prior:

learner_perspective = 0.  # The learner's perspective

perspective_prior_type = 'egocentric'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
perspective_prior_strength = 0.9  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
perspective_prior_strength_string = convert_array_to_string(perspective_prior_strength)


lexicon_prior_type = 'neutral'  # This can be set to either 'neutral', 'ambiguous_fixed', 'half_ambiguous_fixed', 'expressivity_bias' or 'compressibility_bias'.
lexicon_prior_constant = 0.0  # Determines the strength of the lexicon prior, with small c creating a STRONG prior and large c creating a WEAK prior. (And with c = 1000 creating an almost uniform prior.) For the expressivity bias I chose c = 0.3, for the compressibility bias c = 0.0003.
lexicon_prior_constant_string = convert_array_to_string(lexicon_prior_constant)


# 2.5: The parameters that determine the amount of data_dict that the learner gets to see, and the amount of runs of the simulation:

n_utterances = 1  # This parameter determines how many signals the learner gets to observe in each context
n_contexts = 132 # The number of contexts that the learner gets to see.


# 2.7: The parameters that determine the type and number of simulations that are run:

#FIXME: In the current implementation 'half_ambiguous_lex' can have different instantiations. Therefore, if we run a simulation where there are different lexicon_type_probs, we will want the run_type to be 'population_same_pop' to make sure that all the 'half ambiguous' speakers do have the SAME half ambiguous lexicon.
run_type = 'iter'  # This parameter determines whether the learner communicates with only one speaker ('dyadic') or with a population ('population_diff_pop' if there are no speakers with the 'half_ambiguous' lexicon type, 'population_same_pop' if there are, 'population_same_pop_dist_learner' if the learner can distinguish between different speakers), or whether we do an iterated learning model ('iter')
turnover_type = 'whole_pop'  # The type of turnover with which the population is replaced (for the iteration simulation). This can be set to either 'chain' (one agent at a time) or 'whole_pop' (entire population at once)

communication_type = 'lex_only'  # This can be set to either 'lex_only', 'lex_n_context' or 'lex_n_p'
ca_measure_type = 'comp_n_prod'  # This can be set to either "comp_n_prod" or "comp_only"
n_interactions = 12  # The number of interactions used to calculate communicative accuracy

selection_type = 'none'  # This can be set to either 'none', 'p_taking', 'l_learning' or 'ca_with_parent'
selection_weighting = 'none'  # This is a factor with which the fitness of the agents (determined as the probability they assign to the correct perspective hypothesis) is multiplied and then exponentiated in order to weight the relative agent fitness (which in turn determines the probability of becoming a teacher for the next generation). A value of 0. implements neutral selection. A value of 1.0 creates weighting where the fitness is pretty much equal to relative posterior probability on correct p hyp), and the higher the value, the more skewed the weighting in favour of agents with better perspective-taking.
if isinstance(selection_weighting, float):
    selection_weight_string = str(np.int(selection_weighting))
else:
    selection_weight_string = selection_weighting

n_iterations = 1000  # The number of iterations (i.e. new agents in the case of 'chain', generations in the case of 'whole_pop')
report_every_i = 100  # Determines how often the progress is printed as the simulation runs
cut_off_point = 1000  # The cut_off point determines what number of iterations is used to plot on the graph.
burn_in = 400 # This determines from which generation onwards the grand means for the bar plot will be calculated.
n_runs = 1  # The number of runs of the simulation
report_every_r = 1  # Determines how often the progress is printed as the simulation runs

recording = 'minimal'  # This can be set to either 'everything' or 'minimal'. If this is set to 'everything' the posteriors distributions for every single time step (in developmental time) for every single agent of every single generation are recorded for every single run. If this is set to 'minimal' only the selected hypotheses per generation are recorded for every single run.


which_hyps_on_graph = 'lex_hyps_collapsed'  # This is used in the plot_post_probs_over_lex_hyps() function, and can be set to either 'all_hyps' or 'lex_hyps_only' or 'lex_hyps_collapsed'

n_copies = 10  # Specifies the number of copies of the results file
copy_specification = ''  # Can be set to e.g. '_c1' or simply to '' if there is only one copy

lex_measure = 'ca'

#######################################################################################################################





def unpickle_selected_hyps_matrix(directory, filename, n_copies):
    if n_copies == 1:
        multi_run_selected_hyps_per_generation_matrix_all_copies = np.zeros((1, n_runs, n_iterations, pop_size))
        pickle_filename_all_results = 'Results_'+filename+copy_specification
        results_dict = pickle.load(open(directory+pickle_filename_all_results+'.p', 'rb'))
        multi_run_selected_hyps_per_generation_matrix = results_dict[
            'multi_run_selected_hyps_per_generation_matrix']
        multi_run_selected_hyps_per_generation_matrix_all_copies[0] = multi_run_selected_hyps_per_generation_matrix
    elif n_copies > 1:
        multi_run_selected_hyps_per_generation_matrix_all_copies = np.zeros(((n_copies*n_runs), n_iterations, pop_size))
        counter = 0
        for c in range(1, n_copies+1):
            pickle_filename_all_results = 'Results_'+filename+'_c'+str(c)
            results_dict = pickle.load(open(directory+pickle_filename_all_results+'.p', 'rb'))
            for r in range(n_runs):
                multi_run_selected_hyps_per_generation_matrix = results_dict['multi_run_selected_hyps_per_generation_matrix'][r]
                multi_run_selected_hyps_per_generation_matrix_all_copies[counter] = multi_run_selected_hyps_per_generation_matrix
                counter += 1
        multi_run_selected_hyps_per_generation_matrix = multi_run_selected_hyps_per_generation_matrix_all_copies
    return multi_run_selected_hyps_per_generation_matrix


def get_selected_hyps_ordered(multi_run_selected_hyps_per_generation_matrix, hyp_order):
    selected_hyps_new_lex_order_all_runs = np.zeros_like(multi_run_selected_hyps_per_generation_matrix)
    for r in range(n_runs*n_copies):
        for i in range(n_iterations):
            for a in range(pop_size):
                this_agent_hyp = multi_run_selected_hyps_per_generation_matrix[r][i][a]
                if this_agent_hyp >= len(hyp_order):
                    this_agent_hyp = this_agent_hyp-len(hyp_order)
                new_order_index = np.argwhere(hyp_order == this_agent_hyp)
                selected_hyps_new_lex_order_all_runs[r][i][a] = new_order_index
    return selected_hyps_new_lex_order_all_runs




### CATEGORISING LEXICONS BY INFORMATIVENESS BELOW:


print ''
print ''
print "lexicon_hyps are:"
print lexicon_hyps
print "lexicon_hyps.shape are:"
print lexicon_hyps.shape



informativity_per_lexicon = lex.calc_ca_all_lexicons(lexicon_hyps, error, lex_measure)
print ''
print ''
print "informativity_per_lexicon is:"
print informativity_per_lexicon
print "informativity_per_lexicon.shape is:"
print informativity_per_lexicon.shape


argsort_informativity_per_lexicon = np.argsort(informativity_per_lexicon)
print ''
print ''
print "argsort_informativity_per_lexicon is:"
print argsort_informativity_per_lexicon
print "argsort_informativity_per_lexicon.shape is:"
print argsort_informativity_per_lexicon.shape


informativity_per_lexicon_sorted = np.round(informativity_per_lexicon[argsort_informativity_per_lexicon], decimals=2)
print ''
print ''
print "informativity_per_lexicon_sorted is:"
print informativity_per_lexicon_sorted
print "informativity_per_lexicon_sorted.shape is:"
print informativity_per_lexicon_sorted.shape

minimum_informativity = np.amin(informativity_per_lexicon_sorted)
print ''
print ''
print "minimum_informativity is:"
print minimum_informativity

min_info_indices = np.argwhere(informativity_per_lexicon_sorted==minimum_informativity)
print ''
print ''
min_info_indices = min_info_indices.flatten()
print "min_info_indices are:"
print min_info_indices

maximum_informativity = np.amax(informativity_per_lexicon_sorted)
print ''
print ''
print "maximum_informativity is:"
print maximum_informativity

max_info_indices = np.argwhere(informativity_per_lexicon_sorted==maximum_informativity)
print ''
print ''
max_info_indices = max_info_indices.flatten()
print "max_info_indices are:"
print max_info_indices

intermediate_info_indices = np.arange(min_info_indices[-1]+1, max_info_indices[0])
print ''
print ''
print "intermediate_info_indices are:"
print intermediate_info_indices

lexicon_hyps_sorted = lexicon_hyps[argsort_informativity_per_lexicon]
print ''
print ''
print "lexicon_hyps_sorted is:"
print lexicon_hyps_sorted
print "lexicon_hyps_sorted.shape is:"
print lexicon_hyps_sorted.shape






def plot_inf_bars_diff_conds(plot_file_path, plot_file_title, plot_title, p_prior, mean_avg_inf_over_gens_select_none, mean_avg_inf_over_gens_select_ca, mean_avg_inf_over_gens_select_p_taking, yerr_select_none, yerr_select_ca, yerr_select_p_taking, max_inf, text_size):
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    sns.set(font_scale=text_size)
    color_no_selection = sns.color_palette()[0]
    color_select_ca = sns.color_palette()[3]
    color_select_p_taking = sns.color_palette()[4]
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots()
        ax.set_ylim(-0.05, 0.85)
        ax.set_yticks(np.arange(0.0, 0.85, 0.2))
        ax.set_xticks([])
        ax.axhline(max_inf, label='Max. Informativeness', color='black', linestyle='--')
        ax.bar(0, mean_avg_inf_over_gens_select_none, label='No Selection', color=color_no_selection)
        ax.bar(1, mean_avg_inf_over_gens_select_ca, label='Select CS',  color=color_select_ca)
        ax.bar(2, mean_avg_inf_over_gens_select_p_taking, label='Select P-taking', color=color_select_p_taking)
        # ax.bar(mean_avg_inf_over_gens_select_none, label='No Selection', color=color_no_selection, yerr=yerr_select_none, error_kw=dict(ecolor='black', lw=1, capsize=5, capthick=1))
        # ax.bar(mean_avg_inf_over_gens_select_ca, label='Select CS',  color=color_select_ca, yerr=yerr_select_ca, error_kw=dict(ecolor='black', lw=1, capsize=5, capthick=1))
        # ax.bar(mean_avg_inf_over_gens_select_p_taking, label='Select P-taking', color=color_select_p_taking, yerr=yerr_select_p_taking, error_kw=dict(ecolor='black', lw=1, capsize=5, capthick=1))
        # Shrink current axis by 28%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    plt.xlabel('Selection condition')
    plt.ylabel('Mean informativeness')
    plt.title(plot_title)
    plt.gcf().subplots_adjust(bottom=0.15) # This makes room for the xlabel which would otherwise be cut off because of the rotation of the xticks
    plt.savefig(plot_file_path+'Plot_Inf_Bars_'+plot_file_title+'_'+p_prior+'_burnin_'+str(burn_in)+'.png')
    plt.show()



def plot_inf_bars_two_conds(plot_file_path, plot_file_title, plot_title, p_prior, grand_mean_inf_over_gens_ego_select_ca, grand_mean_inf_over_gens_ego_select_p_taking, baseline, max_inf, text_size):
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    sns.set(font_scale=text_size)
    color_no_selection = sns.color_palette()[0]
    color_select_ca = sns.color_palette()[3]
    color_select_p_taking = sns.color_palette()[4]
    ind = np.arange(2)
    width = 0.05
    x_labels = ['Select CA', 'Select P-taking']
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots()
        baseline = ax.axhline(baseline, color='0.6', linestyle='--')
        max_line = ax.axhline(max_inf, color='0.2')
        ax.set_ylim(-0.05, max_inf+0.05)
        ax.set_yticks(np.arange(0.0, (max_inf+0.05), 0.2))
        ax.set_xticks([0.1+0.05, 0.15+0.05+width])
        ax.set_xticklabels(x_labels)
        ax.set_xlabel('Selection type', labelpad=10)
        # ax.axhline(max_inf, label='Max. Informativeness', color='black', linestyle='--')
        ax.bar(0.1+0.025, grand_mean_inf_over_gens_ego_select_ca, width, label='Select CS', color=color_select_ca)
        ax.bar(0.15+0.025+width, grand_mean_inf_over_gens_ego_select_p_taking, width, label='Select P-taking',  color=color_select_p_taking)
        # ax.bar(mean_avg_inf_over_gens_select_none, label='No Selection', color=color_no_selection, yerr=yerr_select_none, error_kw=dict(ecolor='black', lw=1, capsize=5, capthick=1))
        # ax.bar(mean_avg_inf_over_gens_select_ca, label='Select CS',  color=color_select_ca, yerr=yerr_select_ca, error_kw=dict(ecolor='black', lw=1, capsize=5, capthick=1))
    #     # Shrink current axis by 28%
    #     box = ax.get_position()
    #     ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    #     # Put a legend to the right of the current axis
    #     ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    # plt.xlabel('Selection condition')
    if lex_measure == 'mi':
        plt.ylabel('Mean MI of languages in population')
        plt.title('Mean MI '+plot_title)
    elif lex_measure == 'ca':
        ax.set_ylabel('Mean ca of languages in population')
        ax.set_title('Mean ca '+plot_title)
    plt.gcf().subplots_adjust(bottom=0.15) # This makes room for the xlabel which would otherwise be cut off because of the rotation of the xticks
    plt.savefig(plot_file_path+'Plot_Inf_Bars_2_Conds_'+plot_file_title+'_'+p_prior+'_burnin_'+str(burn_in)+'.png')
    plt.show()


def plot_mean_informativeness_all_conds(plot_file_path, plot_file_title, grand_means_select_none, grand_means_select_ca, grand_means_select_p_taking, yerrs_select_none, yerrs_select_ca, yerrs_select_p_taking, baseline, max_inf, text_size):
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    sns.set(font_scale=text_size)
    color_no_selection = sns.color_palette()[0]
    color_select_ca = sns.color_palette()[3]
    color_select_p_taking = sns.color_palette()[4]
    p_prior_labels = ['Exocentric P Prior', 'Uniform P Prior', 'Egocentric P Prior']
    ind = np.arange(len(p_prior_labels))
    width = 0.25
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots()
        baseline = ax.axhline(baseline, color='0.6', linestyle='--')
        max_line = ax.axhline(max_inf, color='0.2')
        rects1 = ax.bar(ind+0.125, grand_means_select_none, width, color=color_no_selection)
        rects2 = ax.bar(ind+0.125+width, grand_means_select_ca, width, color=color_select_ca)
        rects3 = ax.bar(ind+0.125+(2*width), grand_means_select_p_taking, width, color=color_select_p_taking)
        # rects1 = ax.bar(ind, grand_means_select_none, width, color=color_no_selection, yerr=yerrs_select_none, error_kw=dict(ecolor='black', lw=1, capsize=5, capthick=1))
        # rects2 = ax.bar(ind+width, grand_means_select_ca, width, color=color_select_ca, yerr=yerrs_select_ca, error_kw=dict(ecolor='black', lw=1, capsize=5, capthick=1))
        # rects3 = ax.bar(ind+(2*width), grand_means_select_p_taking, width, color=color_select_p_taking, yerr=yerrs_select_p_taking, error_kw=dict(ecolor='black', lw=1, capsize=5, capthick=1))
        # Shrink current axis by 28%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.83, box.height])
        # Put a legend to the right of the current axis
        ax.legend((rects1[0], rects2[0], rects3[0]), ('No Selection', 'Select CS', 'Select P-taking'), loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(p_prior_labels)
    # ax.set_xlabel('Perspective Prior')
    if lex_measure == 'mi':
        ax.set_ylabel('Mean MI of languages in population')
        ax.set_title('Mean MI per P prior and selection condition')
    elif lex_measure == 'ca':
        ax.set_ylabel('Mean ca of languages in population')
        ax.set_title('Mean ca per P prior and selection condition')
    # plt.gcf().subplots_adjust(bottom=0.15) # This makes room for the xlabel which would otherwise be cut off because of the rotation of the xticks
    plt.savefig(plot_file_path+'Plot_Inf_Bars_Conds_'+lex_measure+'_'+plot_file_title+'_'+communication_type+'_'+ca_measure_type+'_burnin_'+str(burn_in)+'.png')
    plt.show()





#####################################################################################

if __name__ == "__main__":

    results_directory = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/Iteration/'


    inf_per_lex_full_hyp_space = informativity_per_lexicon_sorted
    for p in range(len(perspective_hyps)-1):
        inf_per_lex_full_hyp_space = np.hstack((inf_per_lex_full_hyp_space, informativity_per_lexicon_sorted))
    print "inf_per_lex_full_hyp_space is:"
    print inf_per_lex_full_hyp_space
    print "inf_per_lex_full_hyp_space.shape is:"
    print inf_per_lex_full_hyp_space.shape

    max_inf = np.amax(inf_per_lex_full_hyp_space)
    print ''
    print 'max_inf is:'
    print max_inf

    baseline_inf = np.mean(inf_per_lex_full_hyp_space)
    print ''
    print 'baseline_inf is:'
    print baseline_inf


    text_size = 1.6

    plot_file_path = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Plots/Iteration/'

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
#     perspective_prior_strength_string = convert_array_to_string(perspective_prior_strength)
#     selection_type == 'none'
#
#     if context_generation == 'random':
#         filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_turn_'+str(turnover_type)+'_select_'+selection_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
#     elif context_generation == 'only_helpful' or context_generation == 'optimal':
#         filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_turn_'+str(turnover_type)+'_select_'+selection_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
#     print ''
#     print "filename is:"
#     print filename
#
#
#     data_dict_exo_select_none = pickle.load(open(output_pickle_file_directory+'lex_inf_data_'+lex_measure+'_'+filename+'.p', "rb"))
#
#
#     avg_inf_over_gens_matrix_exo_select_none = data_dict_exo_select_none['raw_data']
#
#     mean_inf_over_gens_exo_select_none = data_dict_exo_select_none['mean_inf_over_gens']
#
#     conf_intervals_inf_over_gens_exo_select_none = data_dict_exo_select_none['conf_intervals_inf_over_gens']
#
#     percentiles_inf_over_gens_exo_select_none = data_dict_exo_select_none['percentiles_inf_over_gens']
#
#
#
#
#
# #####################################################################################
# ## PRIOR 1: EXOCENTRIC
# ### CONDITION 2: Selection on CS:
#
#     print ''
#     print ''
#     print 'This is prior 1: Exocentric'
#     print ''
#     print 'This is condition 2: Selection on CS'
#
#     perspective_prior_type = 'egocentric'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
#     perspective_prior_strength = 0.0  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
#     perspective_prior_strength_string = convert_array_to_string(perspective_prior_strength)
#     selection_type = 'ca_with_parent'


#     if context_generation == 'random':
#         filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_turn_'+str(turnover_type)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
#     elif context_generation == 'only_helpful' or context_generation == 'optimal':
#         filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_turn_'+str(turnover_type)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
#     print ''
#     print "filename is:"
#     print filename
#
#
#     data_dict_exo_select_ca = pickle.load(open(output_pickle_file_directory+'lex_inf_data_'+lex_measure+'_'+filename+'.p', "rb"))
#
#
#     avg_inf_over_gens_matrix_exo_select_ca= data_dict_exo_select_ca['raw_data']
#
#     mean_inf_over_gens_exo_select_ca = data_dict_exo_select_ca['mean_inf_over_gens']
#
#     conf_intervals_inf_over_gens_exo_select_ca = data_dict_exo_select_ca['conf_intervals_inf_over_gens']
#
#     percentiles_inf_over_gens_exo_select_ca = data_dict_exo_select_ca['percentiles_inf_over_gens']
#
#
#
#
#
#
#
#
# #####################################################################################
# ## PRIOR 1: EXOCENTRIC
# ### CONDITION 3: Selection on P-taking:
#
#     print ''
#     print ''
#     print 'This is prior 1: Exocentric'
#     print ''
#     print 'This is condition 3: Selection on P-taking'
#
#     perspective_prior_type = 'egocentric'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
#     perspective_prior_strength = 0.0  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
#     perspective_prior_strength_string = convert_array_to_string(perspective_prior_strength)
#     selection_type = 'p_taking'

#     if context_generation == 'random':
#         filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_turn_'+str(turnover_type)+'_select_'+selection_type+'_weight_'+selection_weight_string+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
#     elif context_generation == 'only_helpful' or context_generation == 'optimal':
#         filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_turn_'+str(turnover_type)+'_select_'+selection_type+'_weight_'+selection_weight_string+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
#     print ''
#     print "filename is:"
#     print filename
#
#
#     data_dict_exo_select_p_taking = pickle.load(open(output_pickle_file_directory+'lex_inf_data_'+lex_measure+'_'+filename+'.p', "rb"))
#
#
#     avg_inf_over_gens_matrix_exo_select_p_taking = data_dict_exo_select_p_taking['raw_data']
#
#     mean_inf_over_gens_exo_select_p_taking = data_dict_exo_select_p_taking['mean_inf_over_gens']
#
#     conf_intervals_inf_over_gens_exo_select_p_taking = data_dict_exo_select_p_taking['conf_intervals_inf_over_gens']
#
#     percentiles_inf_over_gens_exo_select_p_taking = data_dict_exo_select_p_taking['percentiles_inf_over_gens']
#
#
#
# #####################################################################################
# ## PRIOR 2: NEUTRAL
# ### CONDITION 1: No Selection:
#
#     print ''
#     print ''
#     print 'This is prior 2: Uniform'
#     print ''
#     print 'This is condition 1: No selection'
#
#     perspective_prior_type = 'neutral'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
#     perspective_prior_strength = 0.0  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
#     perspective_prior_strength_string = convert_array_to_string(perspective_prior_strength)
#     selection_type = 'none'

#     if context_generation == 'random':
#         filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_turn_'+str(turnover_type)+'_select_'+selection_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
#     elif context_generation == 'only_helpful' or context_generation == 'optimal':
#         filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_turn_'+str(turnover_type)+'_select_'+selection_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
#     print ''
#     print "filename is:"
#     print filename
#
#     data_dict_neut_select_none = pickle.load(open(output_pickle_file_directory+'lex_inf_data_'+lex_measure+'_'+filename+'.p', "rb"))
#
#
#     avg_inf_over_gens_matrix_neut_select_none = data_dict_neut_select_none['raw_data']
#
#     mean_inf_over_gens_neut_select_none = data_dict_neut_select_none['mean_inf_over_gens']
#
#     conf_intervals_inf_over_gens_neut_select_none = data_dict_neut_select_none['conf_intervals_inf_over_gens']
#
#     percentiles_inf_over_gens_neut_select_none = data_dict_neut_select_none['percentiles_inf_over_gens']
#
#
#####################################################################################
## PRIOR 2: NEUTRAL
### CONDITION 2: Selection on CS:

    print ''
    print ''
    print 'This is prior 2: Uniform'
    print ''
    print 'This is condition 2: Selection on CS'

    perspective_prior_type = 'neutral'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
    perspective_prior_strength = 0.0  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
    perspective_prior_strength_string = convert_array_to_string(perspective_prior_strength)
    selection_type = 'ca_with_parent'


    if context_generation == 'random':
        filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    elif context_generation == 'only_helpful' or context_generation == 'optimal':
        filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type




    data_dict_neut_select_ca = pickle.load(open(results_directory+'lex_inf_data_'+lex_measure+'_'+filename_short+'.p', "rb"))


    avg_inf_over_gens_matrix_neut_select_ca = data_dict_neut_select_ca['raw_data']

    mean_inf_over_gens_neut_select_ca = data_dict_neut_select_ca['mean_inf_over_gens']

    conf_intervals_inf_over_gens_neut_select_ca = data_dict_neut_select_ca['conf_intervals_inf_over_gens']

    percentiles_inf_over_gens_neut_select_ca = data_dict_neut_select_ca['percentiles_inf_over_gens']



#####################################################################################
## PRIOR 2: NEUTRAL
### CONDITION 3: Selection on P-taking:

    print ''
    print ''
    print 'This is prior 2: Uniform'
    print ''
    print 'This is condition 3: Selection on P-taking'

    perspective_prior_type = 'neutral'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
    perspective_prior_strength = 0.0  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
    perspective_prior_strength_string = convert_array_to_string(perspective_prior_strength)
    selection_type = 'p_taking'


    if context_generation == 'random':
        filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    elif context_generation == 'only_helpful' or context_generation == 'optimal':
        filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type



    data_dict_neut_select_p_taking = pickle.load(open(results_directory+'lex_inf_data_'+lex_measure+'_'+filename_short+'.p', "rb"))


    avg_inf_over_gens_matrix_neut_select_p_taking = data_dict_neut_select_p_taking['raw_data']

    mean_inf_over_gens_neut_select_p_taking = data_dict_neut_select_p_taking['mean_inf_over_gens']

    conf_intervals_inf_over_gens_neut_select_p_taking = data_dict_neut_select_p_taking['conf_intervals_inf_over_gens']

    percentiles_inf_over_gens_neut_select_p_taking = data_dict_neut_select_p_taking['percentiles_inf_over_gens']


#####################################################################################
# ## PRIOR 3: EGOCENTRIC
# ### CONDITION 1: No Selection:
#
#     print ''
#     print ''
#     print 'This is prior 3: Egocentric'
#     print ''
#     print 'This is condition 1: No selection'
#
#     perspective_prior_type = 'egocentric'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
#     perspective_prior_strength = 0.9  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
#     perspective_prior_strength_string = convert_array_to_string(perspective_prior_strength)
#     selection_type = 'none'
#
#     if context_generation == 'random':
#         filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_turn_'+str(turnover_type)+'_select_'+selection_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
#     elif context_generation == 'only_helpful' or context_generation == 'optimal':
#         filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_turn_'+str(turnover_type)+'_select_'+selection_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
#     print ''
#     print "filename is:"
#     print filename
#
#
#
#     data_dict_ego_select_none = pickle.load(open(output_pickle_file_directory+'lex_inf_data_'+lex_measure+'_'+filename+'.p', "rb"))
#
#
#     avg_inf_over_gens_matrix_ego_select_none = data_dict_ego_select_none['raw_data']
#
#     mean_inf_over_gens_ego_select_none = data_dict_ego_select_none['mean_inf_over_gens']
#
#     conf_intervals_inf_over_gens_ego_select_none = data_dict_ego_select_none['conf_intervals_inf_over_gens']
#
#     percentiles_inf_over_gens_ego_select_none = data_dict_ego_select_none['percentiles_inf_over_gens']
#


#####################################################################################
## PRIOR 3: EGOCENTRIC
### CONDITION 2: Selection on CS:

    print ''
    print ''
    print 'This is prior 3: Egocentric'
    print ''
    print 'This is condition 2: Selection on CS'

    perspective_prior_type = 'egocentric'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
    perspective_prior_strength = 0.9  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
    perspective_prior_strength_string = convert_array_to_string(perspective_prior_strength)
    selection_type = 'ca_with_parent'

    if context_generation == 'random':
        filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    elif context_generation == 'only_helpful' or context_generation == 'optimal':
        filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type



    data_dict_ego_select_ca = pickle.load(open(results_directory+'lex_inf_data_'+lex_measure+'_'+filename_short+'.p', "rb"))


    avg_inf_over_gens_matrix_ego_select_ca = data_dict_ego_select_ca['raw_data']

    mean_inf_over_gens_ego_select_ca = data_dict_ego_select_ca['mean_inf_over_gens']

    conf_intervals_inf_over_gens_ego_select_ca = data_dict_ego_select_ca['conf_intervals_inf_over_gens']

    percentiles_inf_over_gens_ego_select_ca = data_dict_ego_select_ca['percentiles_inf_over_gens']




#####################################################################################
## PRIOR 3: EGOCENTRIC
### CONDITION 3: Selection on P-taking:

    print ''
    print ''
    print 'This is prior 3: Egocentric'
    print ''
    print 'This is condition 3: Selection on P-taking'

    perspective_prior_type = 'egocentric'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
    perspective_prior_strength = 0.9  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
    perspective_prior_strength_string = convert_array_to_string(perspective_prior_strength)
    selection_type = 'p_taking'


    if context_generation == 'random':
        filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    elif context_generation == 'only_helpful' or context_generation == 'optimal':
        filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type


    data_dict_ego_select_p_taking = pickle.load(open(results_directory+'lex_inf_data_'+lex_measure+'_'+filename_short+'.p', "rb"))


    avg_inf_over_gens_matrix_ego_select_p_taking = data_dict_ego_select_p_taking['raw_data']

    mean_inf_over_gens_ego_select_p_taking = data_dict_ego_select_p_taking['mean_inf_over_gens']

    conf_intervals_inf_over_gens_ego_select_p_taking = data_dict_ego_select_p_taking['conf_intervals_inf_over_gens']

    percentiles_inf_over_gens_ego_select_p_taking = data_dict_ego_select_p_taking['percentiles_inf_over_gens']




#####################################################################################
## GENERATING THE PLOT:

    if context_generation == 'random':
        plot_file_title = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type+'_'+lex_measure
    elif context_generation == 'only_helpful' or context_generation == 'optimal':
        plot_file_title = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_l_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type+'_'+lex_measure




    # exo_data = np.array([avg_inf_over_gens_matrix_exo_select_none, avg_inf_over_gens_matrix_exo_select_ca, avg_inf_over_gens_matrix_exo_select_p_taking])
    # print "exo_data.shape is:"
    # print exo_data.shape
    #
    # neut_data = np.array([avg_inf_over_gens_matrix_neut_select_none, avg_inf_over_gens_matrix_neut_select_ca, avg_inf_over_gens_matrix_neut_select_p_taking])
    # print "neut_data.shape is:"
    # print neut_data.shape
    #
    # ego_data = np.array([avg_inf_over_gens_matrix_ego_select_none, avg_inf_over_gens_matrix_ego_select_ca, avg_inf_over_gens_matrix_ego_select_p_taking])
    # print "ego_data.shape is:"
    # print ego_data.shape


    # plot_ego_select_none = sns.tsplot(data=ego_data[0, :, 0:cut_off_point], err_style='ci_band', ci=95)
    # sns.plt.savefig(plot_file_path+'Plot_inf_over_gens_'+str(n_contexts)+'_C_'+'_p_prior_ego_10_select_none'+'_cutoff_'+str(cut_off_point)+'.pdf')
    # plt.show()
    #
    # plot_ego_select_ca = sns.tsplot(data=ego_data[1, :, 0:cut_off_point], err_style='ci_band', ci=95)
    # sns.plt.savefig(plot_file_path+'Plot_inf_over_gens_' + str(n_contexts) + '_C_' + '_p_prior_ego_10_select_ca'+'_cutoff_'+str(cut_off_point)+'.pdf')
    # plt.show()
    #
    # plot_ego_select_p_taking = sns.tsplot(data=ego_data[2, :, 0:cut_off_point], err_style='ci_band', ci=95)
    # sns.plt.savefig(plot_file_path+'Plot_inf_over_gens_' + str(n_contexts) + '_C_' + '_p_prior_ego_10_select_p_taking'+'_cutoff_'+str(cut_off_point)+'.pdf')
    # plt.show()

    # print ''
    # print ''
    # print "mean_inf_over_gens_ego_select_none is:"
    # print mean_inf_over_gens_ego_select_none
    # print "mean_inf_over_gens_ego_select_none.shape is:"
    # print mean_inf_over_gens_ego_select_none.shape
    # print ''
    # print "conf_intervals_inf_over_gens_ego_select_none is:"
    # print conf_intervals_inf_over_gens_ego_select_none
    # print "conf_intervals_inf_over_gens_ego_select_none.shape is:"
    # print conf_intervals_inf_over_gens_ego_select_none.shape


    print ''
    print ''
    print "mean_inf_over_gens_ego_select_ca is:"
    print mean_inf_over_gens_ego_select_ca
    print "mean_inf_over_gens_ego_select_ca.shape is:"
    print mean_inf_over_gens_ego_select_ca.shape
    print ''
    print "conf_intervals_inf_over_gens_ego_select_ca is:"
    print conf_intervals_inf_over_gens_ego_select_ca
    print "conf_intervals_inf_over_gens_ego_select_ca.shape is:"
    print conf_intervals_inf_over_gens_ego_select_ca.shape


    print ''
    print ''
    print "mean_inf_over_gens_ego_select_p_taking is:"
    print mean_inf_over_gens_ego_select_p_taking
    print "mean_inf_over_gens_ego_select_p_taking.shape is:"
    print mean_inf_over_gens_ego_select_p_taking.shape
    print ''
    print "conf_intervals_inf_over_gens_ego_select_p_taking is:"
    print conf_intervals_inf_over_gens_ego_select_p_taking
    print "conf_intervals_inf_over_gens_ego_select_p_taking.shape is:"
    print conf_intervals_inf_over_gens_ego_select_p_taking.shape




    #
    # plot_mean_conf_invs_three_conds(plot_file_path, plot_file_title+'_exo', 'Exocentric P Prior', 'Avg. Informativeness', max_inf, text_size, cut_off_point, mean_inf_over_gens_exo_select_none, conf_intervals_inf_over_gens_exo_select_none, mean_inf_over_gens_exo_select_ca, conf_intervals_inf_over_gens_exo_select_ca, mean_inf_over_gens_exo_select_p_taking, conf_intervals_inf_over_gens_exo_select_p_taking)
    #
    #
    # plot_mean_conf_invs_three_conds(plot_file_path, plot_file_title+'_neut', 'Uniform P Prior', 'Avg. Informativeness', max_inf, text_size, cut_off_point, mean_inf_over_gens_neut_select_none, conf_intervals_inf_over_gens_neut_select_none, mean_inf_over_gens_neut_select_ca, conf_intervals_inf_over_gens_neut_select_ca, mean_inf_over_gens_neut_select_p_taking, conf_intervals_inf_over_gens_neut_select_p_taking)
    #
    #
    #
    # plot_mean_conf_invs_three_conds(plot_file_path, plot_file_title+'_ego', 'Egocentric P Prior', 'Avg. Informativeness', max_inf, text_size, cut_off_point, mean_inf_over_gens_ego_select_none, conf_intervals_inf_over_gens_ego_select_none, mean_inf_over_gens_ego_select_ca, conf_intervals_inf_over_gens_ego_select_ca, mean_inf_over_gens_ego_select_p_taking, conf_intervals_inf_over_gens_ego_select_p_taking)




    # plot_percentiles_three_conds(plot_file_path, plot_file_title+'_exo', 'Exocentric P Prior', 'Avg. Informativeness', max_inf, text_size, cut_off_point, percentiles_inf_over_gens_exo_select_none, percentiles_inf_over_gens_exo_select_ca, percentiles_inf_over_gens_exo_select_p_taking)
    #
    # plot_percentiles_three_conds(plot_file_path, plot_file_title+'_neut', 'Uniform P Prior', 'Avg. Informativeness', max_inf, text_size, cut_off_point, percentiles_inf_over_gens_neut_select_none, percentiles_inf_over_gens_neut_select_ca, percentiles_inf_over_gens_neut_select_p_taking)
    #
    # plot_percentiles_three_conds(plot_file_path, plot_file_title+'_ego', 'Egocentric P Prior', 'Avg. Informativeness', max_inf, text_size, cut_off_point, percentiles_inf_over_gens_ego_select_none, percentiles_inf_over_gens_ego_select_ca, percentiles_inf_over_gens_ego_select_p_taking)



    # plot_percentile_timecourses_three_priors_three_conds(plot_file_path, plot_file_title, 'Avg. informativeness over generations', 'Informativeness', 1.6, cut_off_point, percentiles_inf_over_gens_exo_select_none, percentiles_inf_over_gens_exo_select_ca, percentiles_inf_over_gens_exo_select_p_taking, percentiles_inf_over_gens_neut_select_none, percentiles_inf_over_gens_neut_select_ca, percentiles_inf_over_gens_neut_select_p_taking, percentiles_inf_over_gens_ego_select_none, percentiles_inf_over_gens_ego_select_ca, percentiles_inf_over_gens_ego_select_p_taking)


    #
    # grand_mean_inf_over_gens_exo_select_none = np.mean(mean_inf_over_gens_exo_select_none[burn_in:])
    # print ''
    # print "grand_mean_inf_over_gens_exo_select_none is:"
    # print grand_mean_inf_over_gens_exo_select_none
    #
    # grand_mean_inf_over_gens_exo_select_ca = np.mean(mean_inf_over_gens_exo_select_ca[burn_in:])
    # print ''
    # print "grand_mean_inf_over_gens_exo_select_ca is:"
    # print grand_mean_inf_over_gens_exo_select_ca
    #
    # grand_mean_inf_over_gens_exo_select_p_taking = np.mean(mean_inf_over_gens_exo_select_p_taking [burn_in:])
    # print ''
    # print "grand_mean_inf_over_gens_exo_select_p_taking  is:"
    # print grand_mean_inf_over_gens_exo_select_p_taking
    #
    #
    #
    #
    #
    # grand_mean_inf_over_gens_neut_select_none = np.mean(mean_inf_over_gens_neut_select_none[burn_in:])
    # print ''
    # print "grand_mean_inf_over_gens_neut_select_none is:"
    # print grand_mean_inf_over_gens_neut_select_none
    #
    grand_mean_inf_over_gens_neut_select_ca = np.mean(mean_inf_over_gens_neut_select_ca[burn_in:])
    print ''
    print "grand_mean_inf_over_gens_neut_select_ca is:"
    print grand_mean_inf_over_gens_neut_select_ca

    grand_mean_inf_over_gens_neut_select_p_taking = np.mean(mean_inf_over_gens_neut_select_p_taking[burn_in:])
    print ''
    print "grand_mean_inf_over_gens_neut_select_p_taking  is:"
    print grand_mean_inf_over_gens_neut_select_p_taking

    #
    #
    #
    # grand_mean_inf_over_gens_ego_select_none = np.mean(mean_inf_over_gens_ego_select_none[burn_in:])
    # print ''
    # print "grand_mean_inf_over_gens_ego_select_none is:"
    # print grand_mean_inf_over_gens_ego_select_none

    grand_mean_inf_over_gens_ego_select_ca = np.mean(mean_inf_over_gens_ego_select_ca[burn_in:])
    print ''
    print "grand_mean_inf_over_gens_ego_select_ca is:"
    print grand_mean_inf_over_gens_ego_select_ca

    grand_mean_inf_over_gens_ego_select_p_taking = np.mean(mean_inf_over_gens_ego_select_p_taking[burn_in:])
    print ''
    print "grand_mean_inf_over_gens_ego_select_p_taking  is:"
    print grand_mean_inf_over_gens_ego_select_p_taking


    # plot_inf_bars_diff_conds(plot_file_path, plot_file_title, 'Exocentric P Prior', 'exo', grand_mean_inf_over_gens_exo_select_none, grand_mean_inf_over_gens_exo_select_ca, grand_mean_inf_over_gens_exo_select_p_taking, [], [], [], max_inf, text_size=1.6)
    #
    # plot_inf_bars_diff_conds(plot_file_path, plot_file_title, 'Uniform P Prior', 'neut', grand_mean_inf_over_gens_neut_select_none, grand_mean_inf_over_gens_neut_select_ca, grand_mean_inf_over_gens_neut_select_p_taking, [], [], [], max_inf, text_size=1.6)
    #
    # plot_inf_bars_diff_conds(plot_file_path, plot_file_title, 'Egocentric P Prior', 'ego', grand_mean_inf_over_gens_ego_select_none, grand_mean_inf_over_gens_ego_select_ca, grand_mean_inf_over_gens_ego_select_p_taking, [], [], [], max_inf, text_size=1.6)
    #

    # grand_means_select_none = np.array([grand_mean_inf_over_gens_exo_select_none, grand_mean_inf_over_gens_neut_select_none, grand_mean_inf_over_gens_ego_select_none])
    #
    # grand_means_select_ca = np.array([grand_mean_inf_over_gens_exo_select_ca, grand_mean_inf_over_gens_neut_select_ca, grand_mean_inf_over_gens_ego_select_ca])
    #
    # grand_means_select_p_taking = np.array([grand_mean_inf_over_gens_exo_select_p_taking, grand_mean_inf_over_gens_neut_select_p_taking, grand_mean_inf_over_gens_ego_select_p_taking])
    #
    #
    # plot_mean_informativeness_all_conds(plot_file_path, plot_file_title, grand_means_select_none, grand_means_select_ca, grand_means_select_p_taking, [], [], [], baseline, max_inf, text_size)
    #



    plot_title = 'Egocentric perspective prior'
    p_prior = 'neut_00'

    plot_inf_bars_two_conds(plot_file_path, plot_file_title, plot_title, p_prior, grand_mean_inf_over_gens_neut_select_ca, grand_mean_inf_over_gens_neut_select_p_taking, baseline_inf, max_inf, text_size)



    plot_title = 'Egocentric perspective prior'
    p_prior = 'egoc_09'

    plot_inf_bars_two_conds(plot_file_path, plot_file_title, plot_title, p_prior, grand_mean_inf_over_gens_ego_select_ca, grand_mean_inf_over_gens_ego_select_p_taking, baseline_inf, max_inf, text_size)
