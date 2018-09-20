__author__ = 'Marieke Woensdregt'


import numpy as np
from scipy import stats
import time

import hypspace
import lex
import plots
import pop
import prior
import saveresults



# np.set_printoptions(threshold=np.nan)



#######################################################################################################################
# 1: THE PARAMETERS:


##!!!!!! MAKE SURE TO CHANGE THE PATHS BELOW TO MATCH THE FILE SYSTEM OF YOUR MACHINE:
pickle_file_directory = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/'

plot_file_directory = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Plots/'

run_type_dir = 'Iteration/'





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

pop_size = 10

agent_type = 'no_p_distinction' # This can be set to either 'p_distinction' or 'no_p_distinction'. Determines whether the population is made up of DistinctionAgent objects or Agent objects (DistinctionAgents can learn different perspectives for different agents, Agents can only learn one perspective for all agents they learn from).




pragmatic_level = 'prag'  # This can be set to either 'literal', 'perspective-taking' or 'prag'
optimality_alpha = 3.0  # Goodman & Stuhlmuller (2013) fitted sal_alpha = 3.4 to participant data with 4x3 lexicon of three number words ('one', 'two', and 'three') that could be mapped to four different world states (0, 1, 2, and 3)
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
n_contexts = 12  # The number of contexts that the learner gets to see.
speaker_order_type = 'random'  # This can be set to either 'random', 'random_equal' (for random order with making sure that each speaker gets an equal amount of utterances) 'same_first' (first portion of input comes from same perspective, second portion of input from opposite perspective), 'same_first_equal' (where both speakers get to produce the exact same amount of utterances), 'opp_first' (vice versa) or 'opp_first_equal'
first_input_stage_ratio = 0.5  # This is the ratio of contexts that will make up the first input stage (see parameter 'speaker_order_type' above)



# 1.8: The parameters that determine the type and number of simulations that are run:

#FIXME: In the current implementation 'half_ambiguous_lex' can have different instantiations. Therefore, if we run a simulation where there are different lexicon_type_probs, we will want the run_type to be 'population_same_pop' to make sure that all the 'half ambiguous' speakers do have the SAME half ambiguous lexicon.
run_type = 'iter'  # This parameter determines whether the learner communicates with only one speaker ('dyadic') or with a population ('population_diff_pop' if there are no speakers with the 'half_ambiguous' lexicon type, 'population_same_pop' if there are, 'population_same_pop_dist_learner' if the learner can distinguish between different speakers), or whether we do an iterated learning model ('iter')

communication_type = 'prag'  # This can be set to either 'lex_only', 'lex_n_context', 'lex_n_p' or 'prag'
ca_measure_type = 'comp_only'  # This can be set to either "comp_n_prod" or "comp_only"
n_interactions = 6  # The number of interactions used to calculate communicative accuracy

selection_type = 'ca_with_parent'  # This can be set to either 'none' or 'p_taking'
selection_weighting = 'none'  # This is a factor with which the fitness of the agents (determined as the probability they assign to the correct perspective hypothesis) is multiplied and then exponentiated in order to weight the relative agent fitness (which in turn determines the probability of becoming a teacher for the next generation). A value of 0. implements neutral selection. A value of 1.0 creates weighting where the fitness is pretty much equal to relative posterior probability on correct p hyp), and the higher the value, the more skewed the weighting in favour of agents with better perspective-taking.
if isinstance(selection_weighting, float):
    selection_weight_string = str(np.int(selection_weighting))
else:
    selection_weight_string = selection_weighting


turnover_type = 'whole_pop'  # The type of turnover with which the population is replaced (for the iteration simulation). This can be set to either 'chain' (one agent at a time) or 'whole_pop' (entire population at once)
n_iterations = 10  # The number of iterations (i.e. new agents in the case of 'chain', generations in the case of 'whole_pop')
report_every_i = 1
cut_off_point = 5
n_runs = 2  # The number of runs of the simulation
report_every_r = 1


recording = 'minimal'  # This can be set to either 'everything' or 'minimal'

which_hyps_on_graph = 'all_hyps'  # This is used in the plot_post_probs_over_lex_hyps() function, and can be set to either 'all_hyps' or 'lex_hyps_only'

lex_measure = 'ca'  # This can be set to either 'mi' for mutual information or 'ca' for communicative accuracy (of the lexicon with itself)

posterior_threshold = 0.95  # This parameter is used by the 'measur_n_data_points_for_p_learning' function

decoupling = True  # This can be set to either True or False. It determines whether genetic and cultural inheritance are coupled (i.e. from the same cultural parent) or decoupled.


#######################################################################################################################


def get_selected_hyps_ordered(n_runs, n_iterations, pop_size, multi_run_selected_hyps_per_generation_matrix, hyp_order):
            selected_hyps_new_lex_order_all_runs = np.zeros_like(multi_run_selected_hyps_per_generation_matrix)
            for r in range(n_runs):
                for i in range(n_iterations):
                    for a in range(pop_size):
                        this_agent_hyp = multi_run_selected_hyps_per_generation_matrix[r][i][a]
                        if this_agent_hyp >= len(hyp_order):
                            this_agent_hyp = this_agent_hyp-len(hyp_order)
                        new_order_index = np.argwhere(hyp_order == this_agent_hyp)
                        selected_hyps_new_lex_order_all_runs[r][i][a] = new_order_index
            return selected_hyps_new_lex_order_all_runs


def calc_mean_and_conf_invs_distribution(n_runs, n_copies, lexicon_hyps, which_hyps_on_graph, min_info_indices, intermediate_info_indices, max_info_indices, n_iterations, cut_off_point, selected_hyps_new_lex_order_all_runs):
    hist_values_averaged_per_run = np.zeros((n_runs, 3))
    for r in range(n_runs):
        selected_hyps_new_lex_order_final_section = selected_hyps_new_lex_order_all_runs[r][cut_off_point:n_iterations]
        selected_hyps_new_lex_order_final_section = selected_hyps_new_lex_order_final_section.flatten()
        if which_hyps_on_graph == 'lex_hyps_only' or which_hyps_on_graph == 'lex_hyps_collapsed':
            n_lex_hyps = len(lexicon_hyps)
            for i in range(len(selected_hyps_new_lex_order_final_section)):
                hyp_index = selected_hyps_new_lex_order_final_section[i]
                if hyp_index > (n_lex_hyps - 1):
                    selected_hyps_new_lex_order_final_section[i] = hyp_index - n_lex_hyps
        hist_values = np.histogram(selected_hyps_new_lex_order_final_section, bins=[0, min_info_indices[-1]+1, intermediate_info_indices[-1]+1, max_info_indices[-1]])
        hist_values_averaged = np.divide(hist_values[0].astype(float), [float(len(min_info_indices)), float(len(intermediate_info_indices)), float(len(max_info_indices))])
        hist_values_averaged_per_run[r] = hist_values_averaged
    hist_values_averaged_per_run_optimal_first = np.fliplr(hist_values_averaged_per_run)
    mean_selected_hyps_by_lex_type = np.mean(hist_values_averaged_per_run_optimal_first, axis=0)
    std_selected_hyps_by_lex_type = np.std(hist_values_averaged_per_run_optimal_first, axis=0)
    conf_invs_selected_hyps_by_lex_type = stats.norm.interval(0.95, loc=mean_selected_hyps_by_lex_type, scale=std_selected_hyps_by_lex_type / np.sqrt(n_runs*n_copies))
    conf_invs_selected_hyps_by_lex_type = np.array(conf_invs_selected_hyps_by_lex_type)
    lower_yerr_selected_hyps_for_plot = np.subtract(mean_selected_hyps_by_lex_type, conf_invs_selected_hyps_by_lex_type[0])
    upper_yerr_selected_hyps_for_plot = np.subtract(conf_invs_selected_hyps_by_lex_type[1], mean_selected_hyps_by_lex_type)
    yerr_selected_hyps_for_plot = np.array([lower_yerr_selected_hyps_for_plot, upper_yerr_selected_hyps_for_plot])
    hypothesis_count_proportions = np.divide(mean_selected_hyps_by_lex_type, np.sum(mean_selected_hyps_by_lex_type))
    yerr_scaled_selected_hyps_for_plot = np.divide(yerr_selected_hyps_for_plot, np.sum(mean_selected_hyps_by_lex_type))
    return hypothesis_count_proportions, yerr_scaled_selected_hyps_for_plot




### CATEGORISING LEXICONS BY INFORMATIVENESS BELOW:



print ''
print ''
# print "lexicon_hyps are:"
# print lexicon_hyps
print "lexicon_hyps.shape are:"
print lexicon_hyps.shape



informativity_per_lexicon = lex.calc_ca_all_lexicons(lexicon_hyps, error, lex_measure)
print ''
print ''
# print "informativity_per_lexicon is:"
# print informativity_per_lexicon
print "informativity_per_lexicon.shape is:"
print informativity_per_lexicon.shape


argsort_informativity_per_lexicon = np.argsort(informativity_per_lexicon)
print ''
print ''
# print "argsort_informativity_per_lexicon is:"
# print argsort_informativity_per_lexicon
print "argsort_informativity_per_lexicon.shape is:"
print argsort_informativity_per_lexicon.shape


informativity_per_lexicon_sorted = np.round(informativity_per_lexicon[argsort_informativity_per_lexicon], decimals=2)
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
# print "intermediate_info_indices are:"
# print intermediate_info_indices

lexicon_hyps_sorted = lexicon_hyps[argsort_informativity_per_lexicon]
print ''
print ''
# print "lexicon_hyps_sorted is:"
# print lexicon_hyps_sorted
print "lexicon_hyps_sorted.shape is:"
print lexicon_hyps_sorted.shape


#######################################################################################################################



#TODO: Finish the convergence measures inside this iteration function:
def multi_runs_iteration(n_meanings, n_signals, n_runs, n_iterations, report_every_r, report_every_i, turnover_type, selection_type, selection_weighting, communication_type, ca_measure_type, n_interactions, n_contexts, n_utterances, context_generation, context_type, context_size, helpful_contexts, pop_size, teacher_type, speaker_order_type, first_input_stage_ratio, agent_type, perspectives, perspective_probs, sal_alpha, lexicon_probs, error, extra_error, pragmatic_level, optimality_alpha, learning_types, learning_type_probs, hypothesis_space, perspective_hyps, lexicon_hyps, learner_perspective, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant, recording):

    t0 = time.clock()

    if recording == 'everything':
        multi_run_log_posteriors_per_agent_matrix = np.zeros((n_runs, n_iterations, pop_size, len(hypothesis_space)))
        multi_run_log_posteriors_final_agent_matrix = np.zeros((n_runs, pop_size, len(hypothesis_space)))
        multi_run_pop_lexicons_matrix = np.zeros((n_runs, n_iterations, pop_size, n_meanings, n_signals))
        multi_run_log_posteriors_per_data_point_per_agent_per_generation_matrix = np.zeros((n_runs, n_iterations, pop_size, (n_contexts+1), len(hypothesis_space)))
    multi_run_selected_hyps_per_generation_matrix = np.zeros((n_runs, n_iterations, pop_size))
    multi_run_avg_fitness_matrix = np.zeros((n_runs, n_iterations))
    multi_run_parent_probs_matrix = np.zeros((n_runs, n_iterations, pop_size))
    multi_run_selected_parent_indices_matrix = np.zeros((n_runs, n_iterations, pop_size))
    multi_run_parent_lex_indices_matrix = np.zeros((n_runs, n_iterations, pop_size))


    #
    # dataset_array_pickle_file_specs = 'Dataset_array_'+str(n_meanings)+'M_'+str(n_signals)+'S_'+context_generation+'_'+str(len(helpful_contexts))+'_contexts_'+str(n_contexts)+'_observations'
    #
    # dataset_array_pickle_file_name = output_pickle_file_directory + dataset_array_pickle_file_specs + '.p'
    #
    #
    # likelihood_pickle_file_specs = 'Likelihood_array_'+str(n_meanings)+'M_'+str(n_signals)+'S_'+context_generation+'_'+str(len(helpful_contexts))+'_contexts_'+str(len(perspective_hyps))+'_p_hyps_'+str(len(lexicon_hyps))+'_l_hyps_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_error_'+error_string+'_'+agent_type
    #
    # log_likelihood_pickle_file_name = output_pickle_file_directory + likelihood_pickle_file_specs + '.p'


    print ''
    print "This simulation consists of:"
    print str(n_runs)+' runs'
    print 'and'
    print str(n_iterations)+' iterations'
    print 'with population size:'
    print pop_size
    print 'and turnover type:'
    print turnover_type
    print 'the selection type is:'
    print selection_type
    print 'and the selection strength parameter is:'
    print selection_weighting
    if selection_type == 'ca_with_parent':
        print "the communication_type is:"
        print communication_type
        print "and the ca_measure_type is:"
        print ca_measure_type
        print "with a total of "+str(n_interactions)+" interactions"
    print 'communicating with:'
    print str(n_meanings)+' meanings'
    print 'and'
    print str(n_signals) + ' signals'
    print 'observing:'
    print str(n_contexts)+' contexts per learner'
    print 'with context type:'
    print context_generation
    if context_generation == 'only_helpful' or context_generation == 'optimal':
        print 'containing the '+str(len(helpful_contexts))+' most informative contexts'
    print ''
    print 'The teacher type is:'
    print teacher_type
    print ''
    print 'The agent type is:'
    print agent_type
    print ''
    print 'The possible perspectives are:'
    print perspectives
    print 'The perspective probabilities are:'
    print perspective_probs
    print ''
    print ''
    print ''
    print "lexicon_probs.shape are:"
    print lexicon_probs.shape
    print ''
    initial_lex_index = np.argwhere(lexicon_probs==1.0)
    print "initial_lex_index is:"
    print initial_lex_index
    print "lexicon_hyps[initial_lex_index] is:"
    print lexicon_hyps[initial_lex_index]
    print ''
    print 'The pragmatic_level is:'
    print pragmatic_level
    if pragmatic_level == 'prag':
        print 'The optimality_alpha (relevant for pragmatic speakers) is:'
        print optimality_alpha
    print ''
    print 'The possible learning types are:'
    print learning_types
    print 'The learning type probabilities are:'
    print learning_type_probs
    print ''
    print 'The perspective_prior is:'
    print perspective_prior_type
    print 'With strength:'
    print perspective_prior_strength
    print 'The lexicon prior is:'
    print lexicon_prior_type
    print 'With strength:'
    print lexicon_prior_constant
    print ''
    print 'P(error) on production is:'
    print error
    print ''


    for r in range(n_runs):
        if r == 0 or r % report_every_r == 0:
            print 'r = ' + str(r)

        # 1) First the initial population is created:

        if agent_type == 'p_distinction':
            # 1.1) Then the prior probability distribution for all agents is created:
            ## 1.1.1) First the perspective prior is created:
            perspective_prior = prior.create_perspective_prior(perspective_hyps, lexicon_hyps, perspective_prior_type, learner_perspective, perspective_prior_strength)
            ## 1.1.2) Then the lexicon prior is created:
            lexicon_prior = prior.create_lexicon_prior(lexicon_hyps, lexicon_prior_type, lexicon_prior_constant, error)
            ## 1.1.3) And finally the full composite prior matrix is created using the separate lexicon_prior and perspective_prior, and following the configuration of hypothesis_space
            composite_log_prior = prior.list_composite_log_priors_with_speaker_distinction(hypothesis_space, perspective_hyps, lexicon_hyps, perspective_prior, lexicon_prior, pop_size)

            # 1.2) Then the population is created:

            lexicons_per_agent = np.random.choice(np.arange(len(lexicon_hyps)), pop_size, replace=True, p=lexicon_probs)

            perspectives_per_agent = np.random.choice(perspectives, pop_size, replace=True, p=perspective_probs)

            for i in range(pop_size):
                learning_types_per_agent = np.random.choice(learning_types, pop_size, replace=True, p=learning_type_probs)

            population = pop.DistinctionPopulation(pop_size, n_meanings, n_signals, hypothesis_space, perspective_hyps, lexicon_hyps, learner_perspective, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant, composite_log_prior, perspectives, perspectives_per_agent, perspective_probs, sal_alpha, lexicon_hyps, lexicons_per_agent, error, n_contexts, context_type, context_generation, context_size, helpful_contexts, n_utterances, learning_types, learning_types_per_agent, learning_type_probs)


        elif agent_type == 'no_p_distinction':
            population = pop.Population(pop_size, n_meanings, n_signals, hypothesis_space, perspective_hyps, lexicon_hyps, learner_perspective, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant, perspectives, perspective_probs, sal_alpha, lexicon_probs, error, extra_error, pragmatic_level, optimality_alpha, n_contexts, context_type, context_generation, context_size, helpful_contexts, n_utterances, learning_types, learning_type_probs)


        if r == 0:
            print 
            print 'initial population for run 0 is:'
            population.print_population()

        for i in range(n_iterations):
            if i == 0 or i % report_every_i == 0:
                print 'i = ' + str(i)

            if recording == 'everything':
                population_log_posteriors_matrix = population.get_all_log_posteriors_matrix()

                multi_run_log_posteriors_per_agent_matrix[r][i] = population_log_posteriors_matrix
                population_lexicons_matrix = population.get_all_lexicons_matrix()
                multi_run_pop_lexicons_matrix[r][i] = population_lexicons_matrix

            if agent_type == 'p_distinction':
                if recording == 'minimal':
                    selected_hyp_per_agent_matrix, avg_fitness, parent_probs, selected_parent_indices, parent_lex_indices = population.pop_update(recording, context_generation, helpful_contexts, n_meanings, n_signals, error, turnover_type, selection_type, selection_weighting, communication_type, ca_measure_type, n_interactions, teacher_type, speaker_order_type, first_input_stage_ratio, perspectives_per_agent)
                elif recording == 'everything':
                    selected_hyp_per_agent_matrix, avg_fitness, parent_probs, selected_parent_indices, parent_lex_indices, log_posteriors_per_data_point_per_agent_matrix = population.pop_update(recording, context_generation, helpful_contexts, n_meanings, n_signals, error, turnover_type, selection_type, selection_weighting, communication_type, ca_measure_type, n_interactions, teacher_type, speaker_order_type, first_input_stage_ratio, perspectives_per_agent)
            elif agent_type == 'no_p_distinction':
                if recording == 'minimal':
                    selected_hyp_per_agent_matrix, avg_fitness, parent_probs, selected_parent_indices, parent_lex_indices = population.pop_update(recording, context_generation, helpful_contexts, n_meanings, n_signals, error, turnover_type, selection_type, selection_weighting, communication_type, ca_measure_type, n_interactions, teacher_type, perspectives_per_agent=None)
                elif recording == 'everything':
                    selected_hyp_per_agent_matrix, avg_fitness, parent_probs, selected_parent_indices, parent_lex_indices, log_posteriors_per_data_point_per_agent_matrix = population.pop_update(recording, context_generation, helpful_contexts, n_meanings, n_signals, error, turnover_type, selection_type, selection_weighting, communication_type, ca_measure_type, n_interactions, teacher_type, perspectives_per_agent=None)

            multi_run_selected_hyps_per_generation_matrix[r][i] = selected_hyp_per_agent_matrix
            multi_run_avg_fitness_matrix[r][i] = avg_fitness
            multi_run_parent_probs_matrix[r][i] = parent_probs
            multi_run_selected_parent_indices_matrix[r][i] = selected_parent_indices
            multi_run_parent_lex_indices_matrix[r][i] = parent_lex_indices

            if recording == 'everything':
                multi_run_log_posteriors_per_data_point_per_agent_per_generation_matrix[r][i] = log_posteriors_per_data_point_per_agent_matrix

        if recording == 'everything':
            multi_run_log_posteriors_final_agent_matrix[r] = population_log_posteriors_matrix


    run_time_mins = (time.clock()-t0)/60.

    if recording == 'everything':
        results_dict = {'multi_run_log_posteriors_per_agent_matrix':multi_run_log_posteriors_per_agent_matrix,
                        'multi_run_log_posteriors_final_agent_matrix':multi_run_log_posteriors_final_agent_matrix,
                        'multi_run_selected_hyps_per_generation_matrix':multi_run_selected_hyps_per_generation_matrix,
                        'multi_run_pop_lexicons_matrix':multi_run_pop_lexicons_matrix,
                        'multi_run_avg_fitness_matrix':multi_run_avg_fitness_matrix,
                        'multi_run_parent_probs_matrix': multi_run_parent_probs_matrix,
                        'multi_run_selected_parent_indices_matrix': multi_run_selected_parent_indices_matrix,
                        'multi_run_parent_lex_indices_matrix': multi_run_parent_lex_indices_matrix,
                        'multi_run_log_posteriors_per_data_point_per_agent_per_generation_matrix': multi_run_log_posteriors_per_data_point_per_agent_per_generation_matrix,
                        'run_time_mins':run_time_mins}
    elif recording == 'minimal':
        results_dict = {'multi_run_selected_hyps_per_generation_matrix':multi_run_selected_hyps_per_generation_matrix,
                        'multi_run_avg_fitness_matrix':multi_run_avg_fitness_matrix,
                        'multi_run_parent_probs_matrix':multi_run_parent_probs_matrix,
                        'multi_run_selected_parent_indices_matrix':multi_run_selected_parent_indices_matrix,
                        'multi_run_parent_lex_indices_matrix':multi_run_parent_lex_indices_matrix,
                        'run_time_mins':run_time_mins}

    return results_dict






def measur_n_data_points_for_p_learning(hypothesis_space, perspectives, perspective_probs, multi_run_log_posteriors_per_data_point_per_agent_per_generation_matrix, posterior_threshold):
    comp_hyp_indices = np.arange(len(hypothesis_space))
    comp_hyp_indices_split_on_p_hyps = np.split(comp_hyp_indices, len(perspectives))
    correct_p_index = np.where(perspective_probs == 1.0)
    correct_p_hyp_indices = comp_hyp_indices_split_on_p_hyps[correct_p_index[0]]

    min_n_data_points_per_agent_per_generation_per_run = np.zeros((n_runs, n_iterations, pop_size))
    for r in range(n_runs):
        for i in range(n_iterations):
            for a in range(pop_size):
                log_posteriors_per_data_point_matrix = multi_run_log_posteriors_per_data_point_per_agent_per_generation_matrix[r][i][a]
                log_posteriors_correct_hyps = log_posteriors_per_data_point_matrix[:, correct_p_hyp_indices]
                posteriors_correct_hyps = np.exp(log_posteriors_correct_hyps)
                posteriors_correct_summed = np.sum(posteriors_correct_hyps, axis=1)
                threshold_array = np.greater_equal(posteriors_correct_summed, posterior_threshold)
                threshold_indices = np.argwhere(threshold_array)
                if len(threshold_indices) > 0:
                    min_n_data_points = threshold_indices[0][0]
                else:
                    min_n_data_points = 'NaN'

                min_n_data_points_per_agent_per_generation_per_run[r][i][a] = min_n_data_points


    min_n_data_points_per_agent_per_generation_avg_over_runs = np.mean(min_n_data_points_per_agent_per_generation_per_run, axis=0)

    min_n_data_points_per_agent_per_generation_flat = min_n_data_points_per_agent_per_generation_avg_over_runs.flatten()

    return min_n_data_points_per_agent_per_generation_flat




if __name__ == "__main__":

    t0 = time.clock()

    all_results_dict = multi_runs_iteration(n_meanings, n_signals, n_runs, n_iterations, report_every_r, report_every_i, turnover_type, selection_type, selection_weighting, communication_type, ca_measure_type, n_interactions, n_contexts, n_utterances, context_generation, context_type, context_size, helpful_contexts, pop_size, teacher_type, speaker_order_type, first_input_stage_ratio, agent_type, perspectives, perspective_probs, sal_alpha, lexicon_probs, error, extra_error, pragmatic_level, optimality_alpha, learning_types, learning_type_probs, hypothesis_space, perspective_hyps, lexicon_hyps, learner_perspective, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant, recording)


    run_simulation_time = time.clock()-t0
    print
    print 'run_simulation_time is:'
    print str((run_simulation_time/60))+" m"


    t1 = time.clock()


    if recording == 'everything':
        multi_run_log_posteriors_per_agent_matrix = all_results_dict['multi_run_log_posteriors_per_agent_matrix']
        # print "np.exp(multi_run_log_posteriors_per_agent_matrix) is:"
        # print np.exp(multi_run_log_posteriors_per_agent_matrix)
        print "multi_run_log_posteriors_per_agent_matrix.shape is:"
        print multi_run_log_posteriors_per_agent_matrix.shape

        multi_run_log_posteriors_final_agent_matrix = all_results_dict['multi_run_log_posteriors_final_agent_matrix']
        # print "np.exp(multi_run_log_posteriors_final_agent_matrix) is:"
        # print np.exp(multi_run_log_posteriors_final_agent_matrix)
        print "multi_run_log_posteriors_final_agent_matrix.shape is:"
        print multi_run_log_posteriors_final_agent_matrix.shape

        multi_run_pop_lexicons_matrix = all_results_dict['multi_run_pop_lexicons_matrix']
        # print "multi_run_pop_lexicons_matrix is:"
        # print multi_run_pop_lexicons_matrix
        print "multi_run_pop_lexicons_matrix.shape is:"
        print multi_run_pop_lexicons_matrix.shape

        multi_run_log_posteriors_per_data_point_per_agent_per_generation_matrix = all_results_dict['multi_run_log_posteriors_per_data_point_per_agent_per_generation_matrix']
        # print "np.exp(multi_run_log_posteriors_per_data_point_per_agent_per_generation_matrix) is:"
        # print np.exp(multi_run_log_posteriors_per_data_point_per_agent_per_generation_matrix)
        print "multi_run_log_posteriors_per_data_point_per_agent_per_generation_matrix.shape is:"
        print multi_run_log_posteriors_per_data_point_per_agent_per_generation_matrix.shape


    multi_run_selected_hyps_per_generation_matrix = all_results_dict['multi_run_selected_hyps_per_generation_matrix']
    # print "multi_run_selected_hyps_per_generation_matrix is:"
    # print multi_run_selected_hyps_per_generation_matrix
    print "multi_run_selected_hyps_per_generation_matrix.shape is:"
    print multi_run_selected_hyps_per_generation_matrix.shape

    multi_run_avg_fitness_matrix = all_results_dict['multi_run_avg_fitness_matrix']
    # print "multi_run_avg_fitness_matrix is:"
    # print multi_run_avg_fitness_matrix
    print "multi_run_avg_fitness_matrix.shape is:"
    print multi_run_avg_fitness_matrix.shape

    multi_run_parent_probs_matrix = all_results_dict['multi_run_parent_probs_matrix']
    # print "multi_run_parent_probs_matrix is:"
    # print multi_run_parent_probs_matrix
    print "multi_run_parent_probs_matrix.shape is:"
    print multi_run_parent_probs_matrix.shape

    multi_run_selected_parent_indices_matrix = all_results_dict['multi_run_selected_parent_indices_matrix']
    # print "multi_run_selected_parent_indices_matrix is:"
    # print multi_run_selected_parent_indices_matrix
    print "multi_run_selected_parent_indices_matrix.shape is:"
    print multi_run_selected_parent_indices_matrix.shape

    multi_run_parent_lex_indices_matrix = all_results_dict['multi_run_parent_lex_indices_matrix']
    # print "multi_run_parent_lex_indices_matrix is:"
    # print multi_run_parent_lex_indices_matrix
    print "multi_run_parent_lex_indices_matrix.shape is:"
    print multi_run_parent_lex_indices_matrix.shape


    multi_run_proportion_max_offspring_single_parent = np.zeros((n_runs, n_iterations))
    for r in range(n_runs):
        for i in range(n_iterations):
            parent_indices = multi_run_selected_parent_indices_matrix[r][i]
            parent_index_counts = np.bincount(parent_indices.astype(int))
            max_offspring_single_parent = np.amax(parent_index_counts)
            proportion_max_offspring_single_parent = np.divide(max_offspring_single_parent.astype(float), float(pop_size))
            multi_run_proportion_max_offspring_single_parent[r][i] = proportion_max_offspring_single_parent
    print ''
    print ''
    print "multi_run_proportion_max_offspring_single_parent is:"
    print multi_run_proportion_max_offspring_single_parent
    print "multi_run_proportion_max_offspring_single_parent.shape is:"
    print multi_run_proportion_max_offspring_single_parent.shape


    calc_performance_measures_time = time.clock()-t1
    print
    print 'calc_performance_measures_time is:'
    print str((calc_performance_measures_time/60))+" m"




    #############################################################################
    # Below the results are saved in pickle files:


    t2 = time.clock()


    if context_generation == 'random':
        if selection_type == 'none' or selection_type == 'l_learning':
            filename = 'iter_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
        elif selection_type == 'p_taking':
            filename = 'iter_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_weight_'+selection_weight_string+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
        elif selection_type == 'ca_with_parent':
            filename = 'iter_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type


    elif context_generation == 'only_helpful' or context_generation == 'optimal':
        if selection_type == 'none' or selection_type == 'l_learning':
            filename = 'iter_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
        elif selection_type == 'p_taking':
            filename = 'iter_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_weight_'+selection_weight_string+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
        elif selection_type == 'ca_with_parent':
            filename = 'iter_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type




    pickle_file_title_all_results = pickle_file_directory + run_type_dir + 'Results_' + filename

    saveresults.write_results_to_pickle_file(pickle_file_title_all_results, all_results_dict)


    pickle_file_title_max_offspring_single_parent = pickle_file_directory + run_type_dir + 'Max_Offspring_' + filename

    saveresults.write_results_to_pickle_file(pickle_file_title_max_offspring_single_parent, multi_run_proportion_max_offspring_single_parent)


    write_to_files_time = time.clock()-t2
    print
    print 'write_to_files_time is:'
    print str((write_to_files_time/60))+" m"




    #############################################################################
    # The code below makes the plots and saves them:



    t3 = time.clock()


    selected_hyps_new_lex_order_all_runs = get_selected_hyps_ordered(n_runs, n_iterations, pop_size, multi_run_selected_hyps_per_generation_matrix, argsort_informativity_per_lexicon)


    hypothesis_count_proportions, yerr_scaled_selected_hyps_for_plot = calc_mean_and_conf_invs_distribution(n_runs, 1, lexicon_hyps, which_hyps_on_graph, min_info_indices, intermediate_info_indices, max_info_indices, n_iterations, cut_off_point, selected_hyps_new_lex_order_all_runs)
    print ''
    print ''
    print "hypothesis_count_proportions are:"
    print hypothesis_count_proportions
    print ''
    print "yerr_scaled_selected_hyps_for_plot is;"
    print yerr_scaled_selected_hyps_for_plot





    #########################################################
    # BELOW THE PLOT SHOWING THE PROPORTIONS WITH WHICH DIFFERENT HYPOTHESES ARE SELECTED OVER GENERATIONS IS GENERATED:


    plot_file_path = plot_file_directory + run_type_dir

    if context_generation == 'random':
        if selection_type == 'none' or selection_type == 'l_learning':
            plot_file_title = '_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
        elif selection_type == 'p_taking':
            plot_file_title = '_size_'+str(pop_size)+'_select_'+selection_type+'_weight_'+selection_weight_string+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
        elif selection_type == 'ca_with_parent':
            plot_file_title = '_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type

    elif context_generation == 'only_helpful' or context_generation == 'optimal':
        if selection_type == 'none' or selection_type == 'l_learning':
            plot_file_title = '_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
        elif selection_type == 'p_taking':
            plot_file_title = '_size_'+str(pop_size)+'_select_'+selection_type+'_weight_'+selection_weight_string+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
        elif selection_type == 'ca_with_parent':
            plot_file_title = '_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type




    plot_title = 'Egocentric perspective prior & '+str(n_meanings)+'x'+str(n_signals)+' lexicons'

    plots.plot_lex_distribution(plot_file_path, plot_file_title, plot_title, hypothesis_count_proportions, yerr_scaled_selected_hyps_for_plot, cut_off_point, text_size=1.6)









    #########################################################
    # BELOW THE PLOT SHOWING HOW MANY OBSERVATIONS IT TAKES TO LEARN THE PERSPECTIVE OVER GENERATIONS IS GENERATED:

    if recording == 'everything':

        min_n_data_points_per_agent_per_generation_per_run_p_prior_neutral_00_l_prior_neutral_00 = measur_n_data_points_for_p_learning(hypothesis_space, perspectives, perspective_probs, multi_run_log_posteriors_per_data_point_per_agent_per_generation_matrix, posterior_threshold)


        print "min_n_data_points_per_agent_per_generation_per_run_p_prior_neutral_00_l_prior_neutral_00 is:"
        print min_n_data_points_per_agent_per_generation_per_run_p_prior_neutral_00_l_prior_neutral_00


        plotting_time = time.clock()-t3
        print
        print 'plotting_time is:'
        print str((plotting_time/60))+" m"
