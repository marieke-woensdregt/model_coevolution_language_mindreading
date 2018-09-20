__author__ = 'Marieke Woensdregt'


import numpy as np
import pickle
from scipy import stats
import time

import context
import hypspace
import lex
import pop
import prior
import saveresults



# np.set_printoptions(threshold=np.nan)



#######################################################################################################################
# 1: THE PARAMETERS:


##!!!!!! MAKE SURE TO CHANGE THE PATHS BELOW TO MATCH THE FILE SYSTEM OF YOUR MACHINE:
directory = '/exports/eddie/scratch/s1370641/'


directory_laptop = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Eddie_Output/'

results_directory = '/exports/eddie/scratch/s1370641/'

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


random_parent = False
#######################################################################################################################



print ''
print ''
print "n_contexts are:"
print n_contexts
print ''
print ''


def get_avg_fitness_matrix(directory, filename, n_copies):
    if n_copies == 1:
        pickle_filename_all_results = 'Results_'+filename+copy_specification
        results_dict = pickle.load(open(directory+pickle_filename_all_results+'.p', 'rb'))
        avg_fitness_matrix_all_runs = results_dict['multi_run_avg_fitness_matrix']
    elif n_copies > 1:
        avg_fitness_matrix_all_runs = np.zeros(((n_copies*n_runs), n_iterations))
        counter = 0
        for c in range(1, n_copies+1):
            pickle_filename_all_results = 'Results_'+filename+'_c'+str(c)
            results_dict = pickle.load(open(directory+pickle_filename_all_results+'.p', 'rb'))
            for r in range(n_runs):
                multi_run_avg_fitness_matrix = results_dict['multi_run_avg_fitness_matrix'][r]
                avg_fitness_matrix_all_runs[counter] = multi_run_avg_fitness_matrix
                counter += 1
    return avg_fitness_matrix_all_runs




def calc_p_taking_success(multi_run_selected_hyps_per_generation_matrix, multi_run_selected_parent_indices_matrix, hypothesis_space, perspective_hyps, perspective_probs):
    avg_success_per_generation = np.zeros(len(multi_run_selected_parent_indices_matrix))
    parent_perspective_index = np.where(perspective_probs==1.0)[0][0]
    parent_perspective = perspective_hyps[parent_perspective_index]
    for i in range(len(multi_run_selected_hyps_per_generation_matrix)):
        if i == 0:
            avg_success_per_generation[i] = 'NaN'
        else:
            selected_hyps_per_agent = multi_run_selected_hyps_per_generation_matrix[i]
            success_per_agent = np.zeros(len(selected_hyps_per_agent))
            for a in range(len(selected_hyps_per_agent)):
                selected_hyp_index_agent = selected_hyps_per_agent[a]
                selected_hyp_agent = hypothesis_space[int(selected_hyp_index_agent)]
                learner_p_hyp = perspective_hyps[selected_hyp_agent[0]]
                if learner_p_hyp == parent_perspective:
                    success_per_agent[a] = 1.
                else:
                    success_per_agent[a] = 0.
            avg_success_per_generation[i] = np.mean(success_per_agent)
    return avg_success_per_generation






def calc_communication_success(multi_run_selected_hyps_per_generation_matrix, multi_run_selected_parent_indices_matrix, pragmatic_level_parent, pragmatic_level_learner, communication_type, ca_measure_type, n_interactions, hypothesis_space, perspective_hyps, lexicon_hyps, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant, perspective_probs, learner_perspective, learning_type_probs, sal_alpha, production_error):
    avg_success_per_generation = np.zeros(len(multi_run_selected_parent_indices_matrix))
    parent_perspective_index = np.where(perspective_probs==1.0)[0][0]
    parent_perspective = perspective_hyps[parent_perspective_index]
    learning_type_index = np.where(learning_type_probs==1.0)[0][0]
    learning_type = learning_types[learning_type_index]
    perspective_prior = prior.create_perspective_prior(perspective_hyps, lexicon_hyps, perspective_prior_type, learner_perspective, perspective_prior_strength)
    lexicon_prior = prior.create_lexicon_prior(lexicon_hyps, lexicon_prior_type, lexicon_prior_constant, error)
    composite_log_prior = prior.list_composite_log_priors(agent_type, pop_size, hypothesis_space, perspective_hyps, lexicon_hyps, perspective_prior, lexicon_prior)
    for i in range(len(multi_run_selected_hyps_per_generation_matrix)):
        if i == 0:
            avg_success_per_generation[i] = 'NaN'
        else:
            selected_hyps_per_agent = multi_run_selected_hyps_per_generation_matrix[i]
            if random_parent:
                random_parent_index = np.random.choice(len(multi_run_selected_hyps_per_generation_matrix))
                selected_hyps_per_parent = multi_run_selected_hyps_per_generation_matrix[random_parent_index]
            else:
                selected_hyps_per_parent = multi_run_selected_hyps_per_generation_matrix[i-1]
            selected_parent_indices_per_agent = multi_run_selected_parent_indices_matrix[i]
            success_per_agent = np.zeros(len(selected_hyps_per_agent))
            for a in range(len(selected_hyps_per_agent)):
                parent_index = selected_parent_indices_per_agent[a]
                selected_hyp_index_agent = selected_hyps_per_agent[a]
                selected_hyp_agent = hypothesis_space[int(selected_hyp_index_agent)]
                learner_lex_matrix = lexicon_hyps[selected_hyp_agent[1]]
                learner_lexicon = lex.Lexicon('specified_lexicon', n_meanings, n_signals, ambiguous_lex=None, specified_lexicon=learner_lex_matrix)
                selected_hyp_index_parent = selected_hyps_per_parent[int(parent_index)]
                selected_hyp_parent = hypothesis_space[int(selected_hyp_index_parent)]
                parent_lex_matrix = lexicon_hyps[selected_hyp_parent[1]]
                parent_lexicon = lex.Lexicon('specified_lexicon', n_meanings, n_signals, ambiguous_lex=None, specified_lexicon=parent_lex_matrix)
                if pragmatic_level_parent == 'literal' or pragmatic_level_parent == 'perspective-taking':
                    parent = pop.Agent(perspective_hyps, lexicon_hyps, composite_log_prior, composite_log_prior, parent_perspective, sal_alpha, parent_lexicon, 'sample')
                elif pragmatic_level_parent == 'prag':
                    parent = pop.PragmaticAgent(perspective_hyps, lexicon_hyps, composite_log_prior, composite_log_prior, parent_perspective, sal_alpha, parent_lexicon, learning_type, pragmatic_level_parent, pragmatic_level_learner, optimality_alpha, extra_error)
                if pragmatic_level_learner == 'literal' or pragmatic_level_learner == 'perspective-taking':
                    learner = pop.Agent(perspective_hyps, lexicon_hyps, composite_log_prior, composite_log_prior, learner_perspective, sal_alpha, learner_lexicon, learning_type)
                elif pragmatic_level_learner == 'prag':
                    learner = pop.PragmaticAgent(perspective_hyps, lexicon_hyps, composite_log_prior, composite_log_prior, learner_perspective, sal_alpha, learner_lexicon, learning_type, pragmatic_level_parent, pragmatic_level_learner, optimality_alpha, extra_error)
                context_matrix = context.gen_context_matrix('continuous', n_meanings, n_meanings, n_interactions)
                ca = learner.calc_comm_acc(context_matrix, communication_type, ca_measure_type, n_interactions, n_utterances, parent, learner, n_meanings, n_signals, sal_alpha, production_error, parent.pragmatic_level, speaker_p_hyp=parent.perspective, speaker_type_hyp=parent.pragmatic_level)
                success_per_agent[a] = ca
            avg_success_per_generation[i] = np.mean(success_per_agent)
    return avg_success_per_generation





def unpickle_and_calc_success(directory, filename, n_copies, success_type):
    if n_copies == 1:
        pickle_filename_all_results = 'Results_'+filename+copy_specification
        results_dict = pickle.load(open(directory+pickle_filename_all_results+'.p', 'rb'))
        multi_run_selected_hyps_per_generation_matrix = results_dict['multi_run_selected_hyps_per_generation_matrix']
        multi_run_selected_parent_indices_matrix = results_dict['multi_run_selected_parent_indices_matrix']
        if success_type == 'communication':
            avg_success_matrix_all_runs = calc_communication_success(multi_run_selected_hyps_per_generation_matrix, multi_run_selected_parent_indices_matrix, pragmatic_level, pragmatic_level, communication_type, ca_measure_type, n_interactions, hypothesis_space, perspective_hyps, lexicon_hyps, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant, perspective_probs, learner_perspective, learning_type_probs, sal_alpha, error)
        elif success_type == 'p_taking':
            avg_success_matrix_all_runs = calc_p_taking_success(multi_run_selected_hyps_per_generation_matrix, multi_run_selected_parent_indices_matrix, hypothesis_space, perspective_hyps, perspective_probs)
    elif n_copies > 1:
        avg_success_matrix_all_runs = np.zeros(((n_copies*n_runs), n_iterations))
        counter = 0
        for c in range(1, n_copies+1):
            pickle_filename_all_results = 'Results_'+filename+'_c'+str(c)
            results_dict = pickle.load(open(directory+pickle_filename_all_results+'.p', 'rb'))
            for r in range(n_runs):
                multi_run_selected_hyps_per_generation_matrix = results_dict['multi_run_selected_hyps_per_generation_matrix'][r]
                multi_run_selected_parent_indices_matrix = results_dict['multi_run_selected_parent_indices_matrix'][r]
                if success_type == 'p_taking':
                    avg_success_timecourse = calc_p_taking_success(multi_run_selected_hyps_per_generation_matrix, multi_run_selected_parent_indices_matrix, hypothesis_space, perspective_hyps, perspective_probs)
                elif success_type == 'communication':
                    avg_success_timecourse = calc_communication_success(multi_run_selected_hyps_per_generation_matrix, multi_run_selected_parent_indices_matrix, pragmatic_level, pragmatic_level, communication_type, ca_measure_type, n_interactions, hypothesis_space, perspective_hyps, lexicon_hyps, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant, perspective_probs, learner_perspective, learning_type_probs, sal_alpha, error)

                avg_success_matrix_all_runs[counter] = avg_success_timecourse
                counter += 1
    return avg_success_matrix_all_runs






def calc_mean_and_conf_invs_fitness_over_gens(avg_fitness_matrix_all_runs):
    mean_avg_fitness_over_gens = np.mean(avg_fitness_matrix_all_runs, axis=0)
    std_avg_fitness_over_gens = np.std(avg_fitness_matrix_all_runs, axis=0)
    conf_intervals_fitness_over_gens = stats.norm.interval(0.95, loc=mean_avg_fitness_over_gens, scale=std_avg_fitness_over_gens / np.sqrt(n_runs*n_copies))
    conf_intervals_fitness_over_gens = np.array(conf_intervals_fitness_over_gens)
    lower_yerr_fitness_over_gens_for_plot = np.subtract(mean_avg_fitness_over_gens, conf_intervals_fitness_over_gens[0])
    upper_yerr_fitness_over_gens_for_plot = np.subtract(conf_intervals_fitness_over_gens[1], mean_avg_fitness_over_gens)
    yerr_fitness_over_gens_for_plot = np.array([lower_yerr_fitness_over_gens_for_plot, upper_yerr_fitness_over_gens_for_plot])
    return mean_avg_fitness_over_gens, yerr_fitness_over_gens_for_plot



def calc_percentiles_fitness_over_gens(avg_fitness_matrix_all_runs):
    percentile_25_fitness_over_gens = np.percentile(avg_fitness_matrix_all_runs, 25, axis=0)
    median_fitness_over_gens = np.percentile(avg_fitness_matrix_all_runs, 50, axis=0)
    percentile_75_fitness_over_gens = np.percentile(avg_fitness_matrix_all_runs, 75, axis=0)
    percentiles_fitness_over_gens = np.array([percentile_25_fitness_over_gens, median_fitness_over_gens, percentile_75_fitness_over_gens])
    return percentiles_fitness_over_gens



def calc_and_pickle_mean_percentiles_avg_fitness(directory, n_runs, n_copies, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type):
    perspective_prior_strength_string = saveresults.convert_array_to_string(perspective_prior_strength)
    if selection_type == 'none' or selection_type == 'p_taking':
        if context_generation == 'random':
            filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_select_'+selection_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_select_'+selection_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    elif selection_type == 'ca_with_parent':
        if context_generation == 'random':
            filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type

        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            filename = run_type+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_'+agent_type+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    ##
    avg_fitness_matrix_all_runs = get_avg_fitness_matrix(directory, filename, n_copies)
    print ''
    print ''
    # print "avg_fitness_matrix_all_runs is:"
    # print avg_fitness_matrix_all_runs
    print "avg_fitness_matrix_all_runs.shape is:"
    print avg_fitness_matrix_all_runs.shape

    avg_p_taking_success_matrix_all_runs = unpickle_and_calc_success(directory, filename, n_copies, 'p_taking')
    print ''
    print ''
    # print "avg_p_taking_success_matrix_all_runs is:"
    # print avg_p_taking_success_matrix_all_runs
    print "avg_p_taking_success_matrix_all_runs.shape is:"
    print avg_p_taking_success_matrix_all_runs.shape


    avg_communication_success_matrix_all_runs = unpickle_and_calc_success(directory, filename, n_copies, 'communication')
    print ''
    print ''
    # print "avg_communication_success_matrix_all_runs is:"
    # print avg_communication_success_matrix_all_runs
    print "avg_communication_success_matrix_all_runs.shape is:"
    print avg_communication_success_matrix_all_runs.shape


    if selection_type == 'ca_with_parent':
        difference_array = np.subtract(avg_fitness_matrix_all_runs, avg_communication_success_matrix_all_runs)
    elif selection_type == 'p_taking':
        difference_array = np.subtract(avg_fitness_matrix_all_runs, avg_p_taking_success_matrix_all_runs)

    if selection_type != 'none':
        tot_difference = np.sum(np.absolute(difference_array[:][1:]))

        print ''
        print ''
        # print "difference_array is:"
        # print difference_array
        print "difference_array.shape is:"
        print difference_array.shape

        print ''
        print "tot_difference is:"
        print tot_difference

    mean_fitness_over_gens, conf_invs_fitness_over_gens = calc_mean_and_conf_invs_fitness_over_gens(avg_fitness_matrix_all_runs)
    percentiles_fitness_over_gens = calc_percentiles_fitness_over_gens(avg_fitness_matrix_all_runs)
    fitness_timecourse_data = {'raw_data':avg_fitness_matrix_all_runs,
        'mean_over_gens':mean_fitness_over_gens, 'conf_invs_over_gens':conf_invs_fitness_over_gens, 'percentiles_over_gens':percentiles_fitness_over_gens}

    mean_communication_success_over_gens, conf_invs_communication_success_over_gens = calc_mean_and_conf_invs_fitness_over_gens(avg_communication_success_matrix_all_runs)
    percentiles_communication_success_over_gens = calc_percentiles_fitness_over_gens(avg_communication_success_matrix_all_runs)
    communication_success_timecourse_data = {'raw_data':avg_communication_success_matrix_all_runs,
        'mean_over_gens':mean_communication_success_over_gens, 'conf_invs_over_gens':conf_invs_communication_success_over_gens, 'percentiles_over_gens':percentiles_communication_success_over_gens}


    mean_p_taking_success_over_gens, conf_invs_p_taking_success_over_gens = calc_mean_and_conf_invs_fitness_over_gens(avg_p_taking_success_matrix_all_runs)
    percentiles_p_taking_success_over_gens = calc_percentiles_fitness_over_gens(avg_p_taking_success_matrix_all_runs)
    p_taking_success_timecourse_data = {'raw_data':avg_p_taking_success_matrix_all_runs,
        'mean_over_gens':mean_p_taking_success_over_gens, 'conf_invs_over_gens':conf_invs_p_taking_success_over_gens, 'percentiles_over_gens':percentiles_p_taking_success_over_gens}


    print ''
    print ''
    # print "mean_fitness_over_gens is:"
    # print mean_fitness_over_gens
    print "mean_fitness_over_gens.shape is:"
    print mean_fitness_over_gens.shape

    print ''
    print ''
    # print "mean_communication_success_over_gens is:"
    # print mean_communication_success_over_gens
    print "mean_communication_success_over_gens.shape is:"
    print mean_communication_success_over_gens.shape

    difference_means = np.subtract(mean_fitness_over_gens, mean_communication_success_over_gens)
    print ''
    print ''
    # print "difference_means is:"
    # print difference_means

    tot_diff_means = np.sum(np.abs(difference_means[2:]))
    print "tot_diff_means is:"
    print tot_diff_means

    ##
    if selection_type == 'none' or selection_type == 'p_taking':
        if context_generation == 'random':
            filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type+'_random_'+str(random_parent)
        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type+'_random_'+str(random_parent)
    elif selection_type == 'ca_with_parent':
        if context_generation == 'random':
            filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type+'_random_'+str(random_parent)
        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            filename_short = run_type[0:4]+'_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs*n_copies)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type+'_random_'+str(random_parent)

    pickle.dump(fitness_timecourse_data, open(results_directory+'fitness_tmcrse_'+filename_short+'.p', "wb"))
    pickle.dump(communication_success_timecourse_data, open(results_directory+'comm_success_tmcrse_'+filename_short+'.p', "wb"))
    pickle.dump(p_taking_success_timecourse_data, open(results_directory + 'p_taking_success_tmcrse_' + filename_short + '.p', "wb"))




#####################################################################################

if __name__ == "__main__":



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
#     selection_type = 'ca_with_parent'
#
#     folder = 'results_iter_'+str(n_meanings)+'M_'+str(n_signals)+'S_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+ '_p_prior_exo_10_l_prior_' + lexicon_prior_type[0:4]
#     print ''
#     print "folder is:"
#     print folder
#
#     output_pickle_file_directory = directory_laptop+folder+'/'
#     print ''
#     print "output_pickle_file_directory is:"
#     print output_pickle_file_directory
#
#     calc_and_pickle_mean_percentiles_avg_fitness(n_runs, n_copies, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type)
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
#     selection_type = 'p_taking'
#
#     folder = 'results_iter_'+str(n_meanings)+'M_'+str(n_signals)+'S_select_'+selection_type+'_p_prior_exo_10_l_prior_'+lexicon_prior_type[0:4]
#     print ''
#     print "folder is:"
#     print folder
#
#     output_pickle_file_directory = directory_laptop+folder+'/'
#     print ''
#     print "output_pickle_file_directory is:"
#     print output_pickle_file_directory
#
#     calc_and_pickle_mean_percentiles_avg_fitness(n_runs, n_copies, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type)
#


# #####################################################################################
# ## PRIOR 3: UNIFORM
# ### CONDITION 1: No selection:
#
#     print ''
#     print ''
#     print 'This is prior 3: Uniform'
#     print ''
#     print 'This is condition 1: No selection'
#
#     perspective_prior_type = 'neutral'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
#     perspective_prior_strength = 0.0  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
#     selection_type = 'none'
#
#     folder = 'results_iter_'+str(n_meanings)+'M_'+str(n_signals)+'S_select_'+selection_type+'_p_prior_neut_00_l_prior_' + lexicon_prior_type[0:4]+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]
#     print ''
#     print "folder is:"
#     print folder
#
#     output_pickle_file_directory = directory_laptop+folder+'/'
#     print ''
#     print "output_pickle_file_directory is:"
#     print output_pickle_file_directory
#
#     calc_and_pickle_mean_percentiles_avg_fitness(n_runs, n_copies, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type)
#
#
#
#
# #####################################################################################
# ## PRIOR 3: UNIFORM
# ### CONDITION 2: Selection on CS:
#
#     print ''
#     print ''
#     print 'This is prior 3: Uniform'
#     print ''
#     print 'This is condition 2: Selection on CS'
#
#     perspective_prior_type = 'neutral'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
#     perspective_prior_strength = 0.0  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
#     selection_type = 'ca_with_parent'
#
#     folder = 'results_iter_'+str(n_meanings)+'M_'+str(n_signals)+'S_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+ '_p_prior_neut_00_l_prior_' + lexicon_prior_type[0:4]+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]
#     print ''
#     print "folder is:"
#     print folder
#
#     output_pickle_file_directory = directory_laptop+folder+'/'
#     print ''
#     print "output_pickle_file_directory is:"
#     print output_pickle_file_directory
#
#     calc_and_pickle_mean_percentiles_avg_fitness(n_runs, n_copies, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type)
#
#
#
#
# #####################################################################################
# ## PRIOR 3: UNIFORM
# ### CONDITION 3: Selection on P-taking:
#
#     print ''
#     print ''
#     print 'This is prior 3: Uniform'
#     print ''
#     print 'This is condition 3: Selection on P-taking'
#
#     perspective_prior_type = 'neutral'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
#     perspective_prior_strength = 0.0  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
#     selection_type = 'p_taking'
#
#
#     folder = 'results_iter_'+str(n_meanings)+'M_'+str(n_signals)+'S_select_'+selection_type+'_p_prior_neut_00_l_prior_'+lexicon_prior_type[0:4]+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]
#     print ''
#     print "folder is:"
#     print folder
#
#
#     output_pickle_file_directory = directory_laptop+folder+'/'
#     print ''
#     print "output_pickle_file_directory is:"
#     print output_pickle_file_directory
#
#     calc_and_pickle_mean_percentiles_avg_fitness(n_runs, n_copies, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type)
#
#


#####################################################################################
## PRIOR 3: EGOCENTRIC
### CONDITION 1: No selection:

    print ''
    print ''
    print 'This is prior 3: Egocentric'
    print ''
    print 'This is condition 1: No selection'

    perspective_prior_type = 'egocentric'  # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
    perspective_prior_strength = 0.9  # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
    selection_type = 'none'


    folder = 'results_iter_'+str(n_meanings)+'M_'+str(n_signals)+'S_select_'+selection_type+'_p_prior_egoc_09_l_prior_' + lexicon_prior_type[0:4]+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]
    print ''
    print "folder is:"
    print folder

    output_pickle_file_directory = directory_laptop+folder+'/'
    print ''
    print "output_pickle_file_directory is:"
    print output_pickle_file_directory

    calc_and_pickle_mean_percentiles_avg_fitness(directory_laptop, n_runs, n_copies, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type)




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
    selection_type = 'ca_with_parent'
    if pragmatic_level == 'literal':
        pragmatic_level = 'perspective-taking'

    folder = 'results_iter_'+str(n_meanings)+'M_'+str(n_signals)+'S_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+ '_p_prior_egoc_09_l_prior_' + lexicon_prior_type[0:4]+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]
    print ''
    print "folder is:"
    print folder

    directory = directory_laptop+folder+'/'
    print ''
    print "output_pickle_file_directory is:"
    print directory

    t0 = time.clock()

    calc_and_pickle_mean_percentiles_avg_fitness(directory_laptop, n_runs, n_copies, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type)

    calc_time_ego_select_ca = time.clock() - t0
    print ''
    print ''
    print "calc_time_ego_select_ca is:"
    print calc_time_ego_select_ca


    if pragmatic_level == 'perspective-taking':
        pragmatic_level = 'literal'

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
    selection_type = 'p_taking'


    folder = 'results_iter_'+str(n_meanings)+'M_'+str(n_signals)+'S_select_'+selection_type+'_p_prior_egoc_09_l_prior_'+lexicon_prior_type[0:4]+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]
    print ''
    print "folder is:"
    print folder


    directory = directory_laptop+folder+'/'
    print ''
    print "output_pickle_file_directory is:"
    print directory

    t1 = time.clock()

    calc_and_pickle_mean_percentiles_avg_fitness(directory_laptop, n_runs, n_copies, n_contexts, perspective_prior_type, perspective_prior_strength, selection_type, communication_type, ca_measure_type)

    calc_time_ego_select_p_taking = time.clock() - t1
    print ''
    print ''
    print "calc_time_ego_select_p_taking is:"
    print calc_time_ego_select_p_taking
