__author__ = 'Marieke Woensdregt'


import lex
import hypspace
import numpy as np
import pickle
from hypspace import create_all_lexicons, list_hypothesis_space
from lex import calc_ca_all_lexicons, Lexicon
from saveresults import convert_array_to_string
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import itertools




np.set_printoptions(threshold=np.nan)


#######################################################################################################################
# STEP 2: SOME PARAMETER RESETTING:


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


error = 0.05  # The error term on production
error_string = convert_array_to_string(error)
extra_error = True  # Determines whether the error specified above gets added on top AFTER the pragmatic speaker has calculated its production probabilities by maximising the utility to the listener.



# 2.3: The parameters that determine the make-up of the population:

pop_size_initial_pop = 100
pop_size_mixed_pop = 100
n_mutants = 1  # the number of mutants initially inserted into the first generation of the mixed population


lexicon_types = ['optimal_lex', 'half_ambiguous_lex', 'fully_ambiguous_lex'] # The lexicon types that can be present in the population
#FIXME: In the current implementation 'half_ambiguous_lex' can have different instantiations. Therefore, if we run a simulation where there are different lexicon_type_probs, we will want the run_type to be 'population_same_pop' to make sure that all the 'half ambiguous' speakers do have the SAME half ambiguous lexicon.
lexicon_type_probs = np.array([0., 0., 1.]) # The ratios with which the different lexicon types will be present in the population
lexicon_type_probs_string = convert_array_to_string(lexicon_type_probs) # Turns the lexicon type probs into a string in order to add it to file names


perspectives = np.array([0., 1.]) # The different perspectives that agents can have
perspective_probs = np.array([0., 1.]) # The ratios with which the different perspectives will be present in the population
perspective_probs_string = convert_array_to_string(perspective_probs) # Turns the perspective probs into a string in order to add it to file names


learning_types = ['map', 'sample'] # The types of learning that the learners can do
learning_type_probs = np.array([0., 1.]) # The ratios with which the different learning types will be present in the population
learning_type_probs_string = convert_array_to_string(learning_type_probs) # Turns the learning type probs into a string in order to add it to file names
if learning_type_probs[0] == 1.:
    learning_type_string = learning_types[0]
elif learning_type_probs[1] == 1.:
    learning_type_string = learning_types[1]
#learning_type_string = learning_types[np.where(learning_type_probs==1.)[0]]


pragmatic_level_initial_pop = 'perspective-taking'  # This can be set to either 'literal', 'perspective-taking' or 'prag'
optimality_alpha_initial_pop = 1.0  # Goodman & Stuhlmuller (2013) fitted sal_alpha = 3.4 to participant data with 4x3 lexicon of three number words ('one', 'two', and 'three') that could be mapped to four different world states (0, 1, 2, and 3)


pragmatic_level_mutants = 'prag'  # This can be set to either 'literal', 'perspective-taking' or 'prag'
optimality_alpha_mutants = 3.0  # Goodman & Stuhlmuller (2013) fitted sal_alpha = 3.4 to participant data with 4x3 lexicon of three number words ('one', 'two', and 'three') that could be mapped to four different world states (0, 1, 2, and 3)


pragmatic_level_parent_hyp = pragmatic_level_mutants  # This can be set to either 'literal', 'perspective-taking' or 'prag'

teacher_type = 'sng_teacher'  # This can be set to either 'sng_teacher' or 'multi_teacher'



# 2.6: The parameters that determine the learner's hypothesis space:

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


# 2.7: The parameters that determine the learner's prior:

learner_perspective = 0.  # The learner's perspective

perspective_prior_type = 'egocentric' # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
perspective_prior_strength = 0.9 # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
perspective_prior_strength_string = convert_array_to_string(perspective_prior_strength)


lexicon_prior_type = 'neutral' # This can be set to either 'neutral', 'ambiguous_fixed', 'half_ambiguous_fixed', 'expressivity_bias' or 'compressibility_bias'
lexicon_prior_constant = 0.0 # Determines the strength of the lexicon bias, with small c creating a STRONG prior and large c creating a WEAK prior. (And with c = 1000 creating an almost uniform prior.)
lexicon_prior_constant_string = convert_array_to_string(lexicon_prior_constant)

n_utterances = 1  # This parameter determines how many signals the learner gets to observe in each context
n_contexts = 120  # The number of contexts that the learner gets to see. If context_generation = 'optimal' n_contexts for n_meanings = 2 can be anything in the table of 4 (e.g. 20, 40, 60, 80, 100, etc.); for n_meanings = 3 anything in the table of 12 (e.g. 12, 36, 60, 84, 108, etc.); and for n_meanings = 4 anything in the table of 48 (e.g. 48, 96, 144, 192 etc.).
n_contexts_select_none = 0

turnover_type = 'whole_pop'  # The type of turnover with which the population is replaced (for the iteration simulation). This can be set to either 'chain' (one agent at a time) or 'whole_pop' (entire population at once)


communication_type_initial_pop = 'lex_n_p'  # This can be set to either 'lex_only', 'lex_n_context' or 'lex_n_p' or 'prag'
ca_measure_type_initial_pop = 'comp_only'  # This can be set to either "comp_n_prod" or "comp_only"

communication_type_mutants = 'prag'  # This can be set to either 'lex_only', 'lex_n_context' or 'lex_n_p' or 'prag'
ca_measure_type_mutants = 'comp_only'  # This can be set to either "comp_n_prod" or "comp_only"
n_interactions = 6  # The number of interactions used to calculate communicative accuracy

selection_type = 'p_taking'  # This can be set to either 'none', 'p_taking', 'l_learning' or 'ca_with_parent'
selection_weighting = 'none'  # This is a factor with which the fitness of the agents (determined as the probability they assign to the correct perspective hypothesis) is multiplied and then exponentiated in order to weight the relative agent fitness (which in turn determines the probability of becoming a teacher for the next generation). A value of 0. implements neutral selection. A value of 1.0 creates weighting where the fitness is pretty much equal to relative posterior probability on correct p hyp), and the higher the value, the more skewed the weighting in favour of agents with better perspective-taking.
if isinstance(selection_weighting, float):
    selection_weight_string = str(np.int(selection_weighting))
else:
    selection_weight_string = selection_weighting


n_iterations_initial_pop = 500  # The number of iterations (i.e. new agents in the case of 'chain', generations in the case of 'whole_pop')
n_iterations_mixed_pop = 200  # The number of iterations (i.e. new agents in the case of 'chain', generations in the case of 'whole_pop')


lex_measure = 'ca'  # This can be set to either 'mi' for mutual information or 'ca' for communicative accuracy (of the lexicon with itself)


run_type = 'iteration'

# iteration_results_directory = '/exports/eddie/scratch/s1370641/'

iteration_results_directory = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Eddie_Output/'

results_directory = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/Evolvability_Analysis/'

plot_directory = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Plots/Evolvability_Analysis/'

n_runs = 200

decoupling = True  # This can be set to either True or False. It determines whether genetic and cultural inheritance are coupled (i.e. from the same cultural parent) or decoupled.


line_style = 'indiv_runs'  # this can be set to either 'divided_by_genes', 'unit_traces' or 'indiv_runs'
legend = False  # either true or false


s_type_hyp_l0 = 'literal'
s_type_hyp_l1 = 'literal'
s_type_hyp_l2 = 'prag'

#######################################################################################################################



def unpickle_results_multi_runs(n_runs, selection_type):


    pragmatic_level_per_agent_matrix_per_run = [[['' for a in range(pop_size_mixed_pop)] for i in range(n_iterations_mixed_pop)] for r in range(n_runs)]
    selected_parent_indices_matrix_per_run = np.zeros((n_runs, n_iterations_mixed_pop, pop_size_mixed_pop))
    avg_fitness_per_gen_per_run = np.zeros((n_runs, n_iterations_mixed_pop))
    selected_hyps_per_generation_matrix_per_run = np.zeros((n_runs, n_iterations_mixed_pop, pop_size_mixed_pop))


    if selection_type == 'none' or selection_type == 'p_taking':
        folder = 'Results_evo_select_'+selection_type+'_init_pop_'+pragmatic_level_initial_pop+'_a_'+str(optimality_alpha_initial_pop)[0]+'_mutnts_'+pragmatic_level_mutants+'_a_'+str(optimality_alpha_mutants)[0]+'_s_hyp_'+pragmatic_level_parent_hyp+'_p_prior_'+str(perspective_prior_type)[0:4]+'_xtra_err_'+str(extra_error)[0:3]+'/'


    elif selection_type == 'ca_with_parent':
        folder = 'Results_evo_select_'+selection_type+'_init_'+communication_type_initial_pop+'_'+ca_measure_type_initial_pop+'_mutnts_'+communication_type_mutants+'_'+ca_measure_type_mutants+'_init_pop_'+pragmatic_level_initial_pop+'_a_'+str(optimality_alpha_initial_pop)[0]+'_mutnts_'+pragmatic_level_mutants+'_a_'+str(optimality_alpha_mutants)[0]+'_s_hyp_'+pragmatic_level_parent_hyp+'_p_prior_'+str(perspective_prior_type)[0:4]+'_xtra_err_'+str(extra_error)[0:3]+'/'


    for r in range(n_runs):

        if context_generation == 'random':
            if selection_type == 'none' or selection_type == 'l_learning':
                filename = 'evo_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size_mixed_pop)+'_select_'+selection_type+'_'+str(n_iterations_mixed_pop)+'_I_'+str(n_contexts)+'_CO_'+str(n_contexts_select_none)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_xtra_err_'+str(extra_error)[0:3]+'_init_pop_'+pragmatic_level_initial_pop+'_a_'+str(optimality_alpha_initial_pop)[0]+'_mutnts_'+pragmatic_level_mutants+'_a_'+str(optimality_alpha_mutants)[0]+'_s_hyp_'+pragmatic_level_parent_hyp+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_dcpl_'+str(decoupling)[0:3]+'_r'+str(r)
            elif selection_type == 'p_taking':
                filename = 'evo_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size_mixed_pop)+'_select_'+selection_type+'_'+str(n_iterations_mixed_pop)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_xtra_err_'+str(extra_error)[0:3]+pragmatic_level_initial_pop+'_a_'+str(optimality_alpha_initial_pop)[0]+'_mutnts_'+pragmatic_level_mutants+'_a_'+str(optimality_alpha_mutants)[0]+'_s_hyp_'+pragmatic_level_parent_hyp+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_dcpl_'+str(decoupling)[0:3]+'_r'+str(r)
            elif selection_type == 'ca_with_parent':
                filename = 'evo_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size_mixed_pop)+'_select_'+selection_type+'_init_'+communication_type_initial_pop+'_'+ca_measure_type_initial_pop+'_mutnts_'+communication_type_mutants+'_'+ca_measure_type_mutants+'_'+str(n_iterations_mixed_pop)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_xtra_err_'+str(extra_error)[0:3]+'_init_pop_'+pragmatic_level_initial_pop+'_a_'+str(optimality_alpha_initial_pop)[0]+'_mutnts_'+pragmatic_level_mutants+'_a_'+str(optimality_alpha_mutants)[0]+'_s_hyp_'+pragmatic_level_parent_hyp+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_dcpl_'+str(decoupling)[0:3]+'_r'+str(r)

        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            if selection_type == 'none' or selection_type == 'l_learning':
                filename = 'evo_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size_mixed_pop)+'_select_'+selection_type+'_'+str(n_iterations_mixed_pop)+'_I_'+str(n_contexts)+'_CO_'+str(n_contexts_select_none)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_xtra_err_'+str(extra_error)[0:3]+'_init_pop_'+pragmatic_level_initial_pop+'_a_'+str(optimality_alpha_initial_pop)[0]+'_mutnts_'+pragmatic_level_mutants+'_a_'+str(optimality_alpha_mutants)[0]+'_s_hyp_'+pragmatic_level_parent_hyp+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_dcpl_'+str(decoupling)[0:3]+'_r'+str(r)
            elif selection_type == 'p_taking':
                filename = 'evo_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size_mixed_pop)+'_select_'+selection_type+'_'+str(n_iterations_mixed_pop)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_xtra_err_'+str(extra_error)[0:3]+'_init_pop_'+pragmatic_level_initial_pop+'_a_'+str(optimality_alpha_initial_pop)[0]+'_mutnts_'+pragmatic_level_mutants+'_a_'+str(optimality_alpha_mutants)[0]+'_s_hyp_'+pragmatic_level_parent_hyp+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_dcpl_'+str(decoupling)[0:3]+'_r'+str(r)
            elif selection_type == 'ca_with_parent':
                filename = 'evo_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size_mixed_pop)+'_select_'+selection_type+'_init_'+communication_type_initial_pop+'_'+ca_measure_type_initial_pop+'_mutnts_'+communication_type_mutants+'_'+ca_measure_type_mutants+'_'+str(n_iterations_mixed_pop)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+error_string+'_xtra_err_'+str(extra_error)[0:3]+'_init_pop_'+pragmatic_level_initial_pop+'_a_'+str(optimality_alpha_initial_pop)[0]+'_mutnts_'+pragmatic_level_mutants+'_a_'+str(optimality_alpha_mutants)[0]+'_s_hyp_'+pragmatic_level_parent_hyp+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_dcpl_'+str(decoupling)[0:3]+'_r'+str(r)

        pickle_file_title_all_results = iteration_results_directory+folder+'Results_'+filename

        results_dictionary = pickle.load(open(pickle_file_title_all_results+'.p', 'rb'))

        selected_hyps_per_generation_matrix = results_dictionary['selected_hyps_per_generation_matrix']


        selected_hyps_per_generation_matrix_per_run[r] = selected_hyps_per_generation_matrix

        avg_fitness_matrix = results_dictionary['avg_fitness_matrix']

        avg_fitness_per_gen_per_run[r] = avg_fitness_matrix

        parent_probs_matrix = results_dictionary['parent_probs_matrix']

        selected_parent_indices_matrix = results_dictionary['selected_parent_indices_matrix']

        selected_parent_indices_matrix_per_run[r] = selected_parent_indices_matrix

        parent_lex_indices_matrix = results_dictionary['parent_lex_indices_matrix']

        pragmatic_level_per_agent_matrix = results_dictionary['pragmatic_level_per_agent_matrix']

        pragmatic_level_per_agent_matrix_per_run[r] = pragmatic_level_per_agent_matrix

        pickle_file_title_max_offspring_single_parent = iteration_results_directory+folder+'Max_Offspr_'+filename

        # proportion_max_offspring_single_parent = pickle.load(open(pickle_file_title_max_offspring_single_parent+'.p', 'rb'))


    return pragmatic_level_per_agent_matrix_per_run, avg_fitness_per_gen_per_run, selected_hyps_per_generation_matrix_per_run, selected_parent_indices_matrix_per_run




def unpickle_ca_per_lex():

    results_directory = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/Learner_Speaker/'

    n_interactions = 1000

    context_generation = 'random'

    print ''
    print ''
    # print "lexicon_hyps are:"
    # print lexicon_hyps
    print "lexicon_hyps.shape are:"
    print lexicon_hyps.shape



    informativity_per_lexicon = calc_ca_all_lexicons(lexicon_hyps, error, lex_measure)
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
    print "argsort_informativity_per_lexicon.shape is:"
    print argsort_informativity_per_lexicon.shape


    informativity_per_lexicon_sorted = informativity_per_lexicon_rounded[argsort_informativity_per_lexicon]
    print ''
    print ''
    # print "informativity_per_lexicon_sorted is:"
    # print informativity_per_lexicon_sorted
    print "informativity_per_lexicon_sorted.shape is:"
    print informativity_per_lexicon_sorted.shape


    lexicon_hyps_sorted = lexicon_hyps[argsort_informativity_per_lexicon]
    print ''
    print ''
    # print "lexicon_hyps_sorted is:"
    # print lexicon_hyps_sorted
    print "lexicon_hyps_sorted.shape is:"
    print lexicon_hyps_sorted.shape




    # lexicon_hyps_sorted_unpickled = pickle.load(open(output_pickle_file_directory+'lexicon_hyps_sorted.p', "rb" ))
    # print ' '
    # # print "lexicon_hyps_sorted_unpickled  is:"
    # # print lexicon_hyps_sorted_unpickled
    # print "lexicon_hyps_sorted_unpickled.shape is:"
    # print lexicon_hyps_sorted_unpickled.shape
    #
    #
    # informativity_per_lexicon_sorted_unpickled = pickle.load(open(output_pickle_file_directory+'informativity_per_lexicon_sorted.p', "rb" ))
    # print ' '
    # # print "informativity_per_lexicon_sorted_unpickled is:"
    # # print informativity_per_lexicon_sorted_unpickled
    # print "informativity_per_lexicon_sorted_unpickled.shape is:"
    # print informativity_per_lexicon_sorted_unpickled.shape
    #



    unique_informativity_per_lexicon = np.unique(informativity_per_lexicon_sorted)
    print ''
    print ''
    print "unique_informativity_per_lexicon is:"
    print unique_informativity_per_lexicon
    print "unique_informativity_per_lexicon.shape is:"
    print unique_informativity_per_lexicon.shape

    ca_s0p1_l1p0_per_lex_array = pickle.load(open(results_directory+'ca_s0p1_l1p0_per_lex_array_sp_type_hyp_'+s_type_hyp_l1+'_'+context_generation+'_'+str(n_interactions)+'_interactions_'+lex_measure+'_numerical.p', "rb"))
    print ' '
    print ' '
    # print "ca_s0p1_l1p0_per_lex_array is:"
    # print ca_s0p1_l1p0_per_lex_array
    print "ca_s0p1_l1p0_per_lex_array.shape is:"
    print ca_s0p1_l1p0_per_lex_array.shape

    ca_s0p1_l1p1_per_lex_array = pickle.load(open(results_directory+'ca_s0p1_l1p1_per_lex_array_sp_type_hyp_'+s_type_hyp_l1+'_'+context_generation+'_'+str(n_interactions)+'_interactions_'+lex_measure+'_numerical.p', "rb"))
    print ' '
    print ' '
    # print "ca_s0p1_l1p1_per_lex_array is:"
    # print ca_s0p1_l1p1_per_lex_array
    print "ca_s0p1_l1p1_per_lex_array.shape is:"
    print ca_s0p1_l1p1_per_lex_array.shape



    ca_s0_l1_per_lex_average = np.mean(np.array([ca_s0p1_l1p0_per_lex_array, ca_s0p1_l1p1_per_lex_array]), axis=0)
    print ''
    print ''
    # print "ca_s0_l1_per_lex_average is:"
    # print ca_s0_l1_per_lex_average
    print "ca_s0_l1_per_lex_average.shape is:"
    print ca_s0_l1_per_lex_average.shape



    ca_s1p1_l2p0_per_lex_array = pickle.load(open(results_directory+'ca_s1p1_l2p0_per_lex_array_a_'+str(optimality_alpha_mutants)+'_sp_type_hyp_'+s_type_hyp_l2+'_'+context_generation+'_'+str(n_interactions)+'_interactions_'+lex_measure+'_numerical.p', "rb"))
    print ' '
    print ' '
    # print "ca_s1p1_l2p0_per_lex_array is:"
    # print ca_s1p1_l2p0_per_lex_array
    print "ca_s1p1_l2p0_per_lex_array.shape is:"
    print ca_s1p1_l2p0_per_lex_array.shape


    ca_s1p1_l2p1_per_lex_array = pickle.load(open(results_directory+'ca_s1p1_l2p1_per_lex_array_a_'+str(optimality_alpha_mutants)+'_sp_type_hyp_'+s_type_hyp_l2+'_'+context_generation+'_'+str(n_interactions)+'_interactions_'+lex_measure+'_numerical.p', "rb"))
    print ' '
    # print "ca_s1p1_l2p1_per_lex_array is:"
    # print ca_s1p1_l2p1_per_lex_array
    print "ca_s1p1_l2p1_per_lex_array.shape is:"
    print ca_s1p1_l2p1_per_lex_array.shape


    ca_s1_l2_per_lex_average = np.mean(np.array([ca_s1p1_l2p0_per_lex_array, ca_s1p1_l2p1_per_lex_array]), axis=0)
    print ''
    print ''
    # print "ca_s1_l2_per_lex_average is:"
    # print ca_s1_l2_per_lex_average
    print "ca_s1_l2_per_lex_average.shape is:"
    print ca_s1_l2_per_lex_average.shape



    minimum_advantage_array = np.subtract(ca_s1p1_l2p0_per_lex_array, ca_s0p1_l1p1_per_lex_array)
    print "minimum_advantage_array.shape is:"
    print minimum_advantage_array.shape
    min_advantage = np.amin(minimum_advantage_array)
    print ''
    print ''
    print "min_advantage is:"
    print min_advantage


    maximum_advantage_array = np.subtract(ca_s1p1_l2p1_per_lex_array, ca_s0p1_l1p0_per_lex_array)
    max_advantage = np.amax(maximum_advantage_array)
    print ''
    print ''
    print "max_advantage is:"
    print max_advantage

    pragmatic_advantage_per_lex = np.subtract(ca_s1_l2_per_lex_average, ca_s0_l1_per_lex_average)
    print ''
    print ''
    # print "pragmatic_advantage_per_lex is:"
    # print pragmatic_advantage_per_lex
    print "pragmatic_advantage_per_lex.shape is:"
    print pragmatic_advantage_per_lex.shape

    return pragmatic_advantage_per_lex, min_advantage, max_advantage


def calc_proportion_pragmatic_level_per_run(pragmatic_level_per_agent_matrix_per_run, n_runs, n_iterations, pop_size, pragmatic_level):
    proportion_pragmatic_level_over_gens_per_run = np.zeros((n_runs, n_iterations))
    for r in range(n_runs):
        print "this is run number:"
        print r
        for i in range(n_iterations):
            if i == n_iterations-1:
                print "i == n_iterations-1!"
                print "pragmatic_level_per_agent_matrix_per_run[r][i] is:"
                print pragmatic_level_per_agent_matrix_per_run[r][i]
            for a in range(pop_size):
                if pragmatic_level_per_agent_matrix_per_run[r][i][a] == pragmatic_level:
                    proportion_pragmatic_level_over_gens_per_run[r][i] += 1.
            proportion_pragmatic_level_over_gens_per_run[r][i] = np.divide(float(proportion_pragmatic_level_over_gens_per_run[r][i]), float(pop_size))
    print ''
    print ''
    print "proportion_pragmatic_level_over_gens_per_run.shape is:"
    print proportion_pragmatic_level_over_gens_per_run.shape
    final_proportion_pragmatic_level_per_run = proportion_pragmatic_level_over_gens_per_run[:,-1]
    print "final_proportion_pragmatic_level_per_run is:"
    print final_proportion_pragmatic_level_per_run
    print "final_proportion_pragmatic_level_per_run.shape is:"
    print final_proportion_pragmatic_level_per_run.shape
    print "np.argwhere(final_proportion_pragmatic_level_per_run == 1.0) is:"
    print np.argwhere(final_proportion_pragmatic_level_per_run==1.0)
    return proportion_pragmatic_level_over_gens_per_run




def calc_no_pragmatic_parents_per_pragmatic_agent(pragmatic_level_per_agent_matrix_per_run, n_runs, n_iterations, pop_size, pragmatic_level, selected_parent_indices_matrix_per_run):
    binary_pragmatic_level_per_agent_matrix_per_run = np.zeros((n_runs, n_iterations, pop_size))
    pragmatic_parents_selected_count_matrix = np.zeros((n_runs, (n_iterations-1)))
    for r in range(n_runs):
        for i in range(n_iterations-1):
            pragmatic_parents_selected_count = 0
            parent_index_counts = np.bincount(selected_parent_indices_matrix_per_run[r][i+1].astype(int), minlength=pop_size)
            for a in range(pop_size):
                if pragmatic_level_per_agent_matrix_per_run[r][i][a] == pragmatic_level:
                    binary_pragmatic_level_per_agent_matrix_per_run[r][i][a] = 1.
            pragmatic_agents_indices = np.argwhere(binary_pragmatic_level_per_agent_matrix_per_run[r][i] == 1.0)
            for index in pragmatic_agents_indices:
                pragmatic_parents_selected_count += parent_index_counts[index]
            pragmatic_parents_selected_count_matrix[r][i] = pragmatic_parents_selected_count
    return pragmatic_parents_selected_count_matrix




def get_selected_hyps_ordered(selected_hyps_per_generation_matrix_per_run, hyp_order):
    selected_hyps_new_lex_order_all_runs = np.zeros_like(selected_hyps_per_generation_matrix_per_run)
    for r in range(n_runs):
        for i in range(n_iterations_mixed_pop):
            for a in range(pop_size_mixed_pop):
                this_agent_hyp = selected_hyps_per_generation_matrix_per_run[r][i][a]
                if this_agent_hyp >= len(hyp_order):
                    this_agent_hyp = this_agent_hyp-len(hyp_order)
                new_order_index = np.argwhere(hyp_order == this_agent_hyp)
                selected_hyps_new_lex_order_all_runs[r][i][a] = new_order_index
    return selected_hyps_new_lex_order_all_runs




def avg_pragmatic_advantage_over_gens(n_runs, n_copies, n_iterations, pragmatic_advantage_per_lex, selected_hyps_new_lex_order_all_runs):
    avg_pragmatic_advantage_over_gens_matrix = np.zeros(((n_runs * n_copies), n_iterations))
    for r in range((n_runs * n_copies)):
        for i in range(n_iterations):
            generation_hyps = selected_hyps_new_lex_order_all_runs[r][i]
            advantage_per_agent = pragmatic_advantage_per_lex[generation_hyps.astype(int)]
            avg_advantage = np.mean(advantage_per_agent)
            avg_pragmatic_advantage_over_gens_matrix[r][i] = avg_advantage
    print ''
    print ''
    print "avg_pragmatic_advantage_over_gens_matrix.shape is:"
    print avg_pragmatic_advantage_over_gens_matrix.shape
    initial_pragmatic_advantage_per_run = avg_pragmatic_advantage_over_gens_matrix[:,0]
    print "initial_pragmatic_advantage_per_run is:"
    print initial_pragmatic_advantage_per_run
    argmax_initial_pragmatic_advantage = np.argmax(initial_pragmatic_advantage_per_run)
    print "argmax_initial_pragmatic_advantage is:"
    print argmax_initial_pragmatic_advantage
    print "initial_pragmatic_advantage_per_run.shape is:"
    print initial_pragmatic_advantage_per_run.shape
    gen_1_pragmatic_advantage_per_run = avg_pragmatic_advantage_over_gens_matrix[:,1]
    print "gen_1_pragmatic_advantage_per_run is:"
    print gen_1_pragmatic_advantage_per_run
    argmax_gen_1_pragmatic_advantage = np.argmax(gen_1_pragmatic_advantage_per_run)
    print "argmax_gen_1_pragmatic_advantage is:"
    print argmax_gen_1_pragmatic_advantage
    print "gen_1_pragmatic_advantage_per_run.shape is:"
    print gen_1_pragmatic_advantage_per_run.shape
    return avg_pragmatic_advantage_over_gens_matrix




def avg_informativity_over_gens(n_runs, n_copies, n_iterations, inf_per_lex, selected_hyps_new_lex_order_all_runs):
    avg_inf_over_gens_matrix = np.zeros(((n_runs*n_copies), n_iterations))
    for r in range((n_runs*n_copies)):
        for i in range(n_iterations):
            generation_hyps = selected_hyps_new_lex_order_all_runs[r][i]
            inf_per_agent = inf_per_lex[generation_hyps.astype(int)]
            avg_inf = np.mean(inf_per_agent)
            avg_inf_over_gens_matrix[r][i] = avg_inf
    print ''
    print ''
    print "avg_inf_over_gens_matrix.shape is:"
    print avg_inf_over_gens_matrix.shape
    initial_inf_per_run = avg_inf_over_gens_matrix[:,0]
    print "initial_inf_per_run is:"
    print initial_inf_per_run
    argmin_initial_inf_per_run = np.argmin(initial_inf_per_run)
    print "argmin_initial_inf_per_run is:"
    print argmin_initial_inf_per_run
    argmax_initial_inf_per_run = np.argmax(initial_inf_per_run)
    print "argmax_initial_inf_per_run is:"
    print argmax_initial_inf_per_run
    print "initial_inf_per_run.shape is:"
    print initial_inf_per_run.shape
    gen_1_inf_per_run = avg_inf_over_gens_matrix[:,1]
    print "gen_1_inf_per_run is:"
    print gen_1_inf_per_run
    argmin_gen_1_inf_per_run = np.argmin(gen_1_inf_per_run)
    print "argmin_gen_1_inf_per_run is:"
    print argmin_gen_1_inf_per_run
    argmax_gen_1_inf_per_run = np.argmax(gen_1_inf_per_run)
    print "argmax_gen_1_inf_per_run is:"
    print argmax_gen_1_inf_per_run
    print "gen_1_inf_per_run.shape is:"
    print gen_1_inf_per_run.shape
    return avg_inf_over_gens_matrix


def calc_mean_and_conf_invs_over_gens(avg_inf_over_gens_matrix):
    mean_inf_over_gens = np.mean(avg_inf_over_gens_matrix, axis=0)
    std_inf_over_gens = np.std(avg_inf_over_gens_matrix, axis=0)
    conf_intervals_inf_over_gens = stats.norm.interval(0.95, loc=mean_inf_over_gens, scale=std_inf_over_gens / np.sqrt(n_runs*1))
    conf_intervals_inf_over_gens = np.array(conf_intervals_inf_over_gens)
    return mean_inf_over_gens, conf_intervals_inf_over_gens


def calc_percentiles_over_gens(avg_inf_over_gens_matrix):
    percentile_25_inf_over_gens = np.percentile(avg_inf_over_gens_matrix, 25, axis=0)
    median_optimal_inf_over_gens = np.percentile(avg_inf_over_gens_matrix, 50, axis=0)
    percentile_75_inf_over_gens = np.percentile(avg_inf_over_gens_matrix, 75, axis=0)
    percentiles_inf_over_gens = np.array([percentile_25_inf_over_gens, median_optimal_inf_over_gens, percentile_75_inf_over_gens])
    return percentiles_inf_over_gens



def plot_proportion_over_gens(plot_file_name, plot_title, xrange, xlabel, ylabel, proportion_pragmatic_level_over_gens_per_run, y_value_over_gens_per_run, style, legend, legend_loc=None, y_lim=None):
    sns.set_style("whitegrid")
    if y_lim:
        plt.ylim(y_lim)
        plt.yticks(np.arange(0.0, y_lim[1], 0.1))
    if style == 'unit_traces':
        sns.tsplot(y_value_over_gens_per_run, time=xrange, err_style='unit_traces')
    elif style == 'divided_by_genes':
        sns.set_palette("deep")
        color_literal = sns.color_palette()[0]
        color_pragmatic = sns.color_palette()[4]
        literal_runs = []
        pragmatic_runs = []
        for r in range(len(y_value_over_gens_per_run)):
            if proportion_pragmatic_level_over_gens_per_run[r][-1] == 1.0:
                pragmatic_runs.append(y_value_over_gens_per_run[r])
            else:
                literal_runs.append(y_value_over_gens_per_run[r])
        literal_runs = np.array(literal_runs)
        pragmatic_runs = np.array(pragmatic_runs)
        with sns.axes_style("whitegrid"):
            sns.tsplot(pragmatic_runs, time=xrange, err_style='unit_traces', color=color_pragmatic, condition='N pragmatic = '+str(len(pragmatic_runs)))
            sns.tsplot(literal_runs, time=xrange, err_style='unit_traces', color=color_literal, condition='N literal = '+str(len(literal_runs)))
        plt.legend(loc=legend_loc)
    elif style == 'indiv_runs':
        sns.set_palette('tab20', n_runs)
        palette = itertools.cycle(sns.color_palette()[:])
        with sns.axes_style("whitegrid"):
            for r in range(len(y_value_over_gens_per_run)):
                color = next(palette)
                fig = sns.tsplot(y_value_over_gens_per_run[r], time=xrange, condition='r '+str(r+1), color=color, legend=legend)
        if legend == True:
            box = fig.axes.get_position()
            fig.axes.set_position([box.x0, box.y0, box.width*0.75, box.height])
            fig.axes.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6, ncol=2, borderaxespad=0.8)
    plt.tick_params(labelsize=13)
    plt.suptitle(plot_title, fontsize=18)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.savefig(plot_file_name+'.pdf')
    plt.show()



def plot_proportions_two_conditions(plot_file_name, plot_title, xrange, xlabel, ylabel, proportion_pragmatic_level_over_gens_per_run_select_ca, proportion_pragmatic_level_over_gens_per_run_select_p_taking, y_value_over_gens_per_run_select_ca, y_value_over_gens_per_run_select_p_taking, style, legend, legend_loc=None, y_lim=None):
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9.6, 4.8))
    if y_lim:
        plt.ylim(y_lim)
        plt.yticks(np.arange(0.0, y_lim[1], 0.1))
    with sns.axes_style("whitegrid"):
        if style == 'unit_traces':
            ax1 = sns.tsplot(y_value_over_gens_per_run_select_ca, time=xrange, err_style='unit_traces')
            ax2 = sns.tsplot(y_value_over_gens_per_run_select_p_taking, time=xrange, err_style='unit_traces')
            ax1.set_title("Selection for communication", fontsize=15)
            ax1.set_ylabel(ylabel, fontsize=18)
            ax2.set_title("Selection on perspective-inference", fontsize=15)
        elif style == 'divided_by_genes':
            sns.set_palette("deep")
            color_literal = sns.color_palette()[0]
            color_pragmatic = sns.color_palette()[4]
            literal_runs_select_ca = []
            pragmatic_runs_select_ca = []
            for r in range(len(y_value_over_gens_per_run_select_ca)):
                if proportion_pragmatic_level_over_gens_per_run_select_ca[r][-1] == 1.0:
                    pragmatic_runs_select_ca.append(y_value_over_gens_per_run_select_ca[r])
                else:
                    literal_runs_select_ca.append(y_value_over_gens_per_run_select_ca[r])
            pragmatic_runs_select_ca = np.array(pragmatic_runs_select_ca)
            literal_runs_select_ca = np.array(literal_runs_select_ca)
            literal_runs_select_p_taking = []
            pragmatic_runs_select_p_taking = []
            for r in range(len(y_value_over_gens_per_run_select_p_taking)):
                if proportion_pragmatic_level_over_gens_per_run_select_p_taking[r][-1] == 1.0:
                    pragmatic_runs_select_p_taking.append(y_value_over_gens_per_run_select_p_taking[r])
                else:
                    literal_runs_select_p_taking.append(y_value_over_gens_per_run_select_p_taking[r])
            pragmatic_runs_select_p_taking = np.array(pragmatic_runs_select_p_taking)
            literal_runs_select_p_taking = np.array(literal_runs_select_p_taking)
            sns.tsplot(pragmatic_runs_select_ca, time=xrange, err_style='unit_traces', color=color_pragmatic, condition='N pragmatic = '+str(len(pragmatic_runs_select_ca)), ax=ax1)
            sns.tsplot(literal_runs_select_ca, time=xrange, err_style='unit_traces', color=color_literal, condition='N literal = '+str(len(literal_runs_select_ca)), ax=ax1)
            sns.tsplot(pragmatic_runs_select_p_taking, time=xrange, err_style='unit_traces', color=color_pragmatic, condition='N pragmatic = '+str(len(pragmatic_runs_select_p_taking)), ax=ax2)
            sns.tsplot(literal_runs_select_p_taking, time=xrange, err_style='unit_traces', color=color_literal, condition='N literal = '+str(len(literal_runs_select_p_taking)), ax=ax2)
            ax1.set_title("Selection for communication", fontsize=15)
            ax1.set_ylabel(ylabel, fontsize=18)
            ax2.set_title("Selection on perspective-inference", fontsize=15)
            ax2.set_ylabel('', fontsize=18)
            ax1.legend(loc=legend_loc)
            ax2.legend(loc=legend_loc)
        elif style == 'indiv_runs':
            sns.set_palette('tab20', n_runs)
            palette = itertools.cycle(sns.color_palette()[:])
            for r in range(len(y_value_over_gens_per_run_select_ca)):
                color = next(palette)
                sns.tsplot(y_value_over_gens_per_run_select_ca[r], time=xrange, condition='r ' + str(r + 1), color=color, legend=False, ax=ax1)
            ax1.set_title("Selection for communication", fontsize=15)
            ax1.set_ylabel(ylabel, fontsize=18)
            sns.set_palette('tab20', n_runs)
            palette = itertools.cycle(sns.color_palette()[:])
            for r in range(len(y_value_over_gens_per_run_select_p_taking)):
                color = next(palette)
                sns.tsplot(y_value_over_gens_per_run_select_p_taking[r], time=xrange, condition='r ' + str(r + 1), color=color, legend=legend, ax=ax2)
            ax2.set_title("Selection on perspective-inference", fontsize=15)
        if legend == True:
            fig.subplots_adjust(wspace=-0.06, bottom=0.13)
            # Shrink current axis by 70%
            box = ax1.get_position()
            ax1.set_position([box.x0, box.y0, box.width * 0.85, box.height])
            box = ax2.get_position()
            ax2.set_position([box.x0, box.y0, box.width * 0.85, box.height])
            # Put a legend to the right of the current axis
            legend = plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=True, fontsize=9, borderaxespad=0.8)
            legend.get_frame().set_linewidth(1.5)
    x_tick_locs = range(xrange[-1]-n_iterations_mixed_pop, xrange[-1]+1, n_iterations_mixed_pop/4)
    x_tick_labels = [str(x) for x in range(0, (n_iterations_mixed_pop+1), n_iterations_mixed_pop/4)]
    x_tick_locs = x_tick_locs[1:]
    x_tick_labels = x_tick_labels[1:]
    plt.xticks(x_tick_locs, x_tick_labels)
    ax1.tick_params(labelsize=13)
    ax2.tick_params(labelsize=13)
    fig.text(0.5, 0.04, xlabel, ha='center', fontsize=18)
    plt.suptitle(plot_title, fontsize=18)
    plt.savefig(plot_file_name+'.pdf')
    plt.show()



if __name__ == "__main__":





    print ''
    print ''
    print 'n_contexts:'
    print n_contexts


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


    informativity_level_counts = np.bincount(np.multiply(informativity_per_lexicon_sorted, 100).astype(int))
    print ''
    print ''
    print "informativity_level_counts are:"
    print informativity_level_counts


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


    pragmatic_advantage_per_lex, min_advantage, max_advantage = unpickle_ca_per_lex()


    inf_per_lex_full_hyp_space = informativity_per_lexicon_sorted
    pragmatic_advantage_per_lex_full_hyp_space = pragmatic_advantage_per_lex
    for p in range(len(perspective_hyps)-1):
        inf_per_lex_full_hyp_space = np.hstack((inf_per_lex_full_hyp_space, informativity_per_lexicon_sorted))
        pragmatic_advantage_per_lex_full_hyp_space = np.hstack((pragmatic_advantage_per_lex_full_hyp_space, pragmatic_advantage_per_lex))
    # print "inf_per_lex_full_hyp_space is:"
    # print inf_per_lex_full_hyp_space
    print "inf_per_lex_full_hyp_space.shape is:"
    print inf_per_lex_full_hyp_space.shape
    print ''
    print ''
    # print "pragmatic_advantage_per_lex_full_hyp_space is:"
    # print pragmatic_advantage_per_lex_full_hyp_space
    print "pragmatic_advantage_per_lex_full_hyp_space.shape is:"
    print pragmatic_advantage_per_lex_full_hyp_space.shape

    max_inf = np.amax(inf_per_lex_full_hyp_space)
    print ''
    print 'max_inf is:'
    print max_inf

    baseline_inf = np.mean(inf_per_lex_full_hyp_space)
    print ''
    print 'baseline_inf is:'
    print baseline_inf




    ###############################################################################
    selection_type = 'none'
    pragmatic_level_initial_pop = 'literal'


    pragmatic_level_per_agent_matrix_per_run_select_none, avg_fitness_per_gen_per_run_select_none, selected_hyps_per_generation_matrix_per_run_select_none, selected_parent_indices_matrix_per_run_select_none = unpickle_results_multi_runs(n_runs, selection_type)

    print ''
    print ''
    # print "pragmatic_level_per_agent_matrix_per_run_select_none is:"
    # print pragmatic_level_per_agent_matrix_per_run_select_none
    print "len(pragmatic_level_per_agent_matrix_per_run_select_none) is:"
    print len(pragmatic_level_per_agent_matrix_per_run_select_none)

    print ''
    print ''
    # print "avg_fitness_per_gen_per_run_select_none is:"
    # print avg_fitness_per_gen_per_run_select_none
    print "avg_fitness_per_gen_per_run_select_none.shape is:"
    print avg_fitness_per_gen_per_run_select_none.shape

    print ''
    print ''
    # print "selected_hyps_per_generation_matrix_per_run_select_none is:"
    # print selected_hyps_per_generation_matrix_per_run_select_none
    print "selected_hyps_per_generation_matrix_per_run_select_none.shape is:"
    print selected_hyps_per_generation_matrix_per_run_select_none.shape


    proportion_pragmatic_agents_over_gens_per_run_select_none = calc_proportion_pragmatic_level_per_run(pragmatic_level_per_agent_matrix_per_run_select_none, n_runs, n_iterations_mixed_pop, pop_size_mixed_pop, 'prag')
    print ''
    print ''
    print "proportion_pragmatic_agents_over_gens_per_run_select_none.shape is:"
    print proportion_pragmatic_agents_over_gens_per_run_select_none.shape
    print ''
    print "np.mean(proportion_pragmatic_agents_over_gens_per_run_select_none) is:"
    print np.mean(proportion_pragmatic_agents_over_gens_per_run_select_none)


    pragmatic_parents_selected_count_matrix_select_none = calc_no_pragmatic_parents_per_pragmatic_agent(pragmatic_level_per_agent_matrix_per_run_select_none, n_runs, n_iterations_mixed_pop, pop_size_mixed_pop, 'prag', selected_parent_indices_matrix_per_run_select_none)
    print ''
    print ''
    print ''
    print ''
    print "pragmatic_parents_selected_count_matrix_select_none is:"
    print pragmatic_parents_selected_count_matrix_select_none
    print "pragmatic_parents_selected_count_matrix_select_none.shape is:"
    print pragmatic_parents_selected_count_matrix_select_none.shape



    print ''
    print ''
    print "selected_hyps_per_generation_matrix_per_run_select_none.shape is:"
    print selected_hyps_per_generation_matrix_per_run_select_none.shape

    selected_hyps_new_lex_order_all_runs_select_none = get_selected_hyps_ordered(selected_hyps_per_generation_matrix_per_run_select_none, argsort_informativity_per_lexicon)
    print ''
    print ''
    # print "selected_hyps_new_lex_order_all_runs_select_none is:"
    # print selected_hyps_new_lex_order_all_runs_select_none
    print "selected_hyps_new_lex_order_all_runs_select_none.shape is:"
    print selected_hyps_new_lex_order_all_runs_select_none.shape

    avg_inf_over_gens_matrix_select_none = avg_informativity_over_gens(n_runs, 1, n_iterations_mixed_pop, inf_per_lex_full_hyp_space, selected_hyps_new_lex_order_all_runs_select_none)
    # print "avg_inf_over_gens_matrix_select_none is:"
    # print avg_inf_over_gens_matrix_select_none
    print "avg_inf_over_gens_matrix_select_none.shape is:"
    print avg_inf_over_gens_matrix_select_none.shape


    mean_inf_over_gens_select_none, conf_intervals_inf_over_gens_select_none = calc_mean_and_conf_invs_over_gens(avg_inf_over_gens_matrix_select_none)
    print ''
    print ''
    # print "mean_inf_over_gens_select_none is:"
    # print mean_inf_over_gens_select_none
    print "mean_inf_over_gens_select_none.shape is:"
    print mean_inf_over_gens_select_none.shape
    print ''
    # print "conf_intervals_inf_over_gens_select_none is:"
    # print conf_intervals_inf_over_gens_select_none
    print "conf_intervals_inf_over_gens_select_none.shape is:"
    print conf_intervals_inf_over_gens_select_none.shape


    percentiles_inf_over_gens_select_none = calc_percentiles_over_gens(avg_inf_over_gens_matrix_select_none)
    print ''
    print ''
    # print "percentiles_inf_over_gens_select_none is:"
    # print percentiles_inf_over_gens_select_none
    print "percentiles_inf_over_gens_select_none.shape is:"
    print percentiles_inf_over_gens_select_none.shape




    avg_pragmatic_advantage_over_gens_matrix_select_none = avg_pragmatic_advantage_over_gens(n_runs, 1, n_iterations_mixed_pop, pragmatic_advantage_per_lex_full_hyp_space, selected_hyps_new_lex_order_all_runs_select_none)
    # print "avg_pragmatic_advantage_over_gens_matrix_select_none is:"
    # print avg_pragmatic_advantage_over_gens_matrix_select_none
    print "avg_pragmatic_advantage_over_gens_matrix_select_none.shape is:"
    print avg_pragmatic_advantage_over_gens_matrix_select_none.shape


    mean_pragmatic_advantage_over_gens_select_none, conf_intervals_pragmatic_advantage_over_gens_select_none = calc_mean_and_conf_invs_over_gens(avg_pragmatic_advantage_over_gens_matrix_select_none)
    print ''
    print ''
    # print "mean_pragmatic_advantage_over_gens_select_none is:"
    # print mean_pragmatic_advantage_over_gens_select_none
    print "mean_pragmatic_advantage_over_gens_select_none.shape is:"
    print mean_pragmatic_advantage_over_gens_select_none.shape
    print ''
    # print "conf_intervals_pragmatic_advantage_over_gens_select_none is:"
    # print conf_intervals_pragmatic_advantage_over_gens_select_none
    print "conf_intervals_pragmatic_advantage_over_gens_select_none.shape is:"
    print conf_intervals_pragmatic_advantage_over_gens_select_none.shape


    percentiles_pragmatic_advantage_over_gens_select_none = calc_percentiles_over_gens(avg_pragmatic_advantage_over_gens_matrix_select_none)
    print ''
    print ''
    # print "percentiles_pragmatic_advantage_over_gens_select_none is:"
    # print percentiles_pragmatic_advantage_over_gens_select_none
    print "percentiles_pragmatic_advantage_over_gens_select_none.shape is:"
    print percentiles_pragmatic_advantage_over_gens_select_none.shape




    if context_generation == 'random':
        if selection_type == 'none' or selection_type == 'l_learning':
            plot_filename = 'evo_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size_mixed_pop)+'_select_'+selection_type+'_'+str(n_iterations_mixed_pop)+'_I_'+str(n_contexts)+'_C_init_pop_'+pragmatic_level_initial_pop+'_a_'+str(optimality_alpha_initial_pop)[0]+'_mutnts_'+pragmatic_level_mutants+'_a_'+str(optimality_alpha_mutants)[0]+'_s_hyp_'+pragmatic_level_parent_hyp+'_xtra_err_'+str(extra_error)[0:3]+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+str(n_runs)+'_R_dcpl_'+str(decoupling)
        elif selection_type == 'p_taking':
            plot_filename = 'evo_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size_mixed_pop)+'_select_'+selection_type+'_'+str(n_iterations_mixed_pop)+'_I_'+str(n_contexts)+'_C_init_pop_'+pragmatic_level_initial_pop+'_a_'+str(optimality_alpha_initial_pop)[0]+'_mutnts_'+pragmatic_level_mutants+'_a_'+str(optimality_alpha_mutants)[0]+'_s_hyp_'+pragmatic_level_parent_hyp+'_xtra_err_'+str(extra_error)[0:3]+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+str(n_runs)+'_R_dcpl_'+str(decoupling)
        elif selection_type == 'ca_with_parent':
            plot_filename = 'evo_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size_mixed_pop)+'_select_'+selection_type+'_init_'+communication_type_initial_pop+'_'+ca_measure_type_initial_pop+'_mutnts_'+communication_type_mutants+'_'+ca_measure_type_mutants+'_'+str(n_iterations_mixed_pop)+'_I_'+str(n_contexts)+'_C_init_pop_'+pragmatic_level_initial_pop+'_a_'+str(optimality_alpha_initial_pop)[0]+'_mutnts_'+pragmatic_level_mutants+'_a_'+str(optimality_alpha_mutants)[0]+'_s_hyp_'+pragmatic_level_parent_hyp+'_xtra_err_'+str(extra_error)[0:3]+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+str(n_runs)+'_R_dcpl_'+str(decoupling)

    elif context_generation == 'only_helpful' or context_generation == 'optimal':
        if selection_type == 'none' or selection_type == 'l_learning':
            plot_filename = 'evo_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size_mixed_pop)+'_select_'+selection_type+'_'+str(n_iterations_mixed_pop)+'_I_'+str(n_contexts)+'_C_init_pop_'+pragmatic_level_initial_pop+'_a_'+str(optimality_alpha_initial_pop)[0]+'_mutnts_'+pragmatic_level_mutants+'_a_'+str(optimality_alpha_mutants)[0]+'_s_hyp_'+pragmatic_level_parent_hyp+'_xtra_err_'+str(extra_error)[0:3]+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+str(n_runs)+'_R_dcpl_'+str(decoupling)
        elif selection_type == 'p_taking':
            plot_filename = 'evo_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size_mixed_pop)+'_select_'+selection_type+'_'+str(n_iterations_mixed_pop)+'_I_'+str(n_contexts)+'_C_init_pop_'+pragmatic_level_initial_pop+'_a_'+str(optimality_alpha_initial_pop)[0]+'_mutnts_'+pragmatic_level_mutants+'_a_'+str(optimality_alpha_mutants)[0]+'_s_hyp_'+pragmatic_level_parent_hyp+'_xtra_err_'+str(extra_error)[0:3]+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+str(n_runs)+'_R_dcpl_'+str(decoupling)
        elif selection_type == 'ca_with_parent':
            plot_filename = 'evo_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size_mixed_pop)+'_select_'+selection_type+'_init_'+communication_type_initial_pop+'_'+ca_measure_type_initial_pop+'_mutnts_'+communication_type_mutants+'_'+ca_measure_type_mutants+'_'+str(n_iterations_mixed_pop)+'_I_'+str(n_contexts)+'_C_init_pop_'+pragmatic_level_initial_pop+'_a_'+str(optimality_alpha_initial_pop)[0]+'_mutnts_'+pragmatic_level_mutants+'_a_'+str(optimality_alpha_mutants)[0]+'_s_hyp_'+pragmatic_level_parent_hyp+'_xtra_err_'+str(extra_error)[0:3]+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+str(n_runs)+'_R_dcpl_'+str(decoupling)



    plot_title = 'No selection'


    plot_title = 'Prop. prag. agents / gens w/ select '+selection_type

    plot_proportion_over_gens(plot_directory+'Plot_prop_prag_agents_'+plot_filename+'_'+line_style, plot_title, np.arange(1, len(proportion_pragmatic_agents_over_gens_per_run_select_none[0])+1), 'generations', 'proportion pragmatic agents', proportion_pragmatic_agents_over_gens_per_run_select_none, proportion_pragmatic_agents_over_gens_per_run_select_none, line_style, legend, legend_loc='center right', y_lim=[-0.05, 1.05])


    plot_title = 'Avg. informativeness / gens w/ select '+selection_type

    plot_proportion_over_gens(plot_directory+'Plot_avg_info_'+plot_filename+'_'+line_style, plot_title, np.arange(2, len(avg_fitness_per_gen_per_run_select_none[0][1:])+2), 'generations', 'average informativeness', proportion_pragmatic_agents_over_gens_per_run_select_none[:,1:], avg_inf_over_gens_matrix_select_none[:,1:], line_style, legend, legend_loc='lower right', y_lim=[0.0, max_inf+0.05])




    plot_title = 'Avg. prag. benefit / gens w/ select '+selection_type

    plot_proportion_over_gens(plot_directory+'Plot_avg_pragmatic_advantage_'+plot_filename+'_'+line_style, plot_title, np.arange(2, len(avg_pragmatic_advantage_over_gens_matrix_select_none[0][1:])+2), 'generations', 'average pragmatic benefit in ca', proportion_pragmatic_agents_over_gens_per_run_select_none[:,1:], avg_pragmatic_advantage_over_gens_matrix_select_none[:,1:], line_style, legend, legend_loc='upper right', y_lim=[0.0, 0.425]) #y_lim=[0.0, max_advantage+0.025])



    ###############################################################################
    selection_type = 'ca_with_parent'
    pragmatic_level_initial_pop = 'perspective-taking'


    pragmatic_level_per_agent_matrix_per_run_select_ca, avg_fitness_per_gen_per_run_select_ca, selected_hyps_per_generation_matrix_per_run_select_ca, selected_parent_indices_matrix_per_run_select_ca = unpickle_results_multi_runs(n_runs, selection_type)

    print ''
    print ''
    # print "pragmatic_level_per_agent_matrix_per_run_select_ca is:"
    # print pragmatic_level_per_agent_matrix_per_run_select_ca
    print "len(pragmatic_level_per_agent_matrix_per_run_select_ca) is:"
    print len(pragmatic_level_per_agent_matrix_per_run_select_ca)

    print ''
    print ''
    # print "avg_fitness_per_gen_per_run_select_ca is:"
    # print avg_fitness_per_gen_per_run_select_ca
    print "avg_fitness_per_gen_per_run_select_ca.shape is:"
    print avg_fitness_per_gen_per_run_select_ca.shape

    print ''
    print ''
    # print "selected_hyps_per_generation_matrix_per_run_select_ca is:"
    # print selected_hyps_per_generation_matrix_per_run_select_ca
    print "selected_hyps_per_generation_matrix_per_run_select_ca.shape is:"
    print selected_hyps_per_generation_matrix_per_run_select_ca.shape


    proportion_pragmatic_agents_over_gens_per_run_select_ca = calc_proportion_pragmatic_level_per_run(pragmatic_level_per_agent_matrix_per_run_select_ca, n_runs, n_iterations_mixed_pop, pop_size_mixed_pop, 'prag')
    print ''
    print ''
    print "proportion_pragmatic_agents_over_gens_per_run_select_ca.shape is:"
    print proportion_pragmatic_agents_over_gens_per_run_select_ca.shape
    print ''
    print "np.mean(proportion_pragmatic_agents_over_gens_per_run_select_ca) is:"
    print np.mean(proportion_pragmatic_agents_over_gens_per_run_select_ca)




    pragmatic_parents_selected_count_matrix_select_ca = calc_no_pragmatic_parents_per_pragmatic_agent(pragmatic_level_per_agent_matrix_per_run_select_ca, n_runs, n_iterations_mixed_pop, pop_size_mixed_pop, 'prag', selected_parent_indices_matrix_per_run_select_ca)
    print ''
    print ''
    print ''
    print ''
    print "pragmatic_parents_selected_count_matrix_select_ca is:"
    print pragmatic_parents_selected_count_matrix_select_ca
    print "pragmatic_parents_selected_count_matrix_select_ca.shape is:"
    print pragmatic_parents_selected_count_matrix_select_ca.shape




    print ''
    print ''
    print "selected_hyps_per_generation_matrix_per_run_select_ca.shape is:"
    print selected_hyps_per_generation_matrix_per_run_select_ca.shape

    selected_hyps_new_lex_order_all_runs_select_ca = get_selected_hyps_ordered(selected_hyps_per_generation_matrix_per_run_select_ca, argsort_informativity_per_lexicon)
    print ''
    print ''
    # print "selected_hyps_new_lex_order_all_runs_select_ca is:"
    # print selected_hyps_new_lex_order_all_runs_select_ca
    print "selected_hyps_new_lex_order_all_runs_select_ca.shape is:"
    print selected_hyps_new_lex_order_all_runs_select_ca.shape

    avg_inf_over_gens_matrix_select_ca = avg_informativity_over_gens(n_runs, 1, n_iterations_mixed_pop, inf_per_lex_full_hyp_space, selected_hyps_new_lex_order_all_runs_select_ca)
    # print "avg_inf_over_gens_matrix_select_ca is:"
    # print avg_inf_over_gens_matrix_select_ca
    print "avg_inf_over_gens_matrix_select_ca.shape is:"
    print avg_inf_over_gens_matrix_select_ca.shape


    mean_inf_over_gens_select_ca, conf_intervals_inf_over_gens_select_ca = calc_mean_and_conf_invs_over_gens(avg_inf_over_gens_matrix_select_ca)
    print ''
    print ''
    # print "mean_inf_over_gens_select_ca is:"
    # print mean_inf_over_gens_select_ca
    print "mean_inf_over_gens_select_ca.shape is:"
    print mean_inf_over_gens_select_ca.shape
    print ''
    # print "conf_intervals_inf_over_gens_select_ca is:"
    # print conf_intervals_inf_over_gens_select_ca
    print "conf_intervals_inf_over_gens_select_ca.shape is:"
    print conf_intervals_inf_over_gens_select_ca.shape


    percentiles_inf_over_gens_select_ca = calc_percentiles_over_gens(avg_inf_over_gens_matrix_select_ca)
    print ''
    print ''
    # print "percentiles_inf_over_gens_select_ca is:"
    # print percentiles_inf_over_gens_select_ca
    print "percentiles_inf_over_gens_select_ca.shape is:"
    print percentiles_inf_over_gens_select_ca.shape




    avg_pragmatic_advantage_over_gens_matrix_select_ca = avg_pragmatic_advantage_over_gens(n_runs, 1, n_iterations_mixed_pop, pragmatic_advantage_per_lex_full_hyp_space, selected_hyps_new_lex_order_all_runs_select_ca)
    # print "avg_pragmatic_advantage_over_gens_matrix_select_ca is:"
    # print avg_pragmatic_advantage_over_gens_matrix_select_ca
    print "avg_pragmatic_advantage_over_gens_matrix_select_ca.shape is:"
    print avg_pragmatic_advantage_over_gens_matrix_select_ca.shape


    mean_pragmatic_advantage_over_gens_select_ca, conf_intervals_pragmatic_advantage_over_gens_select_ca = calc_mean_and_conf_invs_over_gens(avg_pragmatic_advantage_over_gens_matrix_select_ca)
    print ''
    print ''
    # print "mean_pragmatic_advantage_over_gens_select_ca is:"
    # print mean_pragmatic_advantage_over_gens_select_ca
    print "mean_pragmatic_advantage_over_gens_select_ca.shape is:"
    print mean_pragmatic_advantage_over_gens_select_ca.shape
    print ''
    # print "conf_intervals_pragmatic_advantage_over_gens_select_ca is:"
    # print conf_intervals_pragmatic_advantage_over_gens_select_ca
    print "conf_intervals_pragmatic_advantage_over_gens_select_ca.shape is:"
    print conf_intervals_pragmatic_advantage_over_gens_select_ca.shape


    percentiles_pragmatic_advantage_over_gens_select_ca = calc_percentiles_over_gens(avg_pragmatic_advantage_over_gens_matrix_select_ca)
    print ''
    print ''
    # print "percentiles_pragmatic_advantage_over_gens_select_ca is:"
    # print percentiles_pragmatic_advantage_over_gens_select_ca
    print "percentiles_pragmatic_advantage_over_gens_select_ca.shape is:"
    print percentiles_pragmatic_advantage_over_gens_select_ca.shape



    if context_generation == 'random':
        if selection_type == 'none' or selection_type == 'l_learning':
            plot_filename = 'evo_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size_mixed_pop)+'_select_'+selection_type+'_'+str(n_iterations_mixed_pop)+'_I_'+str(n_contexts)+'_C_init_pop_'+pragmatic_level_initial_pop+'_a_'+str(optimality_alpha_initial_pop)[0]+'_mutnts_'+pragmatic_level_mutants+'_a_'+str(optimality_alpha_mutants)[0]+'_s_hyp_'+pragmatic_level_parent_hyp+'_xtra_err_'+str(extra_error)[0:3]+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+str(n_runs)+'_R_dcpl_'+str(decoupling)
        elif selection_type == 'p_taking':
            plot_filename = 'evo_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size_mixed_pop)+'_select_'+selection_type+'_'+str(n_iterations_mixed_pop)+'_I_'+str(n_contexts)+'_C_init_pop_'+pragmatic_level_initial_pop+'_a_'+str(optimality_alpha_initial_pop)[0]+'_mutnts_'+pragmatic_level_mutants+'_a_'+str(optimality_alpha_mutants)[0]+'_s_hyp_'+pragmatic_level_parent_hyp+'_xtra_err_'+str(extra_error)[0:3]+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+str(n_runs)+'_R_dcpl_'+str(decoupling)
        elif selection_type == 'ca_with_parent':
            plot_filename = 'evo_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size_mixed_pop)+'_select_'+selection_type+'_init_'+communication_type_initial_pop+'_'+ca_measure_type_initial_pop+'_mutnts_'+communication_type_mutants+'_'+ca_measure_type_mutants+'_'+str(n_iterations_mixed_pop)+'_I_'+str(n_contexts)+'_C_init_pop_'+pragmatic_level_initial_pop+'_a_'+str(optimality_alpha_initial_pop)[0]+'_mutnts_'+pragmatic_level_mutants+'_a_'+str(optimality_alpha_mutants)[0]+'_s_hyp_'+pragmatic_level_parent_hyp+'_xtra_err_'+str(extra_error)[0:3]+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+str(n_runs)+'_R_dcpl_'+str(decoupling)

    elif context_generation == 'only_helpful' or context_generation == 'optimal':
        if selection_type == 'none' or selection_type == 'l_learning':
            plot_filename = 'evo_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size_mixed_pop)+'_select_'+selection_type+'_'+str(n_iterations_mixed_pop)+'_I_'+str(n_contexts)+'_C_init_pop_'+pragmatic_level_initial_pop+'_a_'+str(optimality_alpha_initial_pop)[0]+'_mutnts_'+pragmatic_level_mutants+'_a_'+str(optimality_alpha_mutants)[0]+'_s_hyp_'+pragmatic_level_parent_hyp+'_xtra_err_'+str(extra_error)[0:3]+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+str(n_runs)+'_R_dcpl_'+str(decoupling)
        elif selection_type == 'p_taking':
            plot_filename = 'evo_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size_mixed_pop)+'_select_'+selection_type+'_'+str(n_iterations_mixed_pop)+'_I_'+str(n_contexts)+'_C_init_pop_'+pragmatic_level_initial_pop+'_a_'+str(optimality_alpha_initial_pop)[0]+'_mutnts_'+pragmatic_level_mutants+'_a_'+str(optimality_alpha_mutants)[0]+'_s_hyp_'+pragmatic_level_parent_hyp+'_xtra_err_'+str(extra_error)[0:3]+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+str(n_runs)+'_R_dcpl_'+str(decoupling)
        elif selection_type == 'ca_with_parent':
            plot_filename = 'evo_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size_mixed_pop)+'_select_'+selection_type+'_init_'+communication_type_initial_pop+'_'+ca_measure_type_initial_pop+'_mutnts_'+communication_type_mutants+'_'+ca_measure_type_mutants+'_'+str(n_iterations_mixed_pop)+'_I_'+str(n_contexts)+'_C_init_pop_'+pragmatic_level_initial_pop+'_a_'+str(optimality_alpha_initial_pop)[0]+'_mutnts_'+pragmatic_level_mutants+'_a_'+str(optimality_alpha_mutants)[0]+'_s_hyp_'+pragmatic_level_parent_hyp+'_xtra_err_'+str(extra_error)[0:3]+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+str(n_runs)+'_R_dcpl_'+str(decoupling)



    plot_title = 'Selection for communication'


    plot_title = 'Prop. prag. agents / gens w/ select '+selection_type

    plot_proportion_over_gens(plot_directory+'Plot_prop_prag_agents_'+plot_filename+'_'+line_style, plot_title, np.arange(1, len(proportion_pragmatic_agents_over_gens_per_run_select_ca[0])+1), 'generations', 'proportion pragmatic agents', proportion_pragmatic_agents_over_gens_per_run_select_ca, proportion_pragmatic_agents_over_gens_per_run_select_ca, line_style, legend, legend_loc='center right', y_lim=[-0.05, 1.05])


    plot_title = 'Avg. fitness / gens w/ select '+selection_type

    plot_proportion_over_gens(plot_directory+'Plot_avg_fitness_'+plot_filename+'_'+line_style, plot_title, np.arange(2, len(avg_fitness_per_gen_per_run_select_ca[0][1:])+2), 'generations', 'average fitness', proportion_pragmatic_agents_over_gens_per_run_select_ca[:,1:], avg_fitness_per_gen_per_run_select_ca[:,1:], line_style, legend, legend_loc='lower right', y_lim=[0.0, 1.05])

    plot_title = 'Avg. informativeness / gens w/ select '+selection_type

    plot_proportion_over_gens(plot_directory+'Plot_avg_info_'+plot_filename+'_'+line_style, plot_title, np.arange(2, len(avg_fitness_per_gen_per_run_select_ca[0][1:])+2), 'generations', 'average informativeness', proportion_pragmatic_agents_over_gens_per_run_select_ca[:,1:], avg_inf_over_gens_matrix_select_ca[:,1:], line_style, legend, legend_loc='lower right', y_lim=[0.0, max_inf+0.05])


    plot_title = 'Avg. prag. benefit / gens w/ select '+selection_type

    plot_proportion_over_gens(plot_directory+'Plot_avg_pragmatic_advantage_'+plot_filename+'_'+line_style, plot_title, np.arange(2, len(avg_pragmatic_advantage_over_gens_matrix_select_ca[0][1:])+2), 'generations', 'average pragmatic benefit in ca', proportion_pragmatic_agents_over_gens_per_run_select_ca[:,1:], avg_pragmatic_advantage_over_gens_matrix_select_ca[:,1:], line_style, legend, legend_loc='upper right', y_lim=[0.0, 0.425]) #y_lim=[0.0, max_advantage+0.025])


    ##############################################################################
    selection_type = 'p_taking'
    pragmatic_level_initial_pop = 'literal'


    pragmatic_level_per_agent_matrix_per_run_select_p_taking, avg_fitness_per_gen_per_run_select_p_taking, selected_hyps_per_generation_matrix_per_run_select_p_taking, selected_parent_indices_matrix_per_run_select_p_taking = unpickle_results_multi_runs(n_runs, selection_type)

    print ''
    print ''
    # print "pragmatic_level_per_agent_matrix_per_run_select_p_taking is:"
    # print pragmatic_level_per_agent_matrix_per_run_select_p_taking
    print "len(pragmatic_level_per_agent_matrix_per_run_select_p_taking) is:"
    print len(pragmatic_level_per_agent_matrix_per_run_select_p_taking)

    print ''
    print ''
    # print "avg_fitness_per_gen_per_run_select_p_taking is:"
    # print avg_fitness_per_gen_per_run_select_p_taking
    print "avg_fitness_per_gen_per_run_select_p_taking.shape is:"
    print avg_fitness_per_gen_per_run_select_p_taking.shape

    print ''
    print ''
    # print "selected_hyps_per_generation_matrix_per_run_select_p_taking is:"
    # print selected_hyps_per_generation_matrix_per_run_select_p_taking
    print "selected_hyps_per_generation_matrix_per_run_select_p_taking.shape is:"
    print selected_hyps_per_generation_matrix_per_run_select_p_taking.shape


    proportion_pragmatic_agents_over_gens_per_run_select_p_taking = calc_proportion_pragmatic_level_per_run(pragmatic_level_per_agent_matrix_per_run_select_p_taking, n_runs, n_iterations_mixed_pop, pop_size_mixed_pop, 'prag')
    print ''
    print ''
    print "proportion_pragmatic_agents_over_gens_per_run_select_p_taking.shape is:"
    print proportion_pragmatic_agents_over_gens_per_run_select_p_taking.shape
    print ''
    print "np.mean(proportion_pragmatic_agents_over_gens_per_run_select_p_taking) is:"
    print np.mean(proportion_pragmatic_agents_over_gens_per_run_select_p_taking)




    pragmatic_parents_selected_count_matrix_select_p_taking = calc_no_pragmatic_parents_per_pragmatic_agent(pragmatic_level_per_agent_matrix_per_run_select_p_taking, n_runs, n_iterations_mixed_pop, pop_size_mixed_pop, 'prag', selected_parent_indices_matrix_per_run_select_p_taking)
    print ''
    print ''
    print ''
    print ''
    print "pragmatic_parents_selected_count_matrix_select_p_taking is:"
    print pragmatic_parents_selected_count_matrix_select_p_taking
    print "pragmatic_parents_selected_count_matrix_select_p_taking.shape is:"
    print pragmatic_parents_selected_count_matrix_select_p_taking.shape




    print ''
    print ''
    print "selected_hyps_per_generation_matrix_per_run_select_p_taking.shape is:"
    print selected_hyps_per_generation_matrix_per_run_select_p_taking.shape

    selected_hyps_new_lex_order_all_runs_select_p_taking = get_selected_hyps_ordered(selected_hyps_per_generation_matrix_per_run_select_p_taking, argsort_informativity_per_lexicon)
    print ''
    print ''
    # print "selected_hyps_new_lex_order_all_runs_select_p_taking is:"
    # print selected_hyps_new_lex_order_all_runs_select_p_taking
    print "selected_hyps_new_lex_order_all_runs_select_p_taking.shape is:"
    print selected_hyps_new_lex_order_all_runs_select_p_taking.shape

    avg_inf_over_gens_matrix_select_p_taking = avg_informativity_over_gens(n_runs, 1, n_iterations_mixed_pop, inf_per_lex_full_hyp_space, selected_hyps_new_lex_order_all_runs_select_p_taking)
    # print "avg_inf_over_gens_matrix_select_p_taking is:"
    # print avg_inf_over_gens_matrix_select_p_taking
    print "avg_inf_over_gens_matrix_select_p_taking.shape is:"
    print avg_inf_over_gens_matrix_select_p_taking.shape


    mean_inf_over_gens_select_p_taking, conf_intervals_inf_over_gens_select_p_taking = calc_mean_and_conf_invs_over_gens(avg_inf_over_gens_matrix_select_p_taking)
    print ''
    print ''
    # print "mean_inf_over_gens_select_p_taking is:"
    # print mean_inf_over_gens_select_p_taking
    print "mean_inf_over_gens_select_p_taking.shape is:"
    print mean_inf_over_gens_select_p_taking.shape
    print ''
    # print "conf_intervals_inf_over_gens_select_p_taking is:"
    # print conf_intervals_inf_over_gens_select_p_taking
    print "conf_intervals_inf_over_gens_select_p_taking.shape is:"
    print conf_intervals_inf_over_gens_select_p_taking.shape


    percentiles_inf_over_gens_select_p_taking = calc_percentiles_over_gens(avg_inf_over_gens_matrix_select_p_taking)
    print ''
    print ''
    # print "percentiles_inf_over_gens_select_p_taking is:"
    # print percentiles_inf_over_gens_select_p_taking
    print "percentiles_inf_over_gens_select_p_taking.shape is:"
    print percentiles_inf_over_gens_select_p_taking.shape




    avg_pragmatic_advantage_over_gens_matrix_select_p_taking = avg_pragmatic_advantage_over_gens(n_runs, 1, n_iterations_mixed_pop, pragmatic_advantage_per_lex_full_hyp_space, selected_hyps_new_lex_order_all_runs_select_p_taking)
    # print "avg_pragmatic_advantage_over_gens_matrix_select_p_taking is:"
    # print avg_pragmatic_advantage_over_gens_matrix_select_p_taking
    print "avg_pragmatic_advantage_over_gens_matrix_select_p_taking.shape is:"
    print avg_pragmatic_advantage_over_gens_matrix_select_p_taking.shape


    mean_pragmatic_advantage_over_gens_select_p_taking, conf_intervals_pragmatic_advantage_over_gens_select_p_taking = calc_mean_and_conf_invs_over_gens(avg_pragmatic_advantage_over_gens_matrix_select_p_taking)
    print ''
    print ''
    # print "mean_pragmatic_advantage_over_gens_select_p_taking is:"
    # print mean_pragmatic_advantage_over_gens_select_p_taking
    print "mean_pragmatic_advantage_over_gens_select_p_taking.shape is:"
    print mean_pragmatic_advantage_over_gens_select_p_taking.shape
    print ''
    # print "conf_intervals_pragmatic_advantage_over_gens_select_p_taking is:"
    # print conf_intervals_pragmatic_advantage_over_gens_select_p_taking
    print "conf_intervals_pragmatic_advantage_over_gens_select_p_taking.shape is:"
    print conf_intervals_pragmatic_advantage_over_gens_select_p_taking.shape


    percentiles_pragmatic_advantage_over_gens_select_p_taking = calc_percentiles_over_gens(avg_pragmatic_advantage_over_gens_matrix_select_p_taking)
    print ''
    print ''
    # print "percentiles_pragmatic_advantage_over_gens_select_p_taking is:"
    # print percentiles_pragmatic_advantage_over_gens_select_p_taking
    print "percentiles_pragmatic_advantage_over_gens_select_p_taking.shape is:"
    print percentiles_pragmatic_advantage_over_gens_select_p_taking.shape





    if context_generation == 'random':
        if selection_type == 'none' or selection_type == 'l_learning':
            plot_filename = 'evo_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size_mixed_pop)+'_select_'+selection_type+'_'+str(n_iterations_mixed_pop)+'_I_'+str(n_contexts)+'_C_init_pop_'+pragmatic_level_initial_pop+'_a_'+str(optimality_alpha_initial_pop)[0]+'_mutnts_'+pragmatic_level_mutants+'_a_'+str(optimality_alpha_mutants)[0]+'_s_hyp_'+pragmatic_level_parent_hyp+'_xtra_err_'+str(extra_error)[0:3]+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+str(n_runs)+'_R_dcpl_'+str(decoupling)
        elif selection_type == 'p_taking':
            plot_filename = 'evo_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size_mixed_pop)+'_select_'+selection_type+'_'+str(n_iterations_mixed_pop)+'_I_'+str(n_contexts)+'_C_init_pop_'+pragmatic_level_initial_pop+'_a_'+str(optimality_alpha_initial_pop)[0]+'_mutnts_'+pragmatic_level_mutants+'_a_'+str(optimality_alpha_mutants)[0]+'_s_hyp_'+pragmatic_level_parent_hyp+'_xtra_err_'+str(extra_error)[0:3]+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+str(n_runs)+'_R_dcpl_'+str(decoupling)
        elif selection_type == 'ca_with_parent':
            plot_filename = 'evo_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size_mixed_pop)+'_select_'+selection_type+'_init_'+communication_type_initial_pop+'_'+ca_measure_type_initial_pop+'_mutnts_'+communication_type_mutants+'_'+ca_measure_type_mutants+'_'+str(n_iterations_mixed_pop)+'_I_'+str(n_contexts)+'_C_init_pop_'+pragmatic_level_initial_pop+'_a_'+str(optimality_alpha_initial_pop)[0]+'_mutnts_'+pragmatic_level_mutants+'_a_'+str(optimality_alpha_mutants)[0]+'_s_hyp_'+pragmatic_level_parent_hyp+'_xtra_err_'+str(extra_error)[0:3]+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+str(n_runs)+'_R_dcpl_'+str(decoupling)

    elif context_generation == 'only_helpful' or context_generation == 'optimal':
        if selection_type == 'none' or selection_type == 'l_learning':
            plot_filename = 'evo_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size_mixed_pop)+'_select_'+selection_type+'_'+str(n_iterations_mixed_pop)+'_I_'+str(n_contexts)+'_C_init_pop_'+pragmatic_level_initial_pop+'_a_'+str(optimality_alpha_initial_pop)[0]+'_mutnts_'+pragmatic_level_mutants+'_a_'+str(optimality_alpha_mutants)[0]+'_s_hyp_'+pragmatic_level_parent_hyp+'_xtra_err_'+str(extra_error)[0:3]+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+str(n_runs)+'_R_dcpl_'+str(decoupling)
        elif selection_type == 'p_taking':
            plot_filename = 'evo_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size_mixed_pop)+'_select_'+selection_type+'_'+str(n_iterations_mixed_pop)+'_I_'+str(n_contexts)+'_C_init_pop_'+pragmatic_level_initial_pop+'_a_'+str(optimality_alpha_initial_pop)[0]+'_mutnts_'+pragmatic_level_mutants+'_a_'+str(optimality_alpha_mutants)[0]+'_s_hyp_'+pragmatic_level_parent_hyp+'_xtra_err_'+str(extra_error)[0:3]+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+str(n_runs)+'_R_dcpl_'+str(decoupling)
        elif selection_type == 'ca_with_parent':
            plot_filename = 'evo_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size_mixed_pop)+'_select_'+selection_type+'_init_'+communication_type_initial_pop+'_'+ca_measure_type_initial_pop+'_mutnts_'+communication_type_mutants+'_'+ca_measure_type_mutants+'_'+str(n_iterations_mixed_pop)+'_I_'+str(n_contexts)+'_C_init_pop_'+pragmatic_level_initial_pop+'_a_'+str(optimality_alpha_initial_pop)[0]+'_mutnts_'+pragmatic_level_mutants+'_a_'+str(optimality_alpha_mutants)[0]+'_s_hyp_'+pragmatic_level_parent_hyp+'_xtra_err_'+str(extra_error)[0:3]+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+str(n_runs)+'_R_dcpl_'+str(decoupling)



    plot_title = 'Selection on perspective-inference'


    plot_title = 'Prop. prag. agents / gens w/ select '+selection_type

    plot_proportion_over_gens(plot_directory+'Plot_prop_prag_agents_'+plot_filename+'_'+line_style, plot_title, np.arange(1, len(proportion_pragmatic_agents_over_gens_per_run_select_p_taking[0])+1), 'generations', 'proportion pragmatic agents', proportion_pragmatic_agents_over_gens_per_run_select_p_taking,proportion_pragmatic_agents_over_gens_per_run_select_p_taking, line_style, legend, legend_loc='center right', y_lim=[-0.05, 1.05])


    plot_title = 'Avg. fitness / gens w/ select '+selection_type

    plot_proportion_over_gens(plot_directory+'Plot_avg_fitness_'+plot_filename+'_'+line_style, plot_title, np.arange(2, len(avg_fitness_per_gen_per_run_select_p_taking[0][1:])+2), 'generations', 'average fitness', proportion_pragmatic_agents_over_gens_per_run_select_p_taking[:,1:],avg_fitness_per_gen_per_run_select_p_taking[:,1:], line_style, legend, legend_loc='lower right', y_lim=[0.0, 1.05])

    plot_title = 'Avg. informativeness / gens w/ select '+selection_type

    plot_proportion_over_gens(plot_directory+'Plot_avg_info_'+plot_filename+'_'+line_style, plot_title, np.arange(2, len(avg_inf_over_gens_matrix_select_p_taking[0][1:])+2), 'generations', 'average informativeness', proportion_pragmatic_agents_over_gens_per_run_select_p_taking[:,1:], avg_inf_over_gens_matrix_select_p_taking[:,1:], line_style, legend, legend_loc='lower right', y_lim=[0.0, max_inf+0.05])


    plot_title = 'Avg. prag. benefit / gens w/ select '+selection_type

    plot_proportion_over_gens(plot_directory+'Plot_avg_pragmatic_advantage_'+plot_filename+'_'+line_style, plot_title, np.arange(2, len(avg_pragmatic_advantage_over_gens_matrix_select_p_taking[0][1:])+2), 'generations', 'average pragmatic benefit in ca', proportion_pragmatic_agents_over_gens_per_run_select_p_taking[:,1:], avg_pragmatic_advantage_over_gens_matrix_select_p_taking[:,1:], line_style, legend, legend_loc='upper right', y_lim=[0.0, 0.425]) #y_lim=[0.0, max_advantage+0.025])





    plot_title = 'Proportion pragmatic agents after pragmatic mutant is inserted'


    plot_proportions_two_conditions(plot_directory+'Plot_prop_prag_agents_Two_Conds_'+plot_filename+'_'+line_style, plot_title, np.arange(1, len(proportion_pragmatic_agents_over_gens_per_run_select_ca[0])+1), 'generations', 'proportion pragmatic agents', proportion_pragmatic_agents_over_gens_per_run_select_ca, proportion_pragmatic_agents_over_gens_per_run_select_p_taking, proportion_pragmatic_agents_over_gens_per_run_select_ca, proportion_pragmatic_agents_over_gens_per_run_select_p_taking, line_style, legend, legend_loc='center right', y_lim=[-0.05, 1.05])


    plot_title = 'Fitness after pragmatic mutant is inserted'

    plot_proportions_two_conditions(plot_directory+'Plot_fitness_Two_Conds_'+plot_filename+'_'+line_style, plot_title, np.arange(1, len(avg_inf_over_gens_matrix_select_ca[0][1:])+1), 'generations', 'average fitness', proportion_pragmatic_agents_over_gens_per_run_select_ca[:,1:], proportion_pragmatic_agents_over_gens_per_run_select_p_taking[:,1:], avg_fitness_per_gen_per_run_select_ca[:,1:], avg_fitness_per_gen_per_run_select_p_taking[:,1:], line_style, legend, legend_loc='lower right', y_lim=[0.0, 1.05])


    plot_title = 'Informativeness after pragmatic mutant is inserted'

    plot_proportions_two_conditions(plot_directory+'Plot_inf_Two_Conds_'+plot_filename+'_'+line_style, plot_title, np.arange(1, len(avg_inf_over_gens_matrix_select_ca[0][1:])+1), 'generations', 'average informativeness', proportion_pragmatic_agents_over_gens_per_run_select_ca[:,1:], proportion_pragmatic_agents_over_gens_per_run_select_p_taking[:,1:], avg_inf_over_gens_matrix_select_ca[:,1:], avg_inf_over_gens_matrix_select_p_taking[:,1:], line_style, legend, legend_loc='lower right', y_lim=[0.0, max_inf+0.05])


    plot_title = 'Pragmatic benefit after pragmatic mutant is inserted'

    plot_proportions_two_conditions(plot_directory+'Plot_prag_benefit_Two_Conds_'+plot_filename+'_'+line_style, plot_title, np.arange(1, len(avg_pragmatic_advantage_over_gens_matrix_select_ca[0][1:])+1), 'generations', 'average pragmatic benefit in ca', proportion_pragmatic_agents_over_gens_per_run_select_ca[:,1:], proportion_pragmatic_agents_over_gens_per_run_select_p_taking[:,1:], avg_pragmatic_advantage_over_gens_matrix_select_ca[:,1:], avg_pragmatic_advantage_over_gens_matrix_select_p_taking[:,1:], line_style, legend, legend_loc='upper right', y_lim=[0.0, 0.425]) #y_lim=[0.0, max_advantage+0.025])




