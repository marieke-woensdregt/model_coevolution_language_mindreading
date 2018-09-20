__author__ = 'Marieke Woensdregt'


import numpy as np
import pickle
import time

import context
import hypspace
import lex
import measur
import plots
import pop
import prior
import saveresults



# np.set_printoptions(threshold='nan')


#######################################################################################################################
# 1: THE PARAMETERS:


##!!!!!! MAKE SURE TO CHANGE THE PATHS BELOW TO MATCH THE FILE SYSTEM OF YOUR MACHINE:
pickle_file_directory = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/'

plot_file_directory = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Plots/'

run_type_dir = 'Learner_Pop/'



# 1.1: The parameters defining the lexicon size (and thus the number of meanings in the world):

n_meanings = 3  # The number of meanings
n_signals = 3  # The number of signals


# 1.2: The parameters defining the contexts and how they map to the agent's saliencies:

context_generation = 'random'  # This can be set to either 'random', 'only_helpful', 'optimal'
#helpful_contexts = np.array([[0.1, 0.7], [0.3, 0.9], [0.1, 0.6], [0.4, 0.9], [0.1, 0.8], [0.2, 0.9], [0.1, 0.5], [0.5, 0.9], [0.1, 0.4], [0.6, 0.9], [0.7, 0.1], [0.9, 0.3], [0.6, 0.1], [0.9, 0.4], [0.8, 0.1], [0.9, 0.2], [0.5, 0.1], [0.9, 0.5], [0.4, 0.1], [0.9, 0.6]])  # This is a fixed collection of the 20 most helpful contexts (in which the ratio of meaning probability for the one perspective is maximally different from that for the other perspective).
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

sal_alpha = 1.  # The exponent that is used by the saliency function. sal_alpha=1 gives a directly proportional relationship between meaning distances and meaning saliencies, with sal_alpha>1 the saliency of meanings goes down exponentially as their distance from the agent increases (and with sal_alpha<1 the other way around)


error = 0.05  # The error term on production
error_string = saveresults.convert_array_to_string(error)
extra_error = True  # Determines whether the error specified above gets added on top AFTER the pragmatic speaker has calculated its production probabilities by maximising the utility to the listener.


# 1.3: The parameters that determine the make-up of the population:

pop_size = 2


pragmatic_level = 'literal'  # This can be set to either 'literal', 'perspective-taking' or 'prag'
optimality_alpha = 1.0  # Goodman & Stuhlmuller (2013) fitted optimality_alpha = 3.4 to participant data with 4x3 lexicon of three number words ('one', 'two', and 'three') that could be mapped to four different world states (0, 1, 2, and 3)


perspectives = np.array([0., 1.])  # The different perspectives that agents can have
perspective_probs = np.array([.5, .5])  # The proportions with which the different perspectives will be present in the population
perspective_probs_string = saveresults.convert_array_to_string(perspective_probs) # Turns the perspective probs into a string in order to add it to file names


fixed_perspectives = np.array([0., 1.])  # Used to override the perspectives of the population which are otherwise chosen stochastically based on the parameters perspectives and perspective_probs above. This array has to have length equal to pop_size
print ''
print ''
print "fixed_perspectives are:"
print fixed_perspectives

learning_types = ['map', 'sample'] # The types of learning that the learners can do
learning_type_probs = np.array([0., 1.]) # The ratios with which the different learning types will be present in the population
learning_type_probs_string = saveresults.convert_array_to_string(learning_type_probs) # Turns the learning type probs into a string in order to add it to file names
if learning_type_probs[0] == 1.:
    learning_type_string = learning_types[0]
elif learning_type_probs[1] == 1.:
    learning_type_string = learning_types[1]


# 1.4: The parameters that determine the attributes of the learner:

learner_perspective = 0.  # The learner's perspective
learner_lex_type = 'empty_lex'  # The lexicon type of the learner. This will normally be 'empty_lex'
learner_learning_type = 'sample'  # The type of learning that the learner does. This can be set to either 'map' or 'sample'


# 1.5: The parameters that determine the learner's hypothesis space:

perspective_hyps = np.array([0., 1.])  # The perspective hypotheses that the learner will consider (1D numpy array)

which_lexicon_hyps = 'all' # Determines the set of lexicon hypotheses that the learner will consider. This can be set to either 'all', 'all_with_full_s_space' or 'only_optimal'
if which_lexicon_hyps == 'all':
    lexicon_hyps = hypspace.create_all_lexicons(n_meanings, n_signals) # The lexicon hypotheses that the learner will consider (1D numpy array)
elif which_lexicon_hyps == 'all_with_full_s_space':
    all_lexicon_hyps = hypspace.create_all_lexicons(n_meanings, n_signals)
    lexicon_hyps = hypspace.remove_subset_of_signals_lexicons(all_lexicon_hyps) # The lexicon hypotheses that the learner will consider (1D numpy array)
elif which_lexicon_hyps == 'only_optimal':
    lexicon_hyps = hypspace.create_all_optimal_lexicons(n_meanings, n_signals) # The lexicon hypotheses that the learner will consider (1D numpy array)


# 1.6: More parameters that determine the make-up of the population:

lexicon_probs = np.array([0. for x in range(len(lexicon_hyps)-1)]+[1.])

fixed_lexicons = np.array([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]] for x in range(pop_size)])  # Used to override the lexicons of the population which are otherwise chosen stochastically based on the parameters lexicon_hyps and lexicon_probs above. This array has to have length equal to pop_size
print ''
print ''
print "fixed_lexicons are:"
print fixed_lexicons


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
n_contexts = 800  # The number of contexts that the learner gets to see. If context_generation = 'most_optimal' n_contexts for n_meanings = 2 can be anything in the table of 4 (e.g. 20, 40, 60, 80, 100, etc.); for n_meanings = 3 anything in the table of 12 (e.g. 12, 36, 60, 84, 108, etc.); and for n_meanings = 4 anything in the table of 48 (e.g. 48, 96, 144, etc.).


speaker_order_type = 'opp_first_equal' # This can be set to either 'random', 'random_equal' (for random order with making sure that each speaker gets an equal amount of utterances) 'same_first' (first portion of input comes from same perspective, second portion of input from opposite perspective), 'same_first_equal' (where both speakers get to produce the exact same amount of utterances), 'opp_first' (vice versa) or 'opp_first_equal'
first_input_stage_ratio = 0.5 # This is the proportion of contexts that will make up the first input stage (see parameter 'speaker_order_type' above)


print ''
print ''
print "speaker_order_type is:"
print speaker_order_type


# 1.9: The parameters that determine how learning is measured:

theta_fixed = 0.01  # The learner is considered to have acquired the correct hypothesis if the posterior probability for that hypothesis exceeds (1.-theta_fixed)
## The parameters below serve to make a 'convergence time over theta' plot, where the amount of learning trials needed to reach a certain level of convergence on the correct hypothesis is plotted against different values of theta.
theta_start = 0.0
theta_stop = 1.0
theta_step = 0.00001
theta_range = np.arange(theta_start, theta_stop, theta_step)
theta_step_string = str(theta_step)
theta_step_string = theta_step_string.replace(".", "")


# 1.10: The parameters that determine the type and number of simulations that are run:

#FIXME: In the current implementation 'half_ambiguous_lex' can have different instantiations. Therefore, if we run a simulation where there are different lexicon_type_probs, we will want the run_type to be 'population_same_pop' to make sure that all the 'half ambiguous' speakers do have the SAME half ambiguous lexicon.
run_type = 'population_same_pop_dist_learner' # This can be set to 'population_diff_pop' if there are no speakers with the 'half_ambiguous' lexicon type, 'population_same_pop' if there are, 'population_same_pop_dist_learner' if the learner can distinguish between different speakers

if run_type == 'population_diff_pop' or run_type == 'population_same_pop':
    agent_type = 'no_p_distinction' # This can be set to either 'p_distinction' or 'no_p_distinction'. Determines whether the population is made up of DistinctionAgent objects or Agent objects (DistinctionAgents can learn different perspectives for different agents, Agents can only learn one perspective for all agents they learn from).
elif run_type == 'population_same_pop_dist_learner':
    agent_type = 'p_distinction' # This can be set to either 'p_distinction' or 'no_p_distinction'. Determines whether the population is made up of DistinctionAgent objects or Agent objects (DistinctionAgents can learn different perspectives for different agents, Agents can only learn one perspective for all agents they learn from).


if agent_type == 'no_p_distinction':
    hypothesis_space = hypspace.list_hypothesis_space(perspective_hyps, lexicon_hyps) # The full space of composite hypotheses that the learner will consider (2D numpy matrix with composite hypotheses on the rows, perspective hypotheses on column 0 and lexicon hypotheses on column 1)

elif agent_type == 'p_distinction':
    hypothesis_space = hypspace.list_hypothesis_space_with_speaker_distinction(perspective_hyps, lexicon_hyps, pop_size) # The full space of composite hypotheses that the learner will consider (2D numpy matrix with composite hypotheses on the rows, perspective hypotheses on column 0 and lexicon hypotheses on column 1)


n_runs = 10  # The number of runs of the simulation
report_every_r = 1

which_hyps_on_graph = 'all_hyps' # This is used in the plot_post_probs_over_lex_hyps() function, and can be set to either 'all_hyps' or 'lex_hyps_only'



#######################################################################################################################




def multi_runs_population_diff_pop(n_meanings, n_signals, n_runs, n_contexts, n_utterances, context_generation, context_type, context_size, helpful_contexts, learner_perspective, pop_size, speaker_order_type, first_input_stage_ratio, agent_type, perspectives, perspective_probs, sal_alpha, lexicon_hyps, lexicon_probs, error, extra_error, pragmatic_level, optimality_alpha, learning_types, learning_type_probs, learner_lex_type, learner_learning_type, hypothesis_space, perspective_hyps, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant):
    """
    :param n_meanings: integer specifying number of meanings ('objects') in the lexicon
    :param n_signals: integer specifying number of signals in the lexicon
    :param n_runs: integer specifying number of independent simulation runs to be run
    :param n_contexts: integer specifying the number of contexts to be observed by the learner
    :param n_utterances: integer specifying the number of utterances the learner gets to observe *per context*
    :param context_generation: can be set to either 'random', 'only_helpful' or 'optimal'
    :param context_type: can be set to either 'absolute' for meanings being in/out or 'continuous' for all meanings being present (but in different positions)
    :param context_size: this parameter is only used if the context_type is 'absolute' and determines the number of meanings present
    :param helpful_contexts: this parameter is only used if the parameter context_generation is set to either 'only_helpful' or 'optimal'
    :param learner_perspective:  float specifying the perspective of the learner (any float between 0.0 and 1.0)
    :param pop_size: integer specifying how many agents there are in the population
    :param speaker_order_type: can be set to either 'random', 'random_equal' (for random order with making sure that each speaker gets an equal amount of utterances) 'same_first' (first portion of input comes from same perspective, second portion of input from opposite perspective), 'same_first_equal' (where both speakers get to produce the exact same amount of utterances), 'opp_first' (vice versa) or 'opp_first_equal'
    :param first_input_stage_ratio: this is the proportion of contexts that will make up the first input stage (see parameter 'speaker_order_type' above)
    :param agent_type: can be set to either 'p_distinction' or 'no_p_distinction'. Determines whether the population is made up of DistinctionAgent objects or Agent objects (DistinctionAgents can learn different perspectives for different agents, Agents can only learn one perspective for all agents they learn from).
    :param perspectives: 1D numpy array containing floats specifying the different perspectives that agents can have
    :param perspective_probs: 1D numpy array containing floats specifying proportionss with which the different perspectives will be present in the population
    :param sal_alpha: float. Exponent that is used by the saliency function. sal_alpha=1 gives a directly proportional relationship between meaning distances and meaning saliencies, with sal_alpha>1 the saliency of meanings goes down exponentially as their distance from the agent increases (and with sal_alpha<1 the other way around)
    :param lexicon_hyps: the lexicon hypotheses that the learner will consider (1D numpy array)
    :param lexicon_probs: 1D numpy array containing floats which specify the probability with which each of the possible lexicons will occur in the population
    :param error: float specifying the probability that the speaker makes a production error (i.e. randomly chooses a signal that isn't associated with the intended referent)
    :param extra_error: can be set to either True or False. Determines whether the error specified above gets added on top AFTER the pragmatic speaker has calculated its production probabilities by maximising the utility to the listener.
    :param pragmatic_level: can be set to either 'literal', 'perspective-taking' or 'prag'
    :param optimality_alpha: optimality parameter in the RSA model for pragmatic speakers. Only used if the pragmatic_level_learner parameter is set to 'prag'
    :param learning_types: a list of the learning types present in the population; can contain 'sample' and 'map'
    :param learning_type_probs: a 1D numpy array containing floats specifying the proportions with which each learning type will be present in the population
    :param learner_lex_type: lexicon type of the learner. #FIXME: The learner has to be initiated with a lexicon type because I have not yet coded up a subclass of Agent that is only a Learner (for which the lexicon should not have to be specified in advance).
    :param learner_learning_type: can be set to either 'sample' for sampling from the posterior or 'map' for selecting only the maximum a posteriori hypothesis
    :param hypothesis_space: full space of composite hypotheses that the learner will consider (2D numpy matrix with composite hypotheses on the rows, perspective hypotheses on column 0 and lexicon hypotheses on column 1)
    :param perspective_hyps: the perspective hypotheses that the learner will consider (1D numpy array)
    :param perspective_prior_type: can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
    :param perspective_prior_strength: strength of the egocentric prior (only used if the perspective_prior_type parameter is set to 'egocentric')
    :param lexicon_prior_type: can be set to either 'neutral', 'ambiguous_fixed', 'half_ambiguous_fixed', 'expressivity_bias' or 'compressibility_bias'
    :param lexicon_prior_constant: determines the strength of the lexicon bias, with small c creating a STRONG prior and large c creating a WEAK prior. (And with c = 1000 creating an almost uniform prior.)
    :return: A list containing all the result arrays ('result_array_list') and a dictionary containing the keys of those arrays with the corresponding indices ('result_array_keys')
    """

    t0 = time.clock()

    # 1) First the matrices are created that will be saving the data_dict and the posteriors:
    multi_run_context_matrix = np.zeros((n_runs, n_contexts, n_meanings))
    multi_run_log_posterior_matrix = np.zeros((n_runs, ((n_contexts*n_utterances)+1), len(hypothesis_space)))

    # 2) Then the prior probability distribution for all agents is created:
    ## 2.1) First the perspective prior is created:
    perspective_prior = prior.create_perspective_prior(perspective_hyps, lexicon_hyps, perspective_prior_type, learner_perspective, perspective_prior_strength)
    ## 2.2) Then the lexicon prior is created:
    lexicon_prior = prior.create_lexicon_prior(lexicon_hyps, lexicon_prior_type, lexicon_prior_constant, error)

    for r in range(n_runs):
        if r % report_every_r == 0:
            print 'r = '+str(r)

        population = pop.Population(pop_size, n_meanings, n_signals, hypothesis_space, perspective_hyps, lexicon_hyps, learner_perspective, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant, perspectives, perspective_probs, sal_alpha, lexicon_probs, error, extra_error, pragmatic_level, optimality_alpha, n_contexts, context_type, context_generation, context_size, helpful_contexts, n_utterances, learning_types, learning_type_probs)

        perspective_prior_fixed = perspective_probs
        lexicon_prior_fixed = np.zeros(len(lexicon_hyps))
        for i in range(len(lexicon_hyps)):
            lexicon = lexicon_hyps[i]
            for j in range(len(population.lexicons)):
                if np.array_equal(lexicon, population.lexicons[j].lexicon):
                    lexicon_prior_fixed[i] = lexicon_probs[j]

        if context_generation == 'random':
            context_matrix = context.gen_context_matrix(context_type, n_meanings, context_size, n_contexts)
        elif context_generation == 'only_helpful':
            context_matrix = context.gen_helpful_context_matrix(n_meanings, n_contexts, helpful_contexts)
        elif context_generation == 'optimal':
            context_matrix = context.gen_helpful_context_matrix_fixed_order(n_meanings, n_contexts, helpful_contexts)

        # TODO: Figure out why it is that the composite_log_priors_population are not treated as a global variable but rather as a local variable of the learner object that gets updated as learner.log_posteriors get updated --> I think the solution might be to declare it 'global' somewhere in the Agent class

        learner_lexicon = lex.Lexicon(learner_lex_type, n_meanings, n_signals)
        if learner_type == 'perspective_unknown':
            composite_log_priors = prior.list_composite_log_priors(agent_type, pop_size, hypothesis_space, perspective_hyps, lexicon_hyps, perspective_prior, lexicon_prior_fixed) # These have to be recalculated fresh after every run so that each new learner is initialized with the original prior distribution, rather than the final posteriors of the last learner!
        elif learner_type == 'lexicon_unknown':
            composite_log_priors = prior.list_composite_log_priors(agent_type, pop_size, hypothesis_space, perspective_hyps, lexicon_hyps, perspective_prior_fixed, lexicon_prior) # These have to be recalculated fresh after every run so that each new learner is initialized with the original prior distribution, rather than the final posteriors of the last learner!
        elif learner_type == 'both_unknown':
            composite_log_priors = prior.list_composite_log_priors(agent_type, pop_size, hypothesis_space, perspective_hyps, lexicon_hyps, perspective_prior, lexicon_prior) # These have to be recalculated fresh after every run so that each new learner is initialized with the original prior distribution, rather than the final posteriors of the last learner!

        learner = pop.Agent(perspective_hyps, lexicon_hyps, composite_log_priors, composite_log_priors, learner_perspective, sal_alpha, learner_lexicon, learner_learning_type)


        speaker_order = pop.create_speaker_order_single_generation(population, speaker_order_type, n_contexts, first_input_stage_ratio)


        if context_generation == 'random':
            data = population.produce_pop_data(context_matrix, n_utterances, speaker_order)
            log_posteriors_per_data_point_matrix = learner.inference(n_contexts, n_utterances, data, error)
        elif context_generation == 'optimal':
            data = population.produce_pop_data_fixed_contexts(context_matrix, n_utterances, speaker_order, helpful_contexts, n_signals)
            log_posteriors_per_data_point_matrix = learner.inference_on_signal_counts_data(data, error)



        # FIXME: If I want the half_ambiguous lexicon to be generated with the ambiguous mappings chosen at random, I have to make sure that the majority_lex_hyp_indices and majority_composite_hyp_index are logged for each run separately

        multi_run_context_matrix[r] = data.contexts
        multi_run_log_posterior_matrix[r] = log_posteriors_per_data_point_matrix

    run_time_mins = (time.clock()-t0)/60.

    results_dict = {'multi_run_context_matrix':multi_run_context_matrix, 'multi_run_log_posterior_matrix':multi_run_log_posterior_matrix, 'population_lexicons':population.lexicons, 'run_time_mins':run_time_mins}
    return results_dict



def multi_runs_population_same_pop(n_meanings, n_signals, n_runs, n_contexts, n_utterances, context_generation, context_type, context_size, helpful_contexts, learner_perspective, pop_size, speaker_order_type, first_input_stage_ratio, agent_type, perspectives, perspective_probs, sal_alpha, lexicon_hyps, lexicon_probs, error, extra_error, pragmatic_level, optimality_alpha, learning_types, learning_type_probs, learner_lex_type, learner_learning_type, hypothesis_space, perspective_hyps, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant):
    """
    :param n_meanings: integer specifying number of meanings ('objects') in the lexicon
    :param n_signals: integer specifying number of signals in the lexicon
    :param n_runs: integer specifying number of independent simulation runs to be run
    :param n_contexts: integer specifying the number of contexts to be observed by the learner
    :param n_utterances: integer specifying the number of utterances the learner gets to observe *per context*
    :param context_generation: can be set to either 'random', 'only_helpful' or 'optimal'
    :param context_type: can be set to either 'absolute' for meanings being in/out or 'continuous' for all meanings being present (but in different positions)
    :param context_size: this parameter is only used if the context_type is 'absolute' and determines the number of meanings present
    :param helpful_contexts: this parameter is only used if the parameter context_generation is set to either 'only_helpful' or 'optimal'
    :param learner_perspective:  float specifying the perspective of the learner (any float between 0.0 and 1.0)
    :param pop_size: integer specifying how many agents there are in the population
    :param speaker_order_type: can be set to either 'random', 'random_equal' (for random order with making sure that each speaker gets an equal amount of utterances) 'same_first' (first portion of input comes from same perspective, second portion of input from opposite perspective), 'same_first_equal' (where both speakers get to produce the exact same amount of utterances), 'opp_first' (vice versa) or 'opp_first_equal'
    :param first_input_stage_ratio: this is the proportion of contexts that will make up the first input stage (see parameter 'speaker_order_type' above)
    :param agent_type: can be set to either 'p_distinction' or 'no_p_distinction'. Determines whether the population is made up of DistinctionAgent objects or Agent objects (DistinctionAgents can learn different perspectives for different agents, Agents can only learn one perspective for all agents they learn from).
    :param perspectives: 1D numpy array containing floats specifying the different perspectives that agents can have
    :param perspective_probs: 1D numpy array containing floats specifying proportionss with which the different perspectives will be present in the population
    :param sal_alpha: float. Exponent that is used by the saliency function. sal_alpha=1 gives a directly proportional relationship between meaning distances and meaning saliencies, with sal_alpha>1 the saliency of meanings goes down exponentially as their distance from the agent increases (and with sal_alpha<1 the other way around)
    :param lexicon_hyps: the lexicon hypotheses that the learner will consider (1D numpy array)
    :param lexicon_probs: 1D numpy array containing floats which specify the probability with which each of the possible lexicons will occur in the population
    :param error: float specifying the probability that the speaker makes a production error (i.e. randomly chooses a signal that isn't associated with the intended referent)
    :param extra_error: can be set to either True or False. Determines whether the error specified above gets added on top AFTER the pragmatic speaker has calculated its production probabilities by maximising the utility to the listener.
    :param pragmatic_level: can be set to either 'literal', 'perspective-taking' or 'prag'
    :param optimality_alpha: optimality parameter in the RSA model for pragmatic speakers. Only used if the pragmatic_level_learner parameter is set to 'prag'
    :param learning_types: a list of the learning types present in the population; can contain 'sample' and 'map'
    :param learning_type_probs: a 1D numpy array containing floats specifying the proportions with which each learning type will be present in the population
    :param learner_lex_type: lexicon type of the learner. #FIXME: The learner has to be initiated with a lexicon type because I have not yet coded up a subclass of Agent that is only a Learner (for which the lexicon should not have to be specified in advance).
    :param learner_learning_type: can be set to either 'sample' for sampling from the posterior or 'map' for selecting only the maximum a posteriori hypothesis
    :param hypothesis_space: full space of composite hypotheses that the learner will consider (2D numpy matrix with composite hypotheses on the rows, perspective hypotheses on column 0 and lexicon hypotheses on column 1)
    :param perspective_hyps: the perspective hypotheses that the learner will consider (1D numpy array)
    :param perspective_prior_type: can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
    :param perspective_prior_strength: strength of the egocentric prior (only used if the perspective_prior_type parameter is set to 'egocentric')
    :param lexicon_prior_type: can be set to either 'neutral', 'ambiguous_fixed', 'half_ambiguous_fixed', 'expressivity_bias' or 'compressibility_bias'
    :param lexicon_prior_constant: determines the strength of the lexicon bias, with small c creating a STRONG prior and large c creating a WEAK prior. (And with c = 1000 creating an almost uniform prior.)
    :return: A list containing all the result arrays ('result_array_list') and a dictionary containing the keys of those arrays with the corresponding indices ('result_array_keys')
    """

    # 1) First the matrices are created that will be saving the data_dict and the posteriors:
    multi_run_context_matrix = np.zeros((n_runs, n_contexts, n_meanings))
    multi_run_utterances_matrix = np.zeros((n_runs, n_contexts, n_utterances))
    multi_run_log_posterior_matrix = np.zeros((n_runs, ((n_contexts*n_utterances)+1), len(hypothesis_space)))

    # 2) Then the population is created:
    # FIXME: Currently the different runs of the simulation run with the same population, instead of having a new population being initialized with every run. This is because the majority lexicon is not logged for each run separately.



    population = pop.Population(pop_size, n_meanings, n_signals, hypothesis_space, perspective_hyps, lexicon_hyps, learner_perspective, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant, perspectives, perspective_probs, sal_alpha, lexicon_probs, error, extra_error, pragmatic_level, optimality_alpha, n_contexts, context_type, context_generation, context_size, helpful_contexts, n_utterances, learning_types, learning_type_probs)

    # 3) Then the prior probability distribution for all agents is created:
    ## 3.1) First the perspective prior is created:
    perspective_prior = prior.create_perspective_prior(perspective_hyps, lexicon_hyps, perspective_prior_type, learner_perspective, perspective_prior_strength)

    ## 3.2) Then the lexicon prior is created:
    lexicon_prior = prior.create_lexicon_prior(lexicon_hyps, lexicon_prior_type, lexicon_prior_constant, error)


    perspective_prior_fixed = perspective_probs
    lexicon_prior_fixed = np.zeros(len(lexicon_hyps))
    for i in range(len(lexicon_hyps)):
        lexicon = lexicon_hyps[i]
        for j in range(len(population.lexicons)):
            if np.array_equal(lexicon, population.lexicons[j].lexicon):
                lexicon_prior_fixed[i] = lexicon_probs[j]
    for r in range(n_runs):
        print str(r)

        if context_generation == 'random':
            context_matrix = context.gen_context_matrix(context_type, n_meanings, context_size, n_contexts)
        elif context_generation == 'only_helpful':
            context_matrix = context.gen_helpful_context_matrix(n_meanings, n_contexts, helpful_contexts)
        elif context_generation == 'optimal':
            context_matrix = context.gen_helpful_context_matrix_fixed_order(n_meanings, n_contexts, helpful_contexts)

        # TODO: Figure out why it is that the composite_log_priors_population are not treated as a global variable but rather as a local variable of the learner object that gets updated as learner.log_posteriors get updated --> I think the solution might be to declare it 'global' somewhere in the Agent class

        learner_lexicon = lex.Lexicon(learner_lex_type, n_meanings, n_signals)
        if learner_type == 'perspective_unknown':
            composite_log_priors = prior.list_composite_log_priors(agent_type, pop_size, hypothesis_space, perspective_prior, lexicon_prior_fixed) # These have to be recalculated fresh after every run so that each new learner is initialized with the original prior distribution, rather than the final posteriors of the last learner!
        elif learner_type == 'lexicon_unknown':
            composite_log_priors = prior.list_composite_log_priors(agent_type, pop_size, hypothesis_space, perspective_prior_fixed, lexicon_prior) # These have to be recalculated fresh after every run so that each new learner is initialized with the original prior distribution, rather than the final posteriors of the last learner!
        elif learner_type == 'both_unknown':
            composite_log_priors = prior.list_composite_log_priors(agent_type, pop_size, hypothesis_space, perspective_hyps, lexicon_hyps, perspective_prior, lexicon_prior) # These have to be recalculated fresh after every run so that each new learner is initialized with the original prior distribution, rather than the final posteriors of the last learner!

        learner = pop.Agent(perspective_hyps, lexicon_hyps, composite_log_priors, composite_log_priors, learner_perspective, sal_alpha, learner_lexicon, learner_learning_type)

        speaker_order = pop.create_speaker_order_single_generation(population, speaker_order_type, n_contexts, first_input_stage_ratio)
        if context_generation == 'random':
            data = population.produce_pop_data(context_matrix, n_utterances, speaker_order)
        elif context_generation == 'optimal':
            data = population.produce_pop_data_fixed_contexts(context_matrix, n_utterances, speaker_order, helpful_contexts, n_signals)
        log_posteriors_per_data_point_matrix = learner.inference(n_contexts, n_utterances, data, error)

        majority_p_hyp_indices = measur.find_majority_hyp_indices(hypothesis_space, population.perspectives, population.perspective_probs, population.lexicons, population.lexicon_type_probs, 'perspective')
        majority_lex_hyp_indices = measur.find_majority_hyp_indices(hypothesis_space, population.perspectives, population.perspective_probs, population.lexicons, population.lexicon_type_probs, 'lexicon')
        majority_composite_hyp_indices = measur.find_majority_hyp_indices(hypothesis_space, population.perspectives, population.perspective_probs, population.lexicons, population.lexicon_type_probs, 'composite')

        # FIXME: If I want the half_ambiguous lexicon to be generated with the ambiguous mappings chosen at random, I have to make sure that the majority_lex_hyp_indices and majority_composite_hyp_index are logged for each run separately

        majority_lexicon = measur.find_majority_lexicon(population.lexicons, population.lexicon_type_probs)
        multi_run_context_matrix[r] = data.contexts
        multi_run_utterances_matrix[r] = data.utterances
        multi_run_log_posterior_matrix[r] = log_posteriors_per_data_point_matrix
    results_dict = {'multi_run_context_matrix':multi_run_context_matrix, 'multi_run_utterances_matrix':multi_run_utterances_matrix, 'multi_run_log_posterior_matrix':multi_run_log_posterior_matrix, 'majority_p_hyp_indices':majority_p_hyp_indices, 'majority_lex_hyp_indices':majority_lex_hyp_indices, 'majority_composite_hyp_indices':majority_composite_hyp_indices, 'majority_lexicon':majority_lexicon, 'population_lexicons':population.lexicons, 'learner_hypothesis_space':learner.hypothesis_space}
    return results_dict



#TODO: Describe what all the steps in this function are for
def multi_runs_population_same_pop_distinction_learner(n_meanings, n_signals, n_runs, n_contexts, n_utterances, context_generation, context_type, context_size, helpful_contexts, learner_perspective, pop_size, speaker_order_type, first_input_stage_ratio, agent_type, perspectives, perspective_probs, sal_alpha, lexicon_hyps, lexicon_probs, error, extra_error, pragmatic_level, optimality_alpha, learning_types, learning_type_probs, learner_lex_type, learner_learning_type, hypothesis_space, perspective_hyps, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant, fixed_perspectives=None, fixed_lexicons=None):
    """
    :param n_meanings: integer specifying number of meanings ('objects') in the lexicon
    :param n_signals: integer specifying number of signals in the lexicon
    :param n_runs: integer specifying number of independent simulation runs to be run
    :param n_contexts: integer specifying the number of contexts to be observed by the learner
    :param n_utterances: integer specifying the number of utterances the learner gets to observe *per context*
    :param context_generation: can be set to either 'random', 'only_helpful' or 'optimal'
    :param context_type: can be set to either 'absolute' for meanings being in/out or 'continuous' for all meanings being present (but in different positions)
    :param context_size: this parameter is only used if the context_type is 'absolute' and determines the number of meanings present
    :param helpful_contexts: this parameter is only used if the parameter context_generation is set to either 'only_helpful' or 'optimal'
    :param learner_perspective:  float specifying the perspective of the learner (any float between 0.0 and 1.0)
    :param pop_size: integer specifying how many agents there are in the population
    :param speaker_order_type: can be set to either 'random', 'random_equal' (for random order with making sure that each speaker gets an equal amount of utterances) 'same_first' (first portion of input comes from same perspective, second portion of input from opposite perspective), 'same_first_equal' (where both speakers get to produce the exact same amount of utterances), 'opp_first' (vice versa) or 'opp_first_equal'
    :param first_input_stage_ratio: this is the proportion of contexts that will make up the first input stage (see parameter 'speaker_order_type' above)
    :param agent_type: can be set to either 'p_distinction' or 'no_p_distinction'. Determines whether the population is made up of DistinctionAgent objects or Agent objects (DistinctionAgents can learn different perspectives for different agents, Agents can only learn one perspective for all agents they learn from).
    :param perspectives: 1D numpy array containing floats specifying the different perspectives that agents can have
    :param perspective_probs: 1D numpy array containing floats specifying proportionss with which the different perspectives will be present in the population
    :param sal_alpha: float. Exponent that is used by the saliency function. sal_alpha=1 gives a directly proportional relationship between meaning distances and meaning saliencies, with sal_alpha>1 the saliency of meanings goes down exponentially as their distance from the agent increases (and with sal_alpha<1 the other way around)
    :param lexicon_hyps: the lexicon hypotheses that the learner will consider (1D numpy array)
    :param lexicon_probs: 1D numpy array containing floats which specify the probability with which each of the possible lexicons will occur in the population
    :param error: float specifying the probability that the speaker makes a production error (i.e. randomly chooses a signal that isn't associated with the intended referent)
    :param extra_error: can be set to either True or False. Determines whether the error specified above gets added on top AFTER the pragmatic speaker has calculated its production probabilities by maximising the utility to the listener.
    :param pragmatic_level: can be set to either 'literal', 'perspective-taking' or 'prag'
    :param optimality_alpha: optimality parameter in the RSA model for pragmatic speakers. Only used if the pragmatic_level_learner parameter is set to 'prag'
    :param learning_types: a list of the learning types present in the population; can contain 'sample' and 'map'
    :param learning_type_probs: a 1D numpy array containing floats specifying the proportions with which each learning type will be present in the population
    :param learner_lex_type: lexicon type of the learner. #FIXME: The learner has to be initiated with a lexicon type because I have not yet coded up a subclass of Agent that is only a Learner (for which the lexicon should not have to be specified in advance).
    :param learner_learning_type: can be set to either 'sample' for sampling from the posterior or 'map' for selecting only the maximum a posteriori hypothesis
    :param hypothesis_space: full space of composite hypotheses that the learner will consider (2D numpy matrix with composite hypotheses on the rows, perspective hypotheses on column 0 and lexicon hypotheses on column 1)
    :param perspective_hyps: the perspective hypotheses that the learner will consider (1D numpy array)
    :param lexicon_hyps: the lexicon hypotheses that the learner will consider (1D numpy array)
    :param perspective_prior_type: can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
    :param perspective_prior_strength: strength of the egocentric prior (only used if the perspective_prior_type parameter is set to 'egocentric')
    :param lexicon_prior_type: can be set to either 'neutral', 'ambiguous_fixed', 'half_ambiguous_fixed', 'expressivity_bias' or 'compressibility_bias'
    :param lexicon_prior_constant: determines the strength of the lexicon bias, with small c creating a STRONG prior and large c creating a WEAK prior. (And with c = 1000 creating an almost uniform prior.)
    :return: A list containing all the result arrays ('result_array_list') and a dictionary containing the keys of those arrays with the corresponding indices ('result_array_keys')
    """

    print ''
    print ''
    print ''
    print ''
    print 'This is the multi_runs_population_same_pop_distinction_learner() function:'


    ## 1) First the full composite hypothesis space is assembled in a matrix, containing each possible combination of lexicon hypothesis and perspective hypothesis for the different speakers:
    hypothesis_space = hypspace.list_hypothesis_space_with_speaker_distinction(perspective_hyps, lexicon_hyps, pop_size)

    # 2) Then the matrices are created in which the data_dict and the developing posterior distributions will be saved:
    multi_run_context_matrix = np.zeros((n_runs, n_contexts, n_meanings))
    multi_run_utterances_matrix = np.zeros((n_runs, n_contexts, n_utterances))
    multi_run_log_posterior_matrix = np.zeros((n_runs, ((n_contexts*n_utterances)+1), len(hypothesis_space)))
    multi_run_perspectives_per_speaker_matrix = np.zeros((n_runs, pop_size))
    multi_run_lexicons_per_speaker_matrix = np.zeros((n_runs, pop_size, n_meanings, n_signals))

    # 3) Then the prior probability distribution for all agents is created:
    ## 3.1) First the perspective prior is created:
    perspective_prior = prior.create_perspective_prior(perspective_hyps, lexicon_hyps, perspective_prior_type, learner_perspective, perspective_prior_strength)

    ## 3.2) Then the lexicon prior is created:
    lexicon_prior = prior.create_lexicon_prior(lexicon_hyps, lexicon_prior_type, lexicon_prior_constant, error)

    ## 3.3) And finally the full composite prior matrix is created using the separate lexicon_prior and perspective_prior, and following the configuration of hypothesis_space
    composite_log_priors = prior.list_composite_log_priors_with_speaker_distinction(hypothesis_space, perspective_hyps, lexicon_hyps, perspective_prior, lexicon_prior, pop_size)

    # 4) Then the population is created:
    ## 4.1) First the population's lexicons are determined:
    lexicons = []
    for i in range(len(lexicon_hyps)):
        lex_hyp = lexicon_hyps[i]
        lexicon = lex.Lexicon('specified_lexicon', n_meanings, n_signals, specified_lexicon=lex_hyp)
        lexicon = lexicon
        lexicons.append(lexicon)

    lexicons_per_agent = np.random.choice(lexicons, pop_size, replace=True, p=lexicon_probs)

    ## 4.2) Then the population's perspectives are determined:

    perspectives_per_agent = np.random.choice(perspectives, pop_size, replace=True, p=perspective_probs)

    ## 4.3) Then the population's learning types are determined:
    for i in range(pop_size):
        learning_types_per_agent = np.random.choice(learning_types, pop_size, replace=True, p=learning_type_probs)

    ## 4.4) Then the population itself is created:
    population = pop.DistinctionPopulation(pop_size, n_meanings, n_signals, hypothesis_space, perspective_hyps, lexicon_hyps, learner_perspective, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant, composite_log_priors, perspectives, perspectives_per_agent, perspective_probs, sal_alpha, lexicons, lexicons_per_agent, error, n_contexts, context_type, context_generation, context_size, helpful_contexts, n_utterances, learning_types, learning_types_per_agent, learning_type_probs)


    #TODO: I've added the optional input arguments 'fixed_perspectives' and 'fixed_lexicons' and the for-loop below to allow me to set the lexicons and perspectives of the input speakers (to use for the staging of input from different perspectives simulations):
    if fixed_perspectives is not None:
        for i in range(len(population.population)):
            speaker = population.population[i]
            speaker.perspective = fixed_perspectives[i]
        population.perspectives_per_agent = fixed_perspectives

    if fixed_lexicons is not None:
        for i in range(len(population.population)):
            speaker = population.population[i]
            speaker.lexicon = lex.Lexicon('specified_lexicon', n_meanings, n_signals, specified_lexicon=fixed_lexicons[i])

    for r in range(n_runs):
        if r % report_every_r == 0:
            print 'r = '+str(r)

        speaker_order = pop.create_speaker_order_single_generation(population, speaker_order_type, n_contexts, first_input_stage_ratio)

        multi_run_perspectives_per_speaker_matrix[r] = population.perspectives_per_agent

        lexicons_per_speaker = []
        for agent in population.population:
            agent_lexicon = agent.lexicon.lexicon
            lexicons_per_speaker.append(agent_lexicon)
        multi_run_lexicons_per_speaker_matrix[r] = lexicons_per_speaker

        perspective_prior_fixed = perspective_probs
        lexicon_prior_fixed = np.zeros(len(lexicon_hyps))
        for i in range(len(lexicon_hyps)):
            lexicon = lexicon_hyps[i]
            for j in range(len(population.lexicons)):
                if np.array_equal(lexicon, population.lexicons[j].lexicon):
                    lexicon_prior_fixed[i] = lexicon_probs[j]

        if context_generation == 'random':
            context_matrix = context.gen_context_matrix(context_type, n_meanings, context_size, n_contexts)
        elif context_generation == 'only_helpful':
            context_matrix = context.gen_helpful_context_matrix(n_meanings, n_contexts, helpful_contexts)
        elif context_generation == 'optimal':
            context_matrix = context.gen_helpful_context_matrix_fixed_order(n_meanings, n_contexts, helpful_contexts)

        learner_lexicon = lex.Lexicon(learner_lex_type, n_meanings, n_signals)
        if learner_type == 'perspective_unknown':
            composite_log_priors = prior.list_composite_log_priors_with_speaker_distinction(hypothesis_space, perspective_hyps, lexicon_hyps, perspective_prior, lexicon_prior_fixed, pop_size)
            # These have to be recalculated fresh after every run so that each new learner is initialized with the original prior distribution, rather than the final posteriors of the last learner!
        elif learner_type == 'lexicon_unknown':
            composite_log_priors = prior.list_composite_log_priors_with_speaker_distinction(hypothesis_space, perspective_hyps, lexicon_hyps, perspective_prior_fixed, lexicon_prior, pop_size)
            # These have to be recalculated fresh after every run so that each new learner is initialized with the original prior distribution, rather than the final posteriors of the last learner!
        elif learner_type == 'both_unknown':
            composite_log_priors = prior.list_composite_log_priors_with_speaker_distinction(hypothesis_space, perspective_hyps, lexicon_hyps, perspective_prior, lexicon_prior, pop_size) # These have to be recalculated fresh after every run so that each new learner is initialized with the original prior distribution, rather than the final posteriors of the last learner!

        learner = pop.DistinctionAgent(perspective_hyps, lexicon_hyps, composite_log_priors, composite_log_priors, learner_perspective, sal_alpha, learner_lexicon, learner_learning_type, pop_size)

        if r == 0:

            print ''
            print ''
            print ''
            print ''
            print 'population is:'
            population.print_population()

            print ''
            print ''
            print 'learner is:'
            learner.print_agent()

        speaker_annotated_data = population.produce_pop_data(n_meanings, n_signals, error, context_matrix, n_utterances, speaker_order)

        log_posteriors_per_data_point_matrix = learner.inference(n_contexts, n_utterances, speaker_annotated_data, error)
        # TODO: Figure out why it is that the composite_log_priors_population are not treated as a global variable but rather as a local variable of the learner object that gets updated as learner.log_posteriors get updated --> I think the solution might be to declare it 'global' somewhere in the Agent class

        multi_run_context_matrix[r] = speaker_annotated_data.contexts
        multi_run_utterances_matrix[r] = speaker_annotated_data.utterances
        multi_run_log_posterior_matrix[r] = log_posteriors_per_data_point_matrix

    results_dict = {'multi_run_context_matrix':multi_run_context_matrix, 'multi_run_utterances_matrix':multi_run_utterances_matrix, 'multi_run_log_posterior_matrix':multi_run_log_posterior_matrix, 'multi_run_perspectives_per_speaker_matrix':multi_run_perspectives_per_speaker_matrix, 'multi_run_lexicons_per_speaker_matrix':multi_run_lexicons_per_speaker_matrix, 'population_lexicons':population.lexicons, 'learner_hypothesis_space':learner.hypothesis_space}

    return results_dict



######################################################################################################################
# Below the actual running of the simulation happens:

if __name__ == "__main__":
    if n_runs > 0:

        t0 = time.clock()


        if run_type == 'population_diff_pop':
            results_dict = multi_runs_population_diff_pop(n_meanings, n_signals, n_runs, n_contexts, n_utterances, context_generation, context_type, context_size, helpful_contexts, learner_perspective, pop_size, speaker_order_type, first_input_stage_ratio, agent_type, perspectives, perspective_probs, sal_alpha, lexicon_hyps, lexicon_probs, error, extra_error, pragmatic_level, optimality_alpha, learning_types, learning_type_probs, learner_lex_type, learner_learning_type, hypothesis_space, perspective_hyps, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant)

        elif run_type == 'population_same_pop':
            results_dict = multi_runs_population_same_pop(n_meanings, n_signals, n_runs, n_contexts, n_utterances, context_generation, context_type, context_size, helpful_contexts, learner_perspective, pop_size, speaker_order_type, first_input_stage_ratio, agent_type, perspectives, perspective_probs, sal_alpha, lexicon_hyps, lexicon_probs, error, extra_error, pragmatic_level, optimality_alpha, learning_types, learning_type_probs, learner_lex_type, learner_learning_type, hypothesis_space, perspective_hyps, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant)

        elif run_type == 'population_same_pop_dist_learner':
            results_dict = multi_runs_population_same_pop_distinction_learner(n_meanings, n_signals, n_runs, n_contexts, n_utterances, context_generation, context_type, context_size, helpful_contexts, learner_perspective, pop_size, speaker_order_type, first_input_stage_ratio, agent_type, perspectives, perspective_probs, sal_alpha, lexicon_hyps, lexicon_probs, error, extra_error, pragmatic_level, optimality_alpha, learning_types, learning_type_probs, learner_lex_type, learner_learning_type, hypothesis_space, perspective_hyps, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant, fixed_perspectives=fixed_perspectives, fixed_lexicons=fixed_lexicons)


        run_simulation_time = time.clock()-t0
        print 
        print 'run_simulation_time is:'
        print str((run_simulation_time/60))+" m"





        t1 = time.clock()


        multi_run_log_posterior_matrix = results_dict['multi_run_log_posterior_matrix']

        unlogged_multi_run_posterior_matrix = np.exp(multi_run_log_posterior_matrix)
        # print 
        # print 
        # print "unlogged_multi_run_posterior_matrix is:"
        # print unlogged_multi_run_posterior_matrix
        # print "unlogged_multi_run_posterior_matrix.shape is:"
        # print unlogged_multi_run_posterior_matrix.shape


        if run_type == 'population_same_pop_dist_learner':
            #TODO: Do something here
            pass


        if run_type == 'population_same_pop_dist_learner':

            learner_hypothesis_space = results_dict['learner_hypothesis_space']

            # print 
            # print "learner_hypothesis_space (with indices) is:"
            # print learner_hypothesis_space
            #
            # print "perspective_hyps are:"
            # print perspective_hyps
            #
            # print "lexicon_hyps are:"
            # print lexicon_hyps

            multi_run_perspectives_per_speaker_matrix = results_dict['multi_run_perspectives_per_speaker_matrix']

            # print 
            # print "multi_run_perspectives_per_speaker_matrix is:"
            # print multi_run_perspectives_per_speaker_matrix

            multi_run_lexicons_per_speaker_matrix = results_dict['multi_run_lexicons_per_speaker_matrix']

            # print 
            # print "multi_run_lexicons_per_speaker_matrix is:"
            # print multi_run_lexicons_per_speaker_matrix

            real_speaker_perspectives = multi_run_perspectives_per_speaker_matrix[0]
            print ''
            print ''
            print "real_speaker_perspectives are:"
            print real_speaker_perspectives

            real_lexicon = multi_run_lexicons_per_speaker_matrix[0][0]

            print "real_lexicon is:"
            print real_lexicon


            correct_p_hyp_indices_per_speaker = []
            for speaker_id in range(pop_size):
                correct_p_hyp_indices = measur.find_correct_hyp_indices_with_speaker_distinction(learner_hypothesis_space, perspective_hyps, lexicon_hyps, real_speaker_perspectives, speaker_id, real_lexicon, 'perspective')
                correct_p_hyp_indices_per_speaker.append(correct_p_hyp_indices)
                np.asarray(correct_p_hyp_indices_per_speaker)


            for speaker in correct_p_hyp_indices_per_speaker:
                print ''
                print ''
                print "correct_p_hyp_indices this speaker are:"
                print speaker
                print "learner_hypothesis_space[speaker[-1]] is:"
                print learner_hypothesis_space[speaker[-1]]
                print "perspective_hyps[learner_hypothesis_space[speaker[-1]][0]] is:"
                print perspective_hyps[learner_hypothesis_space[speaker[-1]][0]]


            correct_lex_hyp_indices = measur.find_correct_hyp_indices(learner_hypothesis_space, perspective_hyps, lexicon_hyps, real_speaker_perspectives, real_lexicon, 'lexicon')


            print "correct_lex_hyp_indices are:"
            print correct_lex_hyp_indices

            print "learner_hypothesis_space[correct_lex_hyp_indices] is:"
            print learner_hypothesis_space[correct_lex_hyp_indices]

            for hyp in learner_hypothesis_space[correct_lex_hyp_indices]:
                if hyp[1] <= len(lexicon_hyps):
                    print "lexicon_hyps[hyp[1]] is:"
                    print lexicon_hyps[hyp[1]]


            correct_composite_hyp_indices = measur.find_correct_hyp_indices(learner_hypothesis_space, perspective_hyps, lexicon_hyps, real_speaker_perspectives, real_lexicon, 'composite')


            print "correct_composite_hyp_indices are:"
            print correct_composite_hyp_indices

            print "learner_hypothesis_space[correct_composite_hyp_indices] is:"
            print learner_hypothesis_space[correct_composite_hyp_indices]

            if learner_hypothesis_space[correct_composite_hyp_indices][0][1] <= len(lexicon_hyps):
                print "lexicon_hyps[learner_hypothesis_space[correct_composite_hyp_indices][0][1]] is:"
                print lexicon_hyps[learner_hypothesis_space[correct_composite_hyp_indices][0][1]]


            percentiles_p_hyp_posterior_mass_correct_per_speaker = []
            for speaker_id in range(pop_size):
                percentiles_p_hyp_posterior_mass_correct = measur.calc_hyp_correct_posterior_mass_percentiles(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix, correct_p_hyp_indices_per_speaker[speaker_id])
                percentiles_p_hyp_posterior_mass_correct_per_speaker.append(percentiles_p_hyp_posterior_mass_correct)
            percentiles_p_hyp_posterior_mass_correct_per_speaker = np.asarray(percentiles_p_hyp_posterior_mass_correct_per_speaker)


            print ''
            print ''
            print ''
            # print "percentiles_p_hyp_posterior_mass_correct_per_speaker[:, 1] is:"
            # print percentiles_p_hyp_posterior_mass_correct_per_speaker[:, 1]
            print "percentiles_p_hyp_posterior_mass_correct_per_speaker[:, 1].shape is:"
            print percentiles_p_hyp_posterior_mass_correct_per_speaker[:, 1].shape

            print ''
            print ''
            # print "percentiles_p_hyp_posterior_mass_correct_per_speaker[:, 3] is:"
            # print percentiles_p_hyp_posterior_mass_correct_per_speaker[:, 3]
            print "percentiles_p_hyp_posterior_mass_correct_per_speaker[:, 3].shape is:"
            print percentiles_p_hyp_posterior_mass_correct_per_speaker[:, 3].shape


            percentiles_lex_hyp_posterior_mass_correct = measur.calc_hyp_correct_posterior_mass_percentiles(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix, correct_lex_hyp_indices)

            print ''
            print ''
            print ''
            # print "percentiles_lex_hyp_posterior_mass_correct[1] is:"
            # print percentiles_lex_hyp_posterior_mass_correct[1]
            print "percentiles_lex_hyp_posterior_mass_correct[1].shape is:"
            print percentiles_lex_hyp_posterior_mass_correct[1].shape

            print ''
            print ''
            # print "percentiles_lex_hyp_posterior_mass_correct[3] is:"
            # print percentiles_lex_hyp_posterior_mass_correct[3]
            print "percentiles_lex_hyp_posterior_mass_correct[3].shape is:"
            print percentiles_lex_hyp_posterior_mass_correct[3].shape

            percentiles_composite_hyp_posterior_mass_correct = measur.calc_hyp_correct_posterior_mass_percentiles(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix, correct_composite_hyp_indices)

            print ''
            print ''
            print ''
            # print "percentiles_composite_hyp_posterior_mass_correct[1] is:"
            # print percentiles_composite_hyp_posterior_mass_correct[1]
            print "percentiles_composite_hyp_posterior_mass_correct[1].shape is:"
            print percentiles_composite_hyp_posterior_mass_correct[1].shape

            print ''
            print ''
            # print "percentiles_composite_hyp_posterior_mass_correct[3] is:"
            # print percentiles_composite_hyp_posterior_mass_correct[3]
            print "percentiles_composite_hyp_posterior_mass_correct[3].shape is:"
            print percentiles_composite_hyp_posterior_mass_correct[3].shape


            # percentiles_cumulative_belief_perspective_per_speaker = []
            # for speaker_id in range(pop_size):
            #     percentiles_cumulative_belief_perspective = measur.calc_cumulative_belief_in_correct_hyps_per_speaker(n_runs, n_contexts, multi_run_log_posterior_matrix, learner_hypothesis_space, perspective_hyps, lexicon_hyps, multi_run_perspectives_per_speaker_matrix, speaker_id, multi_run_lexicons_per_speaker_matrix, 'perspective')
            #     percentiles_cumulative_belief_perspective_per_speaker.append(percentiles_cumulative_belief_perspective)
            # percentiles_cumulative_belief_perspective_per_speaker = np.asarray(percentiles_cumulative_belief_perspective_per_speaker)
            #
            # print "percentiles_cumulative_belief_perspective_per_speaker[:, 1] is:"
            # print percentiles_cumulative_belief_perspective_per_speaker[:, 1]
            # print "percentiles_cumulative_belief_perspective_per_speaker[:, 1].shape is:"
            # print percentiles_cumulative_belief_perspective_per_speaker[:, 1].shape
            #
            #
            # percentiles_cumulative_belief_lexicon = measur.calc_cumulative_belief_in_correct_hyps_per_speaker(n_runs, n_contexts, multi_run_log_posterior_matrix, learner_hypothesis_space, perspective_hyps, lexicon_hyps, multi_run_perspectives_per_speaker_matrix, 0., multi_run_lexicons_per_speaker_matrix, 'lexicon')
            #
            # print "percentiles_cumulative_belief_lexicon[1] is:"
            # print percentiles_cumulative_belief_lexicon[1]
            # print "percentiles_cumulative_belief_lexicon[1].shape is:"
            # print percentiles_cumulative_belief_lexicon[1].shape
            #
            #
            # percentiles_cumulative_belief_composite = measur.calc_cumulative_belief_in_correct_hyps_per_speaker(n_runs, n_contexts, multi_run_log_posterior_matrix, learner_hypothesis_space, perspective_hyps, lexicon_hyps, multi_run_perspectives_per_speaker_matrix, 0., multi_run_lexicons_per_speaker_matrix, 'composite')
            #
            #
            # print "percentiles_cumulative_belief_composite[1] is:"
            # print percentiles_cumulative_belief_composite[1]
            # print "percentiles_cumulative_belief_composite[1].shape is:"
            # print percentiles_cumulative_belief_composite[1].shape


        if run_type == 'population_same_pop_dist_learner':
            #TODO: Do something here
            pass

        elif run_type == 'population_same_pop' or run_type == 'population_diff_pop':

            hypotheses_percentiles = measur.calc_hypotheses_percentiles(multi_run_log_posterior_matrix)
            # print 
            # print 
            # print "hypotheses_percentiles are:"
            # print hypotheses_percentiles
            # print "hypotheses_percentiles.shape is:"
            # print hypotheses_percentiles.shape

        # correct_hypothesis_index, mirror_hypothesis_index = find_correct_and_mirror_hyp_index(majority_lexicon)
        # # print 
        # # print 
        # # print "correct_hypothesis_index is:"
        # # print correct_hypothesis_index
        # # print "mirror_hypothesis_index is:"
        # # print mirror_hypothesis_index


        if run_type == 'population_same_pop_dist_learner':
            #TODO: Do something here
            pass

        elif run_type == 'population_same_pop' or run_type == 'population_diff_pop':
            # TODO: Do something here
            pass

        if run_type == 'population_same_pop_dist_learner':
            #TODO: Do something here
            pass

        elif run_type == 'population_same_pop' or run_type == 'population_diff_pop':

            population_lexicons = results_dict['population_lexicons']

            percentiles_posterior_pop_probs_approximation = measur.calc_posterior_pop_probs_match(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix, agent_type, pop_size, perspective_probs, lexicon_probs, population_lexicons, hypothesis_space, perspective_hyps, lexicon_hyps)
            # print 
            # print 
            # print "percentiles_posterior_pop_probs_approximation are:"
            # print percentiles_posterior_pop_probs_approximation
            # print "percentiles_posterior_pop_probs_approximation[0] are:"
            # print percentiles_posterior_pop_probs_approximation[0]


        calc_performance_measures_time = time.clock()-t1
        print 
        print 'calc_performance_measures_time is:'
        print str((calc_performance_measures_time/60))+" m"




    #############################################################################
    # Below the actual writing of the results to text and pickle files happens:


    t2 = time.clock()


    if run_type == 'population_diff_pop':
        if context_generation == 'random':
            file_title = 'diff_pop_size_'+str(pop_size)+'_'+str(speaker_order_type)+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+str(context_generation)+'_'+str(context_type)+'_contexts_'+'p_probs_'+perspective_probs_string+'_'+learner_type+'_lex_hyps_'+which_lexicon_hyps+'_lex_prior_'+lexicon_prior_type+'_learning_type_probs_'+learning_type_probs_string+'_p_prior_'+perspective_prior_type+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'

        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            file_title = 'diff_pop_size_'+str(pop_size)+'_'+str(speaker_order_type)+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(context_type)+'_contexts_'+'p_probs_'+perspective_probs_string+'_'+learner_type+'_lex_hyps_'+which_lexicon_hyps+'_lex_prior_'+lexicon_prior_type+'_learning_type_probs_'+learning_type_probs_string+'_p_prior_'+perspective_prior_type+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'

    elif run_type == 'population_same_pop':
        if context_generation == 'random':
            file_title = 'same_pop_size_'+str(pop_size)+'_'+str(speaker_order_type)+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+str(context_generation)+'_'+str(context_type)+'_contexts_'+'p_probs_'+perspective_probs_string+'_'+learner_type+'_lex_hyps_'+which_lexicon_hyps+'_lex_prior_'+lexicon_prior_type+'_learning_type_probs_'+learning_type_probs_string+'_p_prior_'+perspective_prior_type+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'

        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            file_title = 'same_pop_size_'+str(pop_size)+'_'+str(speaker_order_type)+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(context_type)+'_contexts_'+'p_probs_'+perspective_probs_string+'_'+learner_type+'_lex_hyps_'+which_lexicon_hyps+'_lex_prior_'+lexicon_prior_type+'_learning_type_probs_'+learning_type_probs_string+'_p_prior_'+perspective_prior_type+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'


    elif run_type == 'population_same_pop_dist_learner':
        if context_generation == 'random':
            file_title = 'same_pop_dist_size_'+str(pop_size)+'_'+str(speaker_order_type)+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+str(context_generation)+'_'+str(context_type)+'_contexts_'+'p_probs_'+perspective_probs_string+'_'+learner_type+'_lex_hyps_'+which_lexicon_hyps+'_lex_prior_'+lexicon_prior_type+'_learning_type_probs_'+learning_type_probs_string+'_p_prior_'+perspective_prior_type+'_'+perspective_prior_strength_string+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'

        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            file_title = 'same_pop_dist_size_'+str(pop_size)+'_'+str(speaker_order_type)+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(context_type)+'_contexts_'+'p_probs_'+perspective_probs_string+'_'+learner_type+'_lex_hyps_'+which_lexicon_hyps+'_lex_prior_'+lexicon_prior_type+'_learning_type_probs_'+learning_type_probs_string+'_p_prior_'+perspective_prior_type+'_'+perspective_prior_strength_string+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'



    results_pickle_file_title = pickle_file_directory+run_type_dir+'Results_'+file_title

    pickle.dump(results_dict, open(results_pickle_file_title+".p", "wb"))



    if run_type == 'population_same_pop_dist_learner':

        percentiles_belief_perspective_pickle_file_title = pickle_file_directory+run_type_dir+'Belief_Persp_'+file_title

        pickle.dump(percentiles_p_hyp_posterior_mass_correct_per_speaker, open(percentiles_belief_perspective_pickle_file_title+".p", "wb"))


        percentiles_belief_lexicon_pickle_file_title = pickle_file_directory+run_type_dir+'Belief_Lex_'+file_title

        pickle.dump(percentiles_lex_hyp_posterior_mass_correct, open(percentiles_belief_lexicon_pickle_file_title+".p", "wb"))


        percentiles_belief_composite_pickle_file_title = pickle_file_directory+run_type_dir+'Belief_Comp_'+file_title

        pickle.dump(percentiles_composite_hyp_posterior_mass_correct, open(percentiles_belief_composite_pickle_file_title+".p", "wb"))


        #
        # percentiles_cumulative_belief_perspective_pickle_file_title = pickle_file_directory+run_type_dir+'Cum_Belief_Persp_'+file_title
        #
        # saveresults.write_results_to_pickle_file(percentiles_cumulative_belief_perspective_pickle_file_title, percentiles_cumulative_belief_perspective_per_speaker)
        #
        # percentiles_cumulative_belief_lexicon_pickle_file_title = pickle_file_directory+run_type_dir+'Cum_Belief_Lex_'+file_title
        #
        # saveresults.write_results_to_pickle_file(percentiles_cumulative_belief_lexicon_pickle_file_title, percentiles_cumulative_belief_lexicon)
        #
        # percentiles_cumulative_belief_composite_pickle_file_title = pickle_file_directory+run_type_dir+'Cum_Belief_Comp_'+file_title
        #
        # saveresults.write_results_to_pickle_file(percentiles_cumulative_belief_composite_pickle_file_title, percentiles_cumulative_belief_composite)
        #
        #


    write_to_files_time = time.clock()-t2
    print
    print 'write_to_files_time is:'
    print str((write_to_files_time/60))+" m"




    #############################################################################
    # The code below makes the plots and saves them:


    t3 = time.clock()

    plot_file_path = plot_file_directory+run_type_dir
    plot_file_title = file_title



    if run_type == 'population_same_pop' or run_type == 'population_diff_pop':
        scores_plot_title = 'Posterior probability assigned to the correct (part) hypothesis over time'

        hypotheses_plot_title = 'Posterior probability assigned to each composite hypothesis over time'

        # plots.plot_timecourse_hypotheses_percentiles(hypotheses_plot_title, plot_file_title, ((n_contexts*n_utterances)+1), hypotheses_percentiles, correct_hypothesis_index, mirror_hypothesis_index)



        lex_heatmap_title = 'Heatmap of posterior probability distribution over m-s mappings'

        # plots.plot_lexicon_heatmap(lex_heatmap_title, plot_file_path, plot_file_title, lex_posterior_matrix)



        convergence_time_plot_title = 'No. of observations required to reach 1.0-theta posterior on correct hypothesis'

        # plots.plot_convergence_time_over_theta_range(convergence_time_plot_title, plot_file_path, plot_file_title, theta_range, percentiles_convergence_time_over_theta_composite, percentiles_convergence_time_over_theta_perspective, percentiles_convergence_time_over_theta_lexicon)
        #
        #


    elif run_type == 'population_same_pop_dist_learner':

        scores_plot_title = 'Posterior probability assigned to the correct (part) hypothesis over time'


        plots.plot_timecourse_scores_percentiles_with_speaker_distinction(scores_plot_title, plot_file_path, plot_file_title, ((n_contexts*n_utterances)+1), percentiles_composite_hyp_posterior_mass_correct, percentiles_p_hyp_posterior_mass_correct_per_speaker,  percentiles_lex_hyp_posterior_mass_correct)


        plots.plot_timecourse_scores_percentiles_without_error_median_with_speaker_distinction(scores_plot_title, plot_file_path, plot_file_title, ((n_contexts*n_utterances)+1), percentiles_composite_hyp_posterior_mass_correct, percentiles_p_hyp_posterior_mass_correct_per_speaker,  percentiles_lex_hyp_posterior_mass_correct)


        plots.plot_timecourse_scores_percentiles_without_error_mean_with_speaker_distinction(scores_plot_title, plot_file_path, plot_file_title, ((n_contexts*n_utterances)+1), percentiles_composite_hyp_posterior_mass_correct, percentiles_p_hyp_posterior_mass_correct_per_speaker,  percentiles_lex_hyp_posterior_mass_correct)




        cumulative_belief_plot_title = 'Cumulative belief in the correct composite hypothesis over time'


        # plots.plot_cum_belief_percentiles_with_speaker_distinction(cumulative_belief_plot_title, plot_file_path, plot_file_title, ((n_contexts*n_utterances)+1), percentiles_cumulative_belief_composite, percentiles_cumulative_belief_perspective_per_speaker,  percentiles_cumulative_belief_lexicon)
        #
        #
        # plots.plot_cum_belief_percentiles_without_error_median_with_speaker_distinction(cumulative_belief_plot_title, plot_file_path, plot_file_title, ((n_contexts*n_utterances)+1), percentiles_cumulative_belief_composite, percentiles_cumulative_belief_perspective_per_speaker,  percentiles_cumulative_belief_lexicon)
        #
        #
        # plots.plot_cum_belief_percentiles_without_error_mean_with_speaker_distinction(cumulative_belief_plot_title, plot_file_path, plot_file_title, ((n_contexts*n_utterances)+1), percentiles_cumulative_belief_composite, percentiles_cumulative_belief_perspective_per_speaker,  percentiles_cumulative_belief_lexicon)



    plotting_time = time.clock()-t3
    print
    print 'plotting_time is:'
    print str((plotting_time/60))+" m"


