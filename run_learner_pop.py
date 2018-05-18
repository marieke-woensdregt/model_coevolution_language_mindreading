__author__ = 'Marieke Woensdregt'


from params_learner_pop import *
from pop import create_speaker_order_single_generation
from pop import Population
from pop import DistinctionPopulation
from pop import Agent
from pop import DistinctionAgent
import context
import measur
from lex import Lexicon
import time
import saveresults
import plots
import itertools
import hypspace
import prior


#
# np.set_printoptions(threshold='nan')



def multi_runs_population_diff_pop():
    """
    :param: All parameters that this function takes are global variables specified in the params module
    :return: A list containing all the result arrays ('result_array_list') and a dictionary containing the keys of those arrays with the corresponding indices ('result_array_keys')
    """

    t0 = time.clock()

    # 1) First the matrices are created that will be saving the data_dict and the posteriors:
    multi_run_context_matrix = np.zeros((n_runs, n_contexts, n_meanings))
    multi_run_utterances_matrix = np.zeros((n_runs, n_contexts, n_utterances))
    multi_run_log_posterior_matrix = np.zeros((n_runs, ((n_contexts*n_utterances)+1), len(hypothesis_space)))

    # 2) Then the prior probability distribution for all agents is created:
    ## 2.1) First the perspective prior is created:
    perspective_prior = prior.create_perspective_prior(perspective_hyps, lexicon_hyps, perspective_prior_type, learner_perspective, perspective_prior_strength)
    ## 2.2) Then the lexicon prior is created:
    lexicon_prior = prior.create_lexicon_prior(lexicon_hyps, lexicon_prior_type, lexicon_prior_constant, error)

    for r in range(n_runs):
        if r % report_every_r == 0:
            print 'r = '+str(r)

        population = Population(pop_size, n_meanings, n_signals, hypothesis_space, perspective_hyps, lexicon_hyps, learner_perspective, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant, perspectives, perspective_probs, alpha, lexicon_types, lexicon_type_probs, error, pragmatic_level, optimality_alpha, n_contexts, context_type, context_generation, context_size, helpful_contexts, n_utterances, learning_types, learning_type_probs)

        perspective_prior_fixed = perspective_probs
        lexicon_prior_fixed = np.zeros(len(lexicon_hyps))
        for i in range(len(lexicon_hyps)):
            lexicon = lexicon_hyps[i]
            for j in range(len(population.lexicons)):
                if np.array_equal(lexicon, population.lexicons[j].lexicon):
                    lexicon_prior_fixed[i] = lexicon_type_probs[j]

        if context_generation == 'random':
            context_matrix = context.gen_context_matrix(context_type, n_meanings, context_size, n_contexts)
        elif context_generation == 'only_helpful':
            context_matrix = context.gen_helpful_context_matrix(n_meanings, n_contexts, helpful_contexts)
        elif context_generation == 'optimal':
            context_matrix = context.gen_helpful_context_matrix_fixed_order(n_meanings, n_contexts, helpful_contexts)

        # TODO: Figure out why it is that the composite_log_priors_population are not treated as a global variable but rather as a local variable of the learner object that gets updated as learner.log_posteriors get updated --> I think the solution might be to declare it 'global' somewhere in the Agent class

        learner_lexicon = Lexicon(learner_lex_type, n_meanings, n_signals)
        if learner_type == 'perspective_unknown':
            composite_log_priors = prior.list_composite_log_priors(agent_type, pop_size, hypothesis_space, perspective_hyps, lexicon_hyps, perspective_prior, lexicon_prior_fixed) # These have to be recalculated fresh after every run so that each new learner is initialized with the original prior distribution, rather than the final posteriors of the last learner!
        elif learner_type == 'lexicon_unknown':
            composite_log_priors = prior.list_composite_log_priors(agent_type, pop_size, hypothesis_space, perspective_hyps, lexicon_hyps, perspective_prior_fixed, lexicon_prior) # These have to be recalculated fresh after every run so that each new learner is initialized with the original prior distribution, rather than the final posteriors of the last learner!
        elif learner_type == 'both_unknown':
            composite_log_priors = prior.list_composite_log_priors(agent_type, pop_size, hypothesis_space, perspective_hyps, lexicon_hyps, perspective_prior, lexicon_prior) # These have to be recalculated fresh after every run so that each new learner is initialized with the original prior distribution, rather than the final posteriors of the last learner!

        learner = Agent(perspective_hyps, lexicon_hyps, composite_log_priors, composite_log_priors, learner_perspective, alpha, learner_lexicon, learner_learning_type)

        speaker_order = create_speaker_order_single_generation(population, speaker_order_type, n_contexts, first_input_stage_ratio)
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

    run_time_mins = (time.clock()-t0)/60.

    results_dict = {'multi_run_context_matrix':multi_run_context_matrix, 'multi_run_utterances_matrix':multi_run_utterances_matrix, 'multi_run_log_posterior_matrix':multi_run_log_posterior_matrix, 'majority_p_hyp_indices':majority_p_hyp_indices, 'majority_lex_hyp_indices':majority_lex_hyp_indices, 'majority_composite_hyp_indices':majority_composite_hyp_indices, 'majority_lexicon':majority_lexicon, 'population_lexicons':population.lexicons, 'run_time_mins':run_time_mins}
    return results_dict



def multi_runs_population_same_pop():
    """
    :param: All parameters that this function takes are global variables specified in the params module
    :return: A list containing all the result arrays ('result_array_list') and a dictionary containing the keys of those arrays with the corresponding indices ('result_array_keys')
    """

    # 1) First the matrices are created that will be saving the data_dict and the posteriors:
    multi_run_context_matrix = np.zeros((n_runs, n_contexts, n_meanings))
    multi_run_utterances_matrix = np.zeros((n_runs, n_contexts, n_utterances))
    multi_run_log_posterior_matrix = np.zeros((n_runs, ((n_contexts*n_utterances)+1), len(hypothesis_space)))

    # 2) Then the population is created:
    # FIXME: Currently the different runs of the simulation run with the same population, instead of having a new population being initialized with every run. This is because the majority lexicon is not logged for each run separately.



    population = Population(pop_size, n_meanings, n_signals, hypothesis_space, perspective_hyps, lexicon_hyps, learner_perspective, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant, perspectives, perspective_probs, alpha, lexicon_types, lexicon_type_probs, error, pragmatic_level, optimality_alpha, n_contexts, context_type, context_generation, context_size, helpful_contexts, n_utterances, learning_types, learning_type_probs)

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
                lexicon_prior_fixed[i] = lexicon_type_probs[j]
    for r in range(n_runs):
        print str(r)

        if context_generation == 'random':
            context_matrix = context.gen_context_matrix(context_type, n_meanings, context_size, n_contexts)
        elif context_generation == 'only_helpful':
            context_matrix = context.gen_helpful_context_matrix(n_meanings, n_contexts, helpful_contexts)
        elif context_generation == 'optimal':
            context_matrix = context.gen_helpful_context_matrix_fixed_order(n_meanings, n_contexts, helpful_contexts)

        # TODO: Figure out why it is that the composite_log_priors_population are not treated as a global variable but rather as a local variable of the learner object that gets updated as learner.log_posteriors get updated --> I think the solution might be to declare it 'global' somewhere in the Agent class

        learner_lexicon = Lexicon(learner_lex_type, n_meanings, n_signals)
        if learner_type == 'perspective_unknown':
            composite_log_priors = prior.list_composite_log_priors(agent_type, pop_size, hypothesis_space, perspective_prior, lexicon_prior_fixed) # These have to be recalculated fresh after every run so that each new learner is initialized with the original prior distribution, rather than the final posteriors of the last learner!
        elif learner_type == 'lexicon_unknown':
            composite_log_priors = prior.list_composite_log_priors(agent_type, pop_size, hypothesis_space, perspective_prior_fixed, lexicon_prior) # These have to be recalculated fresh after every run so that each new learner is initialized with the original prior distribution, rather than the final posteriors of the last learner!
        elif learner_type == 'both_unknown':
            composite_log_priors = prior.list_composite_log_priors(agent_type, pop_size, hypothesis_space, perspective_hyps, lexicon_hyps, perspective_prior, lexicon_prior) # These have to be recalculated fresh after every run so that each new learner is initialized with the original prior distribution, rather than the final posteriors of the last learner!

        learner = Agent(perspective_hyps, lexicon_hyps, composite_log_priors, composite_log_priors, learner_perspective, alpha, learner_lexicon, learner_learning_type)

        speaker_order = create_speaker_order_single_generation(population, speaker_order_type, n_contexts, first_input_stage_ratio)
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
def multi_runs_population_same_pop_distinction_learner(pop_size, n_runs, n_contexts, n_utterances, n_meanings, n_signals):
    """
    :param: All parameters that this function takes are global variables specified in the params module
    :return: A list containing all the result arrays ('result_array_list') and a dictionary containing the keys of those arrays with the corresponding indices ('result_array_keys')
    """

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
    composite_log_priors = prior.list_composite_log_priors_with_speaker_distinction(hypothesis_space, perspective_prior, lexicon_prior, pop_size)

    # 4) Then the population is created:
    ## 4.1) First the population's lexicons are determined:
    lexicons = []
    for i in range(len(lexicon_types)):
        lex_type = lexicon_types[i]
        lexicon = Lexicon(lex_type, n_meanings, n_signals)
        lexicon = lexicon
        lexicons.append(lexicon)

    #TODO: Note that lexicon selection is currently still random (based on the parameter 'lexicon_type_probs')
    lexicons_per_agent = np.random.choice(lexicons, pop_size, replace=True, p=lexicon_type_probs)

    if opposite_lexicons == 'yes' and lexicon_type_probs[0] == 1.:
        optimal_lexicon = Lexicon('optimal_lex', n_meanings, n_signals)
        mirror_lexicon = Lexicon('mirror_of_optimal_lex', n_meanings, n_signals)
        lexicons = [optimal_lexicon, mirror_lexicon]
        lexicons_per_agent = []
        i = 0
        while i < pop_size:
            for lexicon in lexicons:
                lexicons_per_agent.append(lexicon)
                i += 1
        lexicons_per_agent = np.asarray(lexicons_per_agent)

    elif opposite_lexicons == 'yes' and lexicon_type_probs[1] == 1.:
        first_ambiguous_lexicon = Lexicon('half_ambiguous_lex', n_meanings, n_signals)
        second_ambiguous_lexicon = Lexicon('mirror_of_ambiguous_lex', n_meanings, n_signals, first_ambiguous_lexicon)
        lexicons = [first_ambiguous_lexicon, second_ambiguous_lexicon]
        lexicons_per_agent = []
        i = 0
        while i < pop_size:
            for lexicon in lexicons:
                lexicons_per_agent.append(lexicon)
                i += 1
        lexicons_per_agent = np.asarray(lexicons_per_agent)

    ## 4.2) Then the population's perspectives are determined:
    if opposite_perspectives == 'yes':
        perspectives_per_agent = np.zeros(pop_size)
        i = 0
        perspectives_cycle = itertools.cycle(perspectives)
        while i < pop_size:
            # for perspective in perspectives:
            perspectives_per_agent[i] = next(perspectives_cycle)
            i += 1
    else:
        perspectives_per_agent = np.random.choice(perspectives, pop_size, replace=True, p=perspective_probs)

    ## 4.3) Then the population's learning types are determined:
    for i in range(pop_size):
        learning_types_per_agent = np.random.choice(learning_types, pop_size, replace=True, p=learning_type_probs)

    ## 4.4) Then the population itself is created:
    population = DistinctionPopulation(pop_size, n_meanings, n_signals, hypothesis_space, perspective_hyps, lexicon_hyps, learner_perspective, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant, composite_log_priors, perspectives, perspectives_per_agent, perspective_probs, alpha, lexicons, lexicons_per_agent, error, n_contexts, context_type, context_generation, context_size, helpful_contexts, n_utterances, learning_types, learning_types_per_agent, learning_type_probs)


    for r in range(n_runs):
        if r % report_every_r == 0:
            print 'r = '+str(r)

        speaker_order = create_speaker_order_single_generation(population, speaker_order_type, n_contexts, first_input_stage_ratio)

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
                    lexicon_prior_fixed[i] = lexicon_type_probs[j]

        if context_generation == 'random':
            context_matrix = context.gen_context_matrix(context_type, n_meanings, context_size, n_contexts)
        elif context_generation == 'only_helpful':
            context_matrix = context.gen_helpful_context_matrix(n_meanings, n_contexts, helpful_contexts)
        elif context_generation == 'optimal':
            context_matrix = context.gen_helpful_context_matrix_fixed_order(n_meanings, n_contexts, helpful_contexts)

        learner_lexicon = Lexicon(learner_lex_type, n_meanings, n_signals)
        if learner_type == 'perspective_unknown':
            composite_log_priors = prior.list_composite_log_priors_with_speaker_distinction(hypothesis_space, perspective_prior, lexicon_prior_fixed, pop_size)
            # These have to be recalculated fresh after every run so that each new learner is initialized with the original prior distribution, rather than the final posteriors of the last learner!
        elif learner_type == 'lexicon_unknown':
            composite_log_priors = prior.list_composite_log_priors_with_speaker_distinction(hypothesis_space, perspective_prior_fixed, lexicon_prior, pop_size)
            # These have to be recalculated fresh after every run so that each new learner is initialized with the original prior distribution, rather than the final posteriors of the last learner!
        elif learner_type == 'both_unknown':
            composite_log_priors = prior.list_composite_log_priors_with_speaker_distinction(hypothesis_space, perspective_prior, lexicon_prior, pop_size) # These have to be recalculated fresh after every run so that each new learner is initialized with the original prior distribution, rather than the final posteriors of the last learner!

        learner = DistinctionAgent(perspective_hyps, lexicon_hyps, composite_log_priors, composite_log_priors, learner_perspective, alpha, learner_lexicon, learner_learning_type, pop_size)

        if r == 0:
            print 
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
            results_dict = multi_runs_population_diff_pop()

        elif run_type == 'population_same_pop':
            results_dict = multi_runs_population_same_pop()

        elif run_type == 'population_same_pop_dist_learner':
            results_dict = multi_runs_population_same_pop_distinction_learner(pop_size, n_runs, n_contexts, n_utterances, n_meanings, n_signals)


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


        elif run_type == 'population_same_pop' or run_type == 'population_diff_pop':

            majority_p_hyp_indices = results_dict['majority_p_hyp_indices']

            majority_lex_hyp_indices = results_dict['majority_lex_hyp_indices']

            majority_composite_hyp_indices = results_dict['majority_composite_hyp_indices']

            print 
            print "majority_composite_hyp_indices are:"
            print majority_composite_hyp_indices


            min_convergence_time_perspective, mean_convergence_time_perspective, median_convergence_time_perspective, max_convergence_time_perspective = measur.calc_majority_hyp_convergence_time(multi_run_log_posterior_matrix, majority_p_hyp_indices, theta_fixed)
            # print 
            # print 
            # print "min_convergence_time_perspective is:"
            # print str(min_convergence_time_perspective)+' observations'
            # print "mean_convergence_time_perspective is:"
            # print str(mean_convergence_time_perspective)+' observations'
            # print "median_convergence_time_perspective is:"
            # print str(median_convergence_time_perspective)+' observations'
            # print "max_convergence_time_perspective is:"
            # print str(max_convergence_time_perspective)+' observations'

            min_convergence_time_lexicon, mean_convergence_time_lexicon, median_convergence_time_lexicon, max_convergence_time_lexicon = measur.calc_majority_hyp_convergence_time(multi_run_log_posterior_matrix, majority_lex_hyp_indices, theta_fixed)
            # print 
            # print 
            # print "min_convergence_time_lexicon is:"
            # print str(min_convergence_time_lexicon)+' observations'
            # print "mean_convergence_time_lexicon is:"
            # print str(mean_convergence_time_lexicon)+' observations'
            # print "median_convergence_time_lexicon is:"
            # print str(median_convergence_time_lexicon)+' observations'
            # print "max_convergence_time_lexicon is:"
            # print str(max_convergence_time_lexicon)+' observations'

            min_convergence_time_composite, mean_convergence_time_composite, median_convergence_time_composite, max_convergence_time_composite = measur.calc_majority_hyp_convergence_time(multi_run_log_posterior_matrix, majority_composite_hyp_indices, theta_fixed)
            # print 
            # print 
            # print "min_convergence_time_composite is:"
            # print str(min_convergence_time_composite)+' observations'
            # print "mean_convergence_time_composite is:"
            # print str(mean_convergence_time_composite)+' observations'
            # print "median_convergence_time_composite is:"
            # print str(median_convergence_time_composite)+' observations'
            # print "max_convergence_time_composite is:"
            # print str(max_convergence_time_composite)+' observations'


        if run_type == 'population_same_pop_dist_learner':
            #TODO: Do something here
            pass


        elif run_type == 'population_same_pop' or run_type == 'population_diff_pop':
            percentiles_convergence_time_over_theta_perspective = measur.calc_convergence_time_over_theta_range_percentiles(multi_run_log_posterior_matrix, majority_p_hyp_indices, theta_range)
            # print 
            # print 
            # print "percentiles_convergence_time_over_theta_perspective are:"
            # print percentiles_convergence_time_over_theta_perspective
            # print "percentiles_convergence_time_over_theta_perspective.shape are:"
            # print percentiles_convergence_time_over_theta_perspective.shape

            percentiles_convergence_time_over_theta_lexicon = measur.calc_convergence_time_over_theta_range_percentiles(multi_run_log_posterior_matrix, majority_lex_hyp_indices, theta_range)
            # print 
            # print 
            # print "percentiles_convergence_time_over_theta_lexicon are:"
            # print percentiles_convergence_time_over_theta_lexicon
            # print "percentiles_convergence_time_over_theta_lexicon.shape are:"
            # print percentiles_convergence_time_over_theta_lexicon.shape

            percentiles_convergence_time_over_theta_composite = measur.calc_convergence_time_over_theta_range_percentiles(multi_run_log_posterior_matrix, majority_composite_hyp_indices, theta_range)
            # print 
            # print 
            # print "percentiles_convergence_time_over_theta_composite are:"
            # print percentiles_convergence_time_over_theta_composite
            # print "percentiles_convergence_time_over_theta_composite.shape are:"
            # print percentiles_convergence_time_over_theta_composite.shape


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

            # print "real_speaker_perspectives are:"
            # print real_speaker_perspectives

            real_lexicon = multi_run_lexicons_per_speaker_matrix[0][0]

            # print "real_lexicon is:"
            # print real_lexicon


            correct_p_hyp_indices_per_speaker = []
            for speaker_id in range(pop_size):
                correct_p_hyp_indices = measur.find_correct_hyp_indices_with_speaker_distinction(learner_hypothesis_space, real_speaker_perspectives, speaker_id, real_lexicon, 'perspective')
                correct_p_hyp_indices_per_speaker.append(correct_p_hyp_indices)
                np.asarray(correct_p_hyp_indices_per_speaker)

            # print "correct_p_hyp_indices_per_speaker are:"
            # print correct_p_hyp_indices_per_speaker


            correct_lex_hyp_indices = measur.find_correct_hyp_indices(learner_hypothesis_space, perspective_hyps, lexicon_hyps, real_speaker_perspectives, real_lexicon, 'lexicon')

            # print "correct_lex_hyp_indices are:"
            # print correct_lex_hyp_indices


            correct_composite_hyp_indices = measur.find_correct_hyp_indices(learner_hypothesis_space, perspective_hyps, lexicon_hyps, real_speaker_perspectives, real_lexicon, 'composite')

            print "correct_composite_hyp_indices are:"
            print correct_composite_hyp_indices


            percentiles_p_hyp_posterior_mass_correct_per_speaker = []
            for speaker_id in range(pop_size):
                percentiles_p_hyp_posterior_mass_correct = measur.calc_hyp_correct_posterior_mass_percentiles(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix, correct_p_hyp_indices_per_speaker[speaker_id])
                percentiles_p_hyp_posterior_mass_correct_per_speaker.append(percentiles_p_hyp_posterior_mass_correct)
            percentiles_p_hyp_posterior_mass_correct_per_speaker = np.asarray(percentiles_p_hyp_posterior_mass_correct_per_speaker)

            # print "percentiles_p_hyp_posterior_mass_correct_per_speaker is:"
            # print percentiles_p_hyp_posterior_mass_correct_per_speaker
            # print "percentiles_p_hyp_posterior_mass_correct_per_speaker.shape is:"
            # print percentiles_p_hyp_posterior_mass_correct_per_speaker.shape


            percentiles_lex_hyp_posterior_mass_correct = measur.calc_hyp_correct_posterior_mass_percentiles(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix, correct_lex_hyp_indices)

            # print "percentiles_lex_hyp_posterior_mass_correct is:"
            # print percentiles_lex_hyp_posterior_mass_correct
            # print "percentiles_lex_hyp_posterior_mass_correct.shape is:"
            # print percentiles_lex_hyp_posterior_mass_correct.shape

            percentiles_composite_hyp_posterior_mass_correct = measur.calc_hyp_correct_posterior_mass_percentiles(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix, correct_composite_hyp_indices)

            # print "percentiles_composite_hyp_posterior_mass_correct is:"
            # print percentiles_composite_hyp_posterior_mass_correct
            # print "percentiles_composite_hyp_posterior_mass_correct.shape is:"
            # print percentiles_composite_hyp_posterior_mass_correct.shape


            percentiles_cumulative_belief_perspective_per_speaker = []
            for speaker_id in range(pop_size):
                percentiles_cumulative_belief_perspective = measur.calc_cumulative_belief_in_correct_hyps_per_speaker(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix, learner_hypothesis_space, multi_run_perspectives_per_speaker_matrix, speaker_id, multi_run_lexicons_per_speaker_matrix, 'perspective')
                percentiles_cumulative_belief_perspective_per_speaker.append(percentiles_cumulative_belief_perspective)
            percentiles_cumulative_belief_perspective_per_speaker = np.asarray(percentiles_cumulative_belief_perspective_per_speaker)

            # print "percentiles_cumulative_belief_perspective_per_speaker is:"
            # print percentiles_cumulative_belief_perspective_per_speaker
            # print "percentiles_cumulative_belief_perspective_per_speaker.shape is:"
            # print percentiles_cumulative_belief_perspective_per_speaker.shape
            # print "np.mean(percentiles_cumulative_belief_perspective_per_speaker[0]) is:"
            # print np.mean(percentiles_cumulative_belief_perspective_per_speaker[0])


            percentiles_cumulative_belief_lexicon = measur.calc_cumulative_belief_in_correct_hyps_per_speaker(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix, learner_hypothesis_space, multi_run_perspectives_per_speaker_matrix, 0., multi_run_lexicons_per_speaker_matrix, 'lexicon')

            # print "percentiles_cumulative_belief_lexicon is:"
            # print percentiles_cumulative_belief_lexicon
            # print "percentiles_cumulative_belief_lexicon.shape is:"
            # print percentiles_cumulative_belief_lexicon.shape
            # print "np.mean(percentiles_cumulative_belief_lexicon) is:"
            # print np.mean(percentiles_cumulative_belief_lexicon)


            percentiles_cumulative_belief_composite = measur.calc_cumulative_belief_in_correct_hyps_per_speaker(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix, learner_hypothesis_space, multi_run_perspectives_per_speaker_matrix, 0., multi_run_lexicons_per_speaker_matrix, 'composite')

            # print "percentiles_cumulative_belief_composite is:"
            # print percentiles_cumulative_belief_composite
            # print "percentiles_cumulative_belief_composite.shape is:"
            # print percentiles_cumulative_belief_composite.shape
            # print "np.mean(percentiles_cumulative_belief_composite) is:"
            # print np.mean(percentiles_cumulative_belief_composite)


        elif run_type == 'population_same_pop' or run_type == 'population_diff_pop':

            percentiles_p_hyp_posterior_mass_correct = measur.calc_hyp_correct_posterior_mass_percentiles(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix, majority_p_hyp_indices)
            # print 
            # print 
            # print "percentiles_p_hyp_posterior_mass_correct is:"
            # print percentiles_p_hyp_posterior_mass_correct
            # print "percentiles_p_hyp_posterior_mass_correct[0].shape is:"
            # print percentiles_p_hyp_posterior_mass_correct[0].shape

            percentiles_lex_hyp_posterior_mass_correct = measur.calc_hyp_correct_posterior_mass_percentiles(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix, majority_lex_hyp_indices)
            # print 
            # print 
            # print "percentiles_lex_hyp_posterior_mass_correct is:"
            # print percentiles_lex_hyp_posterior_mass_correct
            # print "percentiles_lex_hyp_posterior_mass_correct[0].shape is:"
            # print percentiles_lex_hyp_posterior_mass_correct[0].shape

            percentiles_composite_hyp_posterior_mass_correct = measur.calc_hyp_correct_posterior_mass_percentiles(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix, majority_composite_hyp_indices)
            # print 
            # print 
            # print "percentiles_composite_hyp_posterior_mass_correct is:"
            # print percentiles_composite_hyp_posterior_mass_correct
            # print "percentiles_composite_hyp_posterior_mass_correct[0].shape is:"
            # print percentiles_composite_hyp_posterior_mass_correct[0].shape


        if run_type == 'population_same_pop_dist_learner':
            #TODO: Do something here
            pass

        elif run_type == 'population_same_pop' or run_type == 'population_diff_pop':

            majority_lexicon = results_dict['majority_lexicon']

            percentiles_lex_approximation_posterior_mass = measur.calc_lex_approximation_posterior_mass(multi_run_log_posterior_matrix, majority_lexicon, lexicon_hyps)
            # print 
            # print 
            # print "percentiles_lex_approximation_posterior_mass is:"
            # print percentiles_lex_approximation_posterior_mass
            # print "percentiles_lex_approximation_posterior_mass[0].shape is:"
            # print percentiles_lex_approximation_posterior_mass[0].shape


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

            percentiles_posterior_pop_probs_approximation = measur.calc_posterior_pop_probs_match(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix, agent_type, pop_size, perspective_probs, lexicon_type_probs, population_lexicons, hypothesis_space, perspective_hyps, lexicon_hyps)
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


    if n_runs > 0:
        t2 = time.clock()

        run_type_dir = 'Learner_Pop'

        if run_type == 'population_diff_pop':
            if context_generation == 'random':
                file_title = 'diff_pop_size_'+str(pop_size)+'_'+str(speaker_order_type)+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+str(context_generation)+'_'+str(context_type)+'_contexts_'+'lex_type_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_'+learner_type+'_lex_hyps_'+which_lexicon_hyps+'_lex_prior_'+lexicon_prior_type+'_opp_L_'+str(opposite_lexicons[0])+'_opp_P_'+str(opposite_perspectives[0])+'_learning_type_probs_'+learning_type_probs_string+'_p_prior_'+perspective_prior_type+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'

            elif context_generation == 'only_helpful' or context_generation == 'optimal':
                file_title = 'diff_pop_size_'+str(pop_size)+'_'+str(speaker_order_type)+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(context_type)+'_contexts_'+'lex_type_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_'+learner_type+'_lex_hyps_'+which_lexicon_hyps+'_lex_prior_'+lexicon_prior_type+'_opp_L_'+str(opposite_lexicons[0])+'_opp_P_'+str(opposite_perspectives[0])+'_learning_type_probs_'+learning_type_probs_string+'_p_prior_'+perspective_prior_type+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'

        elif run_type == 'population_same_pop':
            if context_generation == 'random':
                file_title = 'same_pop_size_'+str(pop_size)+'_'+str(speaker_order_type)+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+str(context_generation)+'_'+str(context_type)+'_contexts_'+'lex_type_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_'+learner_type+'_lex_hyps_'+which_lexicon_hyps+'_lex_prior_'+lexicon_prior_type+'_opp_L_'+str(opposite_lexicons[0])+'_opp_P_'+str(opposite_perspectives[0])+'_learning_type_probs_'+learning_type_probs_string+'_p_prior_'+perspective_prior_type+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'

            elif context_generation == 'only_helpful' or context_generation == 'optimal':
                file_title = 'same_pop_size_'+str(pop_size)+'_'+str(speaker_order_type)+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(context_type)+'_contexts_'+'lex_type_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_'+learner_type+'_lex_hyps_'+which_lexicon_hyps+'_lex_prior_'+lexicon_prior_type+'_opp_L_'+str(opposite_lexicons[0])+'_opp_P_'+str(opposite_perspectives[0])+'_learning_type_probs_'+learning_type_probs_string+'_p_prior_'+perspective_prior_type+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'


        elif run_type == 'population_same_pop_dist_learner':
            if context_generation == 'random':
                file_title = 'same_pop_dist_size_'+str(pop_size)+'_'+str(speaker_order_type)+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+str(context_generation)+'_'+str(context_type)+'_contexts_'+'lex_type_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_'+learner_type+'_lex_hyps_'+which_lexicon_hyps+'_lex_prior_'+lexicon_prior_type+'_opp_L_'+str(opposite_lexicons[0])+'_opp_P_'+str(opposite_perspectives[0])+'_learning_type_probs_'+learning_type_probs_string+'_p_prior_'+perspective_prior_type+'_'+perspective_prior_strength_string+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'

            elif context_generation == 'only_helpful' or context_generation == 'optimal':
                file_title = 'same_pop_dist_size_'+str(pop_size)+'_'+str(speaker_order_type)+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(context_type)+'_contexts_'+'lex_type_probs_'+lexicon_type_probs_string+'_p_probs_'+perspective_probs_string+'_'+learner_type+'_lex_hyps_'+which_lexicon_hyps+'_lex_prior_'+lexicon_prior_type+'_opp_L_'+str(opposite_lexicons[0])+'_opp_P_'+str(opposite_perspectives[0])+'_learning_type_probs_'+learning_type_probs_string+'_p_prior_'+perspective_prior_type+'_'+perspective_prior_strength_string+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U'



        results_pickle_file_title = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/'+run_type_dir+'/'+'Results_'+file_title

        saveresults.write_results_to_pickle_file(results_pickle_file_title, results_dict)


        if run_type == 'population_same_pop_dist_learner':

            percentiles_belief_perspective_pickle_file_title = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/'+run_type_dir+'/'+'Belief_Persp_'+file_title

            saveresults.write_results_to_pickle_file(percentiles_belief_perspective_pickle_file_title, percentiles_p_hyp_posterior_mass_correct_per_speaker)

            percentiles_belief_lexicon_pickle_file_title = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/'+run_type_dir+'/'+'Belief_Lex_'+file_title

            saveresults.write_results_to_pickle_file(percentiles_belief_lexicon_pickle_file_title, percentiles_lex_hyp_posterior_mass_correct)

            percentiles_belief_composite_pickle_file_title = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/'+run_type_dir+'/'+'Belief_Comp_'+file_title

            saveresults.write_results_to_pickle_file(percentiles_belief_composite_pickle_file_title, percentiles_composite_hyp_posterior_mass_correct)



            percentiles_cumulative_belief_perspective_pickle_file_title = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/'+run_type_dir+'/'+'Cum_Belief_Persp_'+file_title

            saveresults.write_results_to_pickle_file(percentiles_cumulative_belief_perspective_pickle_file_title, percentiles_cumulative_belief_perspective_per_speaker)

            percentiles_cumulative_belief_lexicon_pickle_file_title = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/'+run_type_dir+'/'+'Cum_Belief_Lex_'+file_title

            saveresults.write_results_to_pickle_file(percentiles_cumulative_belief_lexicon_pickle_file_title, percentiles_cumulative_belief_lexicon)

            percentiles_cumulative_belief_composite_pickle_file_title = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/'+run_type_dir+'/'+'Cum_Belief_Comp_'+file_title

            saveresults.write_results_to_pickle_file(percentiles_cumulative_belief_composite_pickle_file_title, percentiles_cumulative_belief_composite)




        write_to_files_time = time.clock()-t2
        print 
        print 'write_to_files_time is:'
        print str((write_to_files_time/60))+" m"




    #############################################################################
    # The code below makes the plots and saves them:


    if n_runs > 0:
        t3 = time.clock()

        plot_file_path = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Plots/'+run_type_dir
        plot_file_title = file_title



        if run_type == 'population_same_pop' or run_type == 'population_diff_pop':
            scores_plot_title = 'Posterior probability assigned to the correct (part) hypothesis over time'

            plots.plot_timecourse_scores_percentiles(scores_plot_title, plot_file_path, plot_file_title, ((n_contexts*n_utterances)+1), percentiles_composite_hyp_posterior_mass_correct, percentiles_p_hyp_posterior_mass_correct, percentiles_lex_hyp_posterior_mass_correct)


            plots.plot_timecourse_scores_percentiles_without_error_median(scores_plot_title, plot_file_path, plot_file_title, ((n_contexts*n_utterances)+1), percentiles_composite_hyp_posterior_mass_correct, percentiles_p_hyp_posterior_mass_correct, percentiles_lex_hyp_posterior_mass_correct)


            plots.plot_timecourse_scores_percentiles_without_error_mean(scores_plot_title, plot_file_path, plot_file_title, ((n_contexts*n_utterances)+1), percentiles_composite_hyp_posterior_mass_correct, percentiles_p_hyp_posterior_mass_correct, percentiles_lex_hyp_posterior_mass_correct)

            hypotheses_plot_title = 'Posterior probability assigned to each composite hypothesis over time'

            # plots.plot_timecourse_hypotheses_percentiles(hypotheses_plot_title, plot_file_title, ((n_contexts*n_utterances)+1), hypotheses_percentiles, correct_hypothesis_index, mirror_hypothesis_index)



            lex_heatmap_title = 'Heatmap of posterior probability distribution over m-s mappings'

            # plots.plot_lexicon_heatmap(lex_heatmap_title, plot_file_path, plot_file_title, lex_posterior_matrix)
            #


            convergence_time_plot_title = 'No. of observations required to reach 1.0-theta posterior on correct hypothesis'

            # plots.plot_convergence_time_over_theta_range(convergence_time_plot_title, plot_file_path, plot_file_title, theta_range, percentiles_convergence_time_over_theta_composite, percentiles_convergence_time_over_theta_perspective, percentiles_convergence_time_over_theta_lexicon)
            #
            #


        elif run_type == 'population_same_pop_dist_learner':

            scores_plot_title = 'Posterior probability assigned to the correct (part) hypothesis over time'


            plots.plot_timecourse_scores_percentiles_with_speaker_distinction(scores_plot_title, plot_file_path, plot_file_title, ((n_contexts*n_utterances)+1), percentiles_composite_hyp_posterior_mass_correct, percentiles_p_hyp_posterior_mass_correct_per_speaker,  percentiles_lex_hyp_posterior_mass_correct)


            # plots.plot_timecourse_scores_percentiles_without_error_median_with_speaker_distinction(scores_plot_title, plot_file_path, plot_file_title, ((n_contexts*n_utterances)+1), percentiles_composite_hyp_posterior_mass_correct, percentiles_p_hyp_posterior_mass_correct_per_speaker,  percentiles_lex_hyp_posterior_mass_correct)
            #
            #
            # plots.plot_timecourse_scores_percentiles_without_error_mean_with_speaker_distinction(scores_plot_title, plot_file_path, plot_file_title, ((n_contexts*n_utterances)+1), percentiles_composite_hyp_posterior_mass_correct, percentiles_p_hyp_posterior_mass_correct_per_speaker,  percentiles_lex_hyp_posterior_mass_correct)




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


