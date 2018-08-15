__author__ = 'Marieke Woensdregt'


import numpy as np
import time

import context
import hypspace
import lex
import measur
from pop import Agent, PragmaticAgent
import prior
import saveresults


# np.set_printoptions(threshold='nan')


#######################################################################################################################
# STEP 1: THE PARAMETERS:


# 1.1: The parameters defining the lexicon size (and thus the number of meanings in the world):

n_meanings = 3  # The number of meanings
n_signals = 3  # The number of signals



# 1.2: The parameters defining the contexts and how they map to the agent's saliencies:

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



error = 0.05 # The error term on production
error_string = saveresults.convert_array_to_string(error)
extra_error = True # Determines whether the error specified above gets added on top AFTER the pragmatic speaker has calculated its production probabilities by maximising the utility to the listener.




# 1.3: The parameters that determine the make-up of an individual speaker (for the dyadic condition):

speaker_perspective = 1. # The speaker's perspective
speaker_lex_type = 'specified_lexicon' # The lexicon type of the speaker. This can be set to either 'optimal_lex', 'half_ambiguous_lex', 'fully_ambiguous_lex' or 'specified_lexicon'
speaker_lex_index = 0
speaker_learning_type = 'sample' #FIXME: The speaker has to be initiated with a learning type because I have not yet coded up a subclass of Agent that is only Speaker (for which things like hypothesis space, prior distributiona and learning type would not have to be specified).


# 1.4: The parameters that determine the attributes of the learner:

learner_perspective = 0. # The learner's perspective
learner_lex_type = 'empty_lex' # The lexicon type of the learner. This will normally be 'empty_lex'
learner_learning_type = 'sample' # The type of learning that the learner does. This can be set to either 'map' or 'sample'


pragmatic_level_speaker = 'literal'  # This can be set to either 'literal', 'perspective-taking' or 'prag'
optimality_alpha_speaker = 1.0  # Goodman & Stuhlmuller (2013) fitted sal_alpha = 3.4 to participant data with 4x3 lexicon of three number words ('one', 'two', and 'three') that could be mapped to four different world states (0, 1, 2, and 3)


pragmatic_level_learner = 'perspective-taking'  # This can be set to either 'literal', 'perspective-taking' or 'prag'
optimality_alpha_learner = 1.0  # Goodman & Stuhlmuller (2013) fitted sal_alpha = 3.4 to participant data with 4x3 lexicon of three number words ('one', 'two', and 'three') that could be mapped to four different world states (0, 1, 2, and 3)

pragmatic_level_sp_hyp_lr = 'literal'  # The assumption that the learner has about the speaker's pragmatic level


# 1.5: The parameters that determine the learner's hypothesis space:

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



# 1.6: The parameters that determine the learner's prior:

learner_type = 'both_unknown' # This can be set to either 'perspective_unknown', 'lexicon_unknown' or 'both_unknown'

perspective_prior_type = 'egocentric' # This can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
perspective_prior_strength = 0.9 # The strength of the egocentric prior (only used if the perspective_prior_type is set to 'egocentric')
perspective_prior_strength_string = saveresults.convert_array_to_string(perspective_prior_strength)



lexicon_prior_type = 'neutral' # This can be set to either 'neutral', 'ambiguous_fixed', 'half_ambiguous_fixed', 'expressivity_bias' or 'compressibility_bias'
lexicon_prior_constant = 0.0 # Determines the strength of the lexicon bias, with small c creating a STRONG prior and large c creating a WEAK prior. (And with c = 1000 creating an almost uniform prior.)
lexicon_prior_constant_string = saveresults.convert_array_to_string(lexicon_prior_constant)



# 1.7: The parameters that determine the amount of data_dict that the learner gets to see, and the amount of runs of the simulation:

n_utterances = 1  # This parameter determines how many signals the learner gets to observe in each context
n_contexts = 120  # The number of contexts that the learner gets to see. If context_generation = 'optimal' n_contexts for n_meanings = 2 can be anything in the table of 4 (e.g. 20, 40, 60, 80, 100, etc.); for n_meanings = 3 anything in the table of 12 (e.g. 12, 36, 60, 84, 108, etc.); and for n_meanings = 4 anything in the table of 48 (e.g. 48, 96, 144, etc.).


# 1.8: The parameters that determine how learning is measured:

theta_fixed = 0.01  # The learner is considered to have acquired the correct hypothesis if the posterior probability for that hypothesis exceeds (1.-theta_fixed)
## The parameters below serve to make a 'convergence time over theta' plot, where the amount of learning trials needed to reach a certain level of convergence on the correct hypothesis is plotted against different values of theta.
theta_start = 0.0
theta_stop = 1.0
theta_step = 0.00001
theta_range = np.arange(theta_start, theta_stop, theta_step)
theta_step_string = str(theta_step)
theta_step_string = theta_step_string.replace(".", "")



# 1.9: The parameters that determine the type and number of simulations that are run:

#FIXME: In the current implementation 'half_ambiguous_lex' can have different instantiations. Therefore, if we run a simulation where there are different lexicon_type_probs, we will want the run_type to be 'population_same_pop' to make sure that all the 'half ambiguous' speakers do have the SAME half ambiguous lexicon.
run_type = 'dyadic' # This parameter determines whether the learner communicates with only one speaker ('dyadic') or with a population ('population_diff_pop' if there are no speakers with the 'half_ambiguous' lexicon type, 'population_same_pop' if there are, 'population_same_pop_dist_learner' if the learner can distinguish between different speakers), or whether we do an iterated learning model ('iter')
n_runs = 2  # The number of runs of the simulation
report_every_r = 1

which_hyps_on_graph = 'all_hyps' # This is used in the plot_post_probs_over_lex_hyps() function, and can be set to either 'all_hyps' or 'lex_hyps_only'
x_axis_steps = 12  # This determines where the xticks are placed on the plot.

lex_measure = 'ca'  # This can be set to either 'mi' for mutual information or 'ca' for communicative accuracy (of the lexicon with itself)

#######################################################################################################################





#######################################################################################################################


def multi_runs_dyadic(n_meanings, n_signals, n_runs, n_contexts, n_utterances, context_generation, context_type, context_size, helpful_contexts, speaker_lex_type, speaker_lex_index, error, extra_error, pragmatic_level_speaker, optimality_alpha_speaker, pragmatic_level_learner, optimality_alpha_learner, speaker_perspective, sal_alpha, speaker_learning_type, learner_perspective, learner_lex_type, learner_learning_type, pragmatic_level_sp_hyp_lr, hypothesis_space, perspective_hyps, lexicon_hyps, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant):
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
    :param speaker_lex_type: lexicon type of the speaker. This can be set to either 'optimal_lex', 'half_ambiguous_lex', 'fully_ambiguous_lex' or 'specified_lexicon'
    :param speaker_lex_index: this parameter is only used if the parameter speaker_lex_type is set to 'specified_lexicon'
    :param error: float specifying the probability that the speaker makes a production error (i.e. randomly chooses a signal that isn't associated with the intended referent)
    :param extra_error: can be set to either True or False. Determines whether the error specified above gets added on top AFTER the pragmatic speaker has calculated its production probabilities by maximising the utility to the listener.
    :param pragmatic_level_speaker: can be set to either 'literal', 'perspective-taking' or 'prag'
    :param optimality_alpha_speaker: optimality parameter in the RSA model for pragmatic speakers. Only used if the pragmatic_level_speaker parameter is set to 'prag'
    :param pragmatic_level_learner: can be set to either 'literal', 'perspective-taking' or 'prag'
    :param optimality_alpha_learner: optimality parameter in the RSA model for pragmatic speakers. Only used if the pragmatic_level_learner parameter is set to 'prag'
    :param speaker_perspective: float specifying the perspective of the speaker (any float between 0.0 and 1.0, but make sure this aligns with the 'perspective_hyps' parameter below!)
    :param sal_alpha: float. Exponent that is used by the saliency function. sal_alpha=1 gives a directly proportional relationship between meaning distances and meaning saliencies, with sal_alpha>1 the saliency of meanings goes down exponentially as their distance from the agent increases (and with sal_alpha<1 the other way around)
    :param speaker_learning_type: can be set to either 'sample' for sampling from the posterior or 'map' for selecting only the maximum a posteriori hypothesis #FIXME: The speaker has to be initiated with a learning type because I have not yet coded up a subclass of Agent that is only Speaker (for which things like hypothesis space, prior distributiona and learning type would not have to be specified).
    :param learner_perspective:  float specifying the perspective of the learner (any float between 0.0 and 1.0)
    :param learner_lex_type: lexicon type of the learner. #FIXME: The learner has to be initiated with a lexicon type because I have not yet coded up a subclass of Agent that is only a Learner (for which the lexicon should not have to be specified in advance).
    :param learner_learning_type: can be set to either 'sample' for sampling from the posterior or 'map' for selecting only the maximum a posteriori hypothesis
    :param pragmatic_level_sp_hyp_lr: assumption that the learner has about the speaker's pragmatic level. Can be set to either 'literal', 'perspective-taking' or 'prag'
    :param hypothesis_space: full space of composite hypotheses that the learner will consider (2D numpy matrix with composite hypotheses on the rows, perspective hypotheses on column 0 and lexicon hypotheses on column 1)
    :param perspective_hyps: the perspective hypotheses that the learner will consider (1D numpy array)
    :param lexicon_hyps: the lexicon hypotheses that the learner will consider (1D numpy array)
    :param perspective_prior_type: can be set to either 'neutral', 'egocentric', 'same_as_lexicon' or 'zero_order_tom'
    :param perspective_prior_strength: strength of the egocentric prior (only used if the perspective_prior_type parameter is set to 'egocentric')
    :param lexicon_prior_type: can be set to either 'neutral', 'ambiguous_fixed', 'half_ambiguous_fixed', 'expressivity_bias' or 'compressibility_bias'
    :param lexicon_prior_constant: determines the strength of the lexicon bias, with small c creating a STRONG prior and large c creating a WEAK prior. (And with c = 1000 creating an almost uniform prior.)
    :return: A dictionary containing:
    1) 'multi_run_context_matrix':multi_run_context_matrix,
    2) 'multi_run_utterances_matrix':multi_run_utterances_matrix,
    3) 'multi_run_log_posterior_matrix':multi_run_log_posterior_matrix,
    4) 'correct_p_hyp_indices':correct_p_hyp_indices,
    5) 'correct_lex_hyp_indices':correct_lex_hyp_indices,
    6) 'correct_composite_hyp_indices':correct_composite_hyp_indices,
    7) 'speaker_lexicon':speaker.lexicon.lexicon,
    8) 'run_time_mins':run_time_mins
    """

    t0 = time.clock()

    multi_run_context_matrix = np.zeros((n_runs, n_contexts, n_meanings))
    multi_run_utterances_matrix = np.zeros((n_runs, n_contexts, n_utterances))
    multi_run_log_posterior_matrix = np.zeros((n_runs, ((n_contexts*n_utterances)+1), len(hypothesis_space)))


    speaker_lexicon_matrix = lexicon_hyps[speaker_lex_index]

    speaker_lexicon = lex.Lexicon(speaker_lex_type, n_meanings, n_signals, specified_lexicon=speaker_lexicon_matrix)



    print "This simulation will contain:"
    print str(n_runs)+' runs'
    print 'with'
    print str(n_contexts)+' contexts'
    print 'of type'
    print context_generation
    print 'The speakers perspective is:'
    print speaker_perspective
    print 'The speakers lexicon index is:'
    print speaker_lex_index
    print 'The speakers lexicon is:'
    speaker_lexicon.print_lexicon()
    print 'The perspective_prior is:'
    print perspective_prior_type
    print 'With strength:'
    print perspective_prior_strength
    print 'The lexicon prior is:'
    print lexicon_prior_type
    print 'With strength:'
    print lexicon_prior_constant
    print "pragmatic_level speaker is:"
    print pragmatic_level_speaker
    if pragmatic_level_speaker == 'prag':
        print "optimality_alpha_speaker is:"
        print optimality_alpha_speaker
    print "pragmatic_level learner is:"
    print pragmatic_level_learner
    if pragmatic_level_learner == 'prag':
        print "optimality_alpha_learner is:"
        print optimality_alpha_learner


    for r in range(n_runs):

        if context_generation == 'random':
            context_matrix = context.gen_context_matrix(context_type, n_meanings, context_size, n_contexts)
        elif context_generation == 'only_helpful':
            context_matrix = context.gen_helpful_context_matrix(n_meanings, n_contexts, helpful_contexts)
        elif context_generation == 'optimal':
            context_matrix = context.gen_helpful_context_matrix_fixed_order(n_meanings, n_contexts, helpful_contexts)


        # if r == 0:
        #     print ''
        #     print 'contexts are:'
        #     print context_matrix[0:len(helpful_contexts)]

        # TODO: Figure out why it is that the composite_log_priors_population are not treated as a global variable but rather as a local variable of the learner object that gets updated as learner.log_posteriors get updated --> I think the solution might be to declare it 'global' somewhere in the Agent class
        perspective_prior = prior.create_perspective_prior(perspective_hyps, lexicon_hyps, perspective_prior_type, learner_perspective, perspective_prior_strength)
        lexicon_prior = prior.create_lexicon_prior(lexicon_hyps, lexicon_prior_type, lexicon_prior_constant, error)
        composite_log_priors = prior.list_composite_log_priors('no_p_distinction', 1, hypothesis_space, perspective_hyps, lexicon_hyps, perspective_prior, lexicon_prior) # These have to be recalculated fresh after every run so that each new learner is initialized with the original prior distribution, rather than the final posteriors of the last learner!

        learner_lexicon = lex.Lexicon(learner_lex_type, n_meanings, n_signals)

        if pragmatic_level_learner == 'literal':
            learner = Agent(perspective_hyps, lexicon_hyps, composite_log_priors, composite_log_priors, learner_perspective, sal_alpha, learner_lexicon, learner_learning_type)
        elif pragmatic_level_learner == 'perspective-taking':
            learner = PragmaticAgent(perspective_hyps, lexicon_hyps, composite_log_priors, composite_log_priors, learner_perspective, sal_alpha, learner_lexicon, learner_learning_type, pragmatic_level_sp_hyp_lr, pragmatic_level_learner, optimality_alpha_learner, extra_error)
        elif pragmatic_level_learner == 'prag':
            learner = PragmaticAgent(perspective_hyps, lexicon_hyps, composite_log_priors, composite_log_priors, learner_perspective, sal_alpha, learner_lexicon, learner_learning_type, pragmatic_level_sp_hyp_lr, pragmatic_level_learner, optimality_alpha_learner, extra_error)


        if r == 0:
            # print "The learner's prior distribution is:"
            # print np.exp(learner.log_priors)
            print "The number of hypotheses is:"
            print len(learner.log_priors)

        if pragmatic_level_speaker == 'literal':
            speaker = Agent(perspective_hyps, lexicon_hyps, composite_log_priors, composite_log_priors, speaker_perspective, sal_alpha, speaker_lexicon, speaker_learning_type)
        elif pragmatic_level_speaker == 'perspective-taking':
            speaker = PragmaticAgent(perspective_hyps, lexicon_hyps, composite_log_priors, composite_log_priors, speaker_perspective, sal_alpha, speaker_lexicon, speaker_learning_type, 'literal', 'perspective-taking', optimality_alpha_speaker, extra_error)
        elif pragmatic_level_speaker == 'prag':
            speaker = PragmaticAgent(perspective_hyps, lexicon_hyps, composite_log_priors, composite_log_priors, speaker_perspective, sal_alpha, speaker_lexicon, speaker_learning_type, 'prag', 'prag', optimality_alpha_speaker, extra_error)

        data = speaker.produce_data(n_meanings, n_signals, context_matrix, n_utterances, error, extra_error)
        log_posteriors_per_data_point_matrix = learner.inference(n_contexts, n_utterances, data, error)
        correct_p_hyp_indices = measur.find_correct_hyp_indices(hypothesis_space, perspective_hyps, lexicon_hyps, speaker.perspective, speaker.lexicon.lexicon, 'perspective')


        # print
        # print 'data is:'
        # print data.print_data()



        # FIXME: If I want the half_ambiguous lexicon to be generated with the ambiguous mappings chosen at random, I have to make sure that the majority_lex_hyp_indices and majority_composite_hyp_index are logged for each run separately

        if r == 0:
            correct_lex_hyp_indices = measur.find_correct_hyp_indices(hypothesis_space, perspective_hyps, lexicon_hyps, speaker.perspective, speaker.lexicon.lexicon, 'lexicon')
            correct_composite_hyp_indices = measur.find_correct_hyp_indices(hypothesis_space, perspective_hyps, lexicon_hyps, speaker.perspective, speaker.lexicon.lexicon, 'composite')
        multi_run_context_matrix[r] = data.contexts
        multi_run_utterances_matrix[r] = data.utterances
        multi_run_log_posterior_matrix[r] = log_posteriors_per_data_point_matrix

    run_time_mins = (time.clock()-t0)/60.

    results_dict = {'multi_run_context_matrix':multi_run_context_matrix, 'multi_run_utterances_matrix':multi_run_utterances_matrix, 'multi_run_log_posterior_matrix':multi_run_log_posterior_matrix, 'correct_p_hyp_indices':correct_p_hyp_indices, 'correct_lex_hyp_indices':correct_lex_hyp_indices, 'correct_composite_hyp_indices':correct_composite_hyp_indices, 'speaker_lexicon':speaker.lexicon.lexicon,
    'run_time_mins':run_time_mins}
    return results_dict



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


lexicon_hyps_sorted = lexicon_hyps[argsort_informativity_per_lexicon]
print ''
print ''
# print "lexicon_hyps_sorted is:"
# print lexicon_hyps_sorted
print "lexicon_hyps_sorted.shape is:"
print lexicon_hyps_sorted.shape


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

    t0 = time.clock()

    # Below the actual running of the simulation happens:

    all_results_dict = multi_runs_dyadic(n_meanings, n_signals, n_runs, n_contexts, n_utterances, context_generation, context_type, context_size, helpful_contexts, speaker_lex_type, speaker_lex_index, error, extra_error, pragmatic_level_speaker, optimality_alpha_speaker, pragmatic_level_learner, optimality_alpha_learner, speaker_perspective, sal_alpha, speaker_learning_type, learner_perspective, learner_lex_type, learner_learning_type, pragmatic_level_sp_hyp_lr, hypothesis_space, perspective_hyps, lexicon_hyps, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant)


    run_simulation_time = time.clock()-t0
    print
    print 'run_simulation_time is:'
    print str((run_simulation_time/60))+" m"


    multi_run_context_matrix = all_results_dict['multi_run_context_matrix']
    # print
    # print
    # print "multi_run_context_matrix is:"
    # print multi_run_context_matrix
    # print "multi_run_context_matrix.shape is:"
    # print multi_run_context_matrix.shape


    multi_run_utterances_matrix = all_results_dict['multi_run_utterances_matrix']
    # print
    # print
    # print "multi_run_utterances_matrix is:"
    # print multi_run_utterances_matrix


    multi_run_log_posterior_matrix = all_results_dict['multi_run_log_posterior_matrix']
    # print
    # print
    # print "multi_run_log_posterior_matrix is:"
    # print multi_run_log_posterior_matrix
    # print "multi_run_log_posterior_matrix.shape is:"
    # print multi_run_log_posterior_matrix.shape


    correct_p_hyp_indices = all_results_dict['correct_p_hyp_indices']
    # print ''
    # print ''
    # print "correct_p_hyp_indices is:"
    # print correct_p_hyp_indices



    correct_lex_hyp_indices = all_results_dict['correct_lex_hyp_indices']
    # print ''
    # print ''
    # print "correct_lex_hyp_indices is:"
    # print correct_lex_hyp_indices


    correct_composite_hyp_indices = all_results_dict['correct_composite_hyp_indices']
    # print ''
    # print ''
    # print "correct_composite_hyp_indices is:"
    # print correct_composite_hyp_indices


    speaker_lexicon = all_results_dict['speaker_lexicon']
    # print
    # print
    # print "speaker_lexicon is:"
    # print speaker_lexicon



    correct_hypothesis_index, mirror_hypothesis_index = measur.find_correct_and_mirror_hyp_index(n_meanings, n_signals, lexicon_hyps, speaker_lexicon)
    # print
    # print
    # print "correct_hypothesis_index is:"
    # print correct_hypothesis_index
    # print "mirror_hypothesis_index is:"
    # print mirror_hypothesis_index




    # lex_posterior_matrix = measur.create_lex_posterior_matrix(n_runs, n_contexts, n_utterances, lexicon_hyps, multi_run_log_posterior_matrix)
    # # print
    # # print
    # # print "lex_posterior_matrix is:"
    # # print lex_posterior_matrix



    # mean_final_posteriors = measur.calc_mean_of_final_posteriors(multi_run_log_posterior_matrix)
    # print
    # print
    # # print "mean_final_posteriors is:"
    # # print mean_final_posteriors
    # print "mean_final_posteriors.shape is:"
    # print mean_final_posteriors.shape
    #
    #
    # std_final_posteriors = measur.calc_std_of_final_posteriors(multi_run_log_posterior_matrix)
    # print
    # print
    # # print "std_final_posteriors is:"
    # # print std_final_posteriors
    # print "std_final_posteriors.shape is:"
    # print std_final_posteriors.shape
    #
    #
    # mean_std_final_posteriors_dict = {'mean_final_posteriors':mean_final_posteriors,
    #                                   'std_final_posteriors':std_final_posteriors}


    percentiles_p_hyp_posterior_mass_correct = measur.calc_hyp_correct_posterior_mass_percentiles(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix, correct_p_hyp_indices)
    # print
    # print
    # print "percentiles_p_hyp_posterior_mass_correct is:"
    # print percentiles_p_hyp_posterior_mass_correct
    # print "percentiles_p_hyp_posterior_mass_correct[0].shape is:"
    # print percentiles_p_hyp_posterior_mass_correct[0].shape



    percentiles_lex_hyp_posterior_mass_correct = measur.calc_hyp_correct_posterior_mass_percentiles(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix, correct_lex_hyp_indices)
    # print
    # print
    # print "percentiles_lex_hyp_posterior_mass_correct is:"
    # print percentiles_lex_hyp_posterior_mass_correct
    # print "percentiles_lex_hyp_posterior_mass_correct[0].shape is:"
    # print percentiles_lex_hyp_posterior_mass_correct[0].shape


    percentiles_composite_hyp_posterior_mass_correct = measur.calc_hyp_correct_posterior_mass_percentiles(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix, correct_composite_hyp_indices)
    # print
    # print
    # print "percentiles_composite_hyp_posterior_mass_correct is:"
    # print percentiles_composite_hyp_posterior_mass_correct
    # print "percentiles_composite_hyp_posterior_mass_correct[0].shape is:"
    # print percentiles_composite_hyp_posterior_mass_correct[0].shape


    percentiles_lex_approximation_posterior_mass = measur.calc_lex_approximation_posterior_mass(multi_run_log_posterior_matrix, speaker_lexicon, lexicon_hyps)
    # print
    # print
    # print "percentiles_lex_approximation_posterior_mass is:"
    # print percentiles_lex_approximation_posterior_mass
    # print "percentiles_lex_approximation_posterior_mass[0].shape is:"
    # print percentiles_lex_approximation_posterior_mass[0].shape



    percentiles_correct_hyp_posterior_mass_dict = {
        'percentiles_p_hyp_posterior_mass_correct': percentiles_p_hyp_posterior_mass_correct,
        'percentiles_lex_hyp_posterior_mass_correct': percentiles_lex_hyp_posterior_mass_correct,
        'percentiles_composite_hyp_posterior_mass_correct': percentiles_composite_hyp_posterior_mass_correct,
        'percentiles_lex_approximation_posterior_mass': percentiles_lex_approximation_posterior_mass}

    mean_lex_hyp_posterior_mass_correct = percentiles_lex_hyp_posterior_mass_correct[3]
    # print ''
    # print ''
    # print "mean_lex_hyp_posterior_mass_correct.shape is:"
    # print mean_lex_hyp_posterior_mass_correct.shape


    mean_composite_hyp_posterior_mass_correct = percentiles_composite_hyp_posterior_mass_correct[3]
    # print ''
    # print ''
    # print "mean_composite_hyp_posterior_mass_correct.shape is:"
    # print mean_composite_hyp_posterior_mass_correct.shape


    mean_correct_hyp_posterior_mass_dict = {
        'mean_lex_hyp_posterior_mass_correct': mean_lex_hyp_posterior_mass_correct,
        'mean_composite_hyp_posterior_mass_correct': mean_composite_hyp_posterior_mass_correct}

    contexts_past_threshold = np.argwhere(mean_lex_hyp_posterior_mass_correct>0.5)
    # print "contexts_past_threshold is:"
    # print contexts_past_threshold
    if len(contexts_past_threshold) > 0:
        min_no_contexts_for_threshold = contexts_past_threshold[0]
    else:
        min_no_contexts_for_threshold = 'NaN'
    # print "min_no_contexts_for_threshold is:"
    # print min_no_contexts_for_threshold


    # hypotheses_percentiles = measur.calc_hypotheses_percentiles(multi_run_log_posterior_matrix)
    # # print
    # # print
    # # print "hypotheses_percentiles are:"
    # # print hypotheses_percentiles
    # # print "hypotheses_percentiles.shape is:"
    # # print hypotheses_percentiles.shape

    #
    #
    #
    #
    # min_convergence_time_perspective, mean_convergence_time_perspective, median_convergence_time_perspective, max_convergence_time_perspective = measur.calc_majority_hyp_convergence_time(multi_run_log_posterior_matrix, correct_p_hyp_indices, theta_fixed)
    # # print
    # # print
    # # print "min_convergence_time_perspective is:"
    # # print str(min_convergence_time_perspective)+' observations'
    # # print "mean_convergence_time_perspective is:"
    # # print str(mean_convergence_time_perspective)+' observations'
    # # print "median_convergence_time_perspective is:"
    # # print str(median_convergence_time_perspective)+' observations'
    # # print "max_convergence_time_perspective is:"
    # # print str(max_convergence_time_perspective)+' observations'
    #
    #
    # min_convergence_time_lexicon, mean_convergence_time_lexicon, median_convergence_time_lexicon, max_convergence_time_lexicon = measur.calc_majority_hyp_convergence_time(multi_run_log_posterior_matrix, correct_lex_hyp_indices, theta_fixed)
    # # print
    # # print
    # # print "min_convergence_time_lexicon is:"
    # # print str(min_convergence_time_lexicon)+' observations'
    # # print "mean_convergence_time_lexicon is:"
    # # print str(mean_convergence_time_lexicon)+' observations'
    # # print "median_convergence_time_lexicon is:"
    # # print str(median_convergence_time_lexicon)+' observations'
    # # print "max_convergence_time_lexicon is:"
    # # print str(max_convergence_time_lexicon)+' observations'
    #
    #
    # min_convergence_time_composite, mean_convergence_time_composite, median_convergence_time_composite, max_convergence_time_composite = measur.calc_majority_hyp_convergence_time(multi_run_log_posterior_matrix, correct_composite_hyp_indices, theta_fixed)
    # # print
    # # print
    # # print "min_convergence_time_composite is:"
    # # print str(min_convergence_time_composite)+' observations'
    # # print "mean_convergence_time_composite is:"
    # # print str(mean_convergence_time_composite)+' observations'
    # # print "median_convergence_time_composite is:"
    # # print str(median_convergence_time_composite)+' observations'
    # # print "max_convergence_time_composite is:"
    # # print str(max_convergence_time_composite)+' observations'
    #
    #
    # convergence_time_dict = {'min_convergence_time_perspective' : min_convergence_time_perspective, 'mean_convergence_time_perspective' : mean_convergence_time_perspective, 'median_convergence_time_perspective' : median_convergence_time_perspective, 'max_convergence_time_perspective' : max_convergence_time_perspective, 'min_convergence_time_lexicon' : min_convergence_time_lexicon, 'mean_convergence_time_lexicon' : mean_convergence_time_lexicon, 'median_convergence_time_lexicon' : median_convergence_time_lexicon, 'max_convergence_time_lexicon' : max_convergence_time_lexicon, 'min_convergence_time_composite' : min_convergence_time_composite, 'mean_convergence_time_composite' : mean_convergence_time_composite, 'median_convergence_time_composite' : median_convergence_time_composite, 'max_convergence_time_composite' : max_convergence_time_composite}
    #
    #
    #
    #
    # percentiles_convergence_time_over_theta_perspective = measur.calc_convergence_time_over_theta_range_percentiles(multi_run_log_posterior_matrix, correct_p_hyp_indices, theta_range)
    # # print
    # # print
    # # print "percentiles_convergence_time_over_theta_perspective are:"
    # # print percentiles_convergence_time_over_theta_perspective
    # # print "percentiles_convergence_time_over_theta_perspective.shape are:"
    # # print percentiles_convergence_time_over_theta_perspective.shape
    #
    #
    #
    # percentiles_convergence_time_over_theta_lexicon = measur.calc_convergence_time_over_theta_range_percentiles(multi_run_log_posterior_matrix, correct_lex_hyp_indices, theta_range)
    # # print
    # # print
    # # print "percentiles_convergence_time_over_theta_lexicon are:"
    # # print percentiles_convergence_time_over_theta_lexicon
    # # print "percentiles_convergence_time_over_theta_lexicon.shape are:"
    # # print percentiles_convergence_time_over_theta_lexicon.shape
    #
    #
    #
    # percentiles_convergence_time_over_theta_composite = measur.calc_convergence_time_over_theta_range_percentiles(multi_run_log_posterior_matrix, correct_composite_hyp_indices, theta_range)
    # # print
    # # print
    # # print "percentiles_convergence_time_over_theta_composite are:"
    # # print percentiles_convergence_time_over_theta_composite
    # # print "percentiles_convergence_time_over_theta_composite.shape are:"
    # # print percentiles_convergence_time_over_theta_composite.shape
    #
    #
    # percentiles_converge_time_over_theta_dict = {'percentiles_convergence_time_over_theta_perspective':percentiles_convergence_time_over_theta_perspective, 'percentiles_convergence_time_over_theta_lexicon':percentiles_convergence_time_over_theta_lexicon, 'percentiles_convergence_time_over_theta_composite':percentiles_convergence_time_over_theta_composite}
    #
    #



    calc_performance_measures_time = time.clock()-t0
    print
    print 'calc_performance_measures_time is:'
    print str((calc_performance_measures_time/60))+" m"







    #############################################################################
    # Below the results are saved as pickle files:


    if n_runs > 0 and run_type == 'dyadic':
        t2 = time.clock()

        run_type_dir = 'Learner_Speaker'



        if context_generation == 'random':
            file_title = run_type+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+str(context_generation)+'_err_'+error_string+'_sp_'+pragmatic_level_speaker+'_sp_a_'+str(optimality_alpha_speaker)[0]+'_sp_lex_'+speaker_lex_type[:-4]+'_index_'+str(speaker_lex_index)+'_inf_'+inf_level+'_sp_p_'+str(speaker_perspective)[0]+'_lr_'+pragmatic_level_learner+'_lr_a_'+str(optimality_alpha_learner)[0]+'_lr_sp_hyp_'+pragmatic_level_sp_hyp_lr+'_'+learner_learning_type+'_lex_prior_'+lexicon_prior_type+'_'+lexicon_prior_constant_string+'_p_prior_'+perspective_prior_type+'_'+perspective_prior_strength_string+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U_'+lex_measure

        elif context_generation == 'only_helpful' or context_generation == 'optimal':
            file_title = run_type+'_'+str(n_runs)+'_R_'+str(int(n_meanings))+'_M_'+str(int(n_signals))+'_S_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_C'+'_err_'+error_string+'_sp_'+pragmatic_level_speaker+'_sp_a_'+str(optimality_alpha_speaker)[0]+'_sp_lex_'+speaker_lex_type[:-4]+'_index_'+str(speaker_lex_index)+'_inf_'+inf_level+'_sp_p_'+str(speaker_perspective)[0]+'_lr_'+pragmatic_level_learner+'_lr_a_'+str(optimality_alpha_learner)[0]+'_lr_sp_hyp_'+pragmatic_level_sp_hyp_lr+'_'+learner_learning_type+'_lex_prior_'+lexicon_prior_type+'_'+lexicon_prior_constant_string+'_p_prior_'+perspective_prior_type+'_'+perspective_prior_strength_string+'_'+str(int(n_contexts))+'_C_'+str(int(n_utterances))+'_U_'+lex_measure




        pickle_file_title_all_results = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/'+run_type_dir+'/Results_'+file_title

        saveresults.write_results_to_pickle_file(pickle_file_title_all_results, all_results_dict)


        #
        # pickle_file_title_mean_std_final_posteriors = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/'+run_type_dir+'/Mean_Std_Final_Post_'+file_title
        #
        # saveresults.write_results_to_pickle_file(pickle_file_title_mean_std_final_posteriors, mean_std_final_posteriors_dict)
        #
        #

        # pickle_file_title_lex_posterior_matrix = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/' + run_type_dir + '/Lex_Post_Matrix' + file_title
        #
        # saveresults.write_results_to_pickle_file(pickle_file_title_lex_posterior_matrix, lex_posterior_matrix)
        #


        pickle_file_title_correct_hyp_posterior_mass_percentiles = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/' + run_type_dir + '/Correct_Hyp_Post_Mass_Percentiles_' + file_title

        saveresults.write_results_to_pickle_file(pickle_file_title_correct_hyp_posterior_mass_percentiles, percentiles_correct_hyp_posterior_mass_dict)



        pickle_file_title_correct_hyp_posterior_mass_mean = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/' + run_type_dir + '/Correct_Hyp_Post_Mass_Mean_' + file_title

        saveresults.write_results_to_pickle_file(pickle_file_title_correct_hyp_posterior_mass_mean, mean_correct_hyp_posterior_mass_dict)



        # pickle_file_title_hypothesis_percentiles = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/' + run_type_dir + '/Hyp_Percentiles' + file_title
        #
        # saveresults.write_results_to_pickle_file(pickle_file_title_hypothesis_percentiles, hypotheses_percentiles)
        #

        #
        # pickle_file_title_convergence_time_min_max = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/'+run_type_dir+'/Convergence_Time_Percentiles'+file_title
        #
        # saveresults.write_results_to_pickle_file(pickle_file_title_convergence_time_min_max, convergence_time_dict)
        #
        #
        #
        # pickle_file_title_convergence_time_percentiles = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/'+run_type_dir+'/Convergence_Time_Percentiles'+file_title
        #
        # saveresults.write_results_to_pickle_file(pickle_file_title_convergence_time_percentiles, percentiles_converge_time_over_theta_dict)
        #





        write_to_files_time = time.clock()-t2
        print
        print 'write_to_files_time is:'
        print str((write_to_files_time/60))+" m"


