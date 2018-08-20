__author__ = 'Marieke Woensdregt'


import numpy as np
import time

import context
import lex
import measur
from pop import Agent, PragmaticAgent
import prior


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



    mean_posteriors_over_observations = np.mean(np.exp(multi_run_log_posterior_matrix), axis=0)

    std_posteriors_over_observations = np.std(np.exp(multi_run_log_posterior_matrix), axis=0)

    first_quartile_posteriors_over_observations = np.percentile(np.exp(multi_run_log_posterior_matrix), q=25, axis=0)
    median_posteriors_over_observations = np.percentile(np.exp(multi_run_log_posterior_matrix), q=50, axis=0)
    third_quartile_posteriors_over_observations = np.percentile(np.exp(multi_run_log_posterior_matrix), q=75, axis=0)
    percentiles_posteriors_over_observations = np.array([first_quartile_posteriors_over_observations, median_posteriors_over_observations, third_quartile_posteriors_over_observations])

    run_time_mins = (time.clock()-t0)/60.

    results_dict = {'multi_run_context_matrix':multi_run_context_matrix, 'multi_run_utterances_matrix':multi_run_utterances_matrix, 'mean_posteriors_over_observations':mean_posteriors_over_observations, 'std_posteriors_over_observations':std_posteriors_over_observations, 'percentiles_posteriors_over_observations':percentiles_posteriors_over_observations, 'correct_p_hyp_indices':correct_p_hyp_indices, 'correct_lex_hyp_indices':correct_lex_hyp_indices, 'correct_composite_hyp_indices':correct_composite_hyp_indices, 'speaker_lexicon':speaker.lexicon.lexicon,
    'run_time_mins':run_time_mins}
    return results_dict
