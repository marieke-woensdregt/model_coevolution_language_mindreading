__author__ = 'Marieke Woensdregt'


import time
from params_learner_speaker import *
import lex
import context
import prior
from pop import Agent, PragmaticAgent
import measur_for_eddie


def multi_runs_dyadic(n_meanings, n_signals, n_runs, n_contexts, n_utterances, context_generation, context_type, context_size, helpful_contexts, speaker_lex_type, speaker_lex_index, error, pragmatic_level_speaker, optimality_alpha_speaker, pragmatic_level_learner, optimality_alpha_learner, speaker_perspective, sal_alpha, speaker_learning_type, learner_perspective, learner_lex_type, learner_learning_type, hypothesis_space, perspective_hyps, lexicon_hyps, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant):
    """
    :param: All parameters that this function takes are variables specified in the params_learner_speaker module
    :return: A dictionary containing by index 0) multi_run_context_matrix; 1) multi_run_utterances_matrix; 2) multi_run_log_posterior_matrix; 3) majority_p_hyp_indices; 4) majority_lex_hyp_indices; 5) majority_composite_hyp_index and 6) speaker.lexicon.lexicon
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
        if r % report_every_r == 0:
            print 'r = '+str(r)

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
        correct_p_hyp_indices = measur_for_eddie.find_correct_hyp_indices(hypothesis_space, perspective_hyps, lexicon_hyps, speaker.perspective, speaker.lexicon.lexicon, 'perspective')


        # print
        # print 'data is:'
        # print data.print_data()



        # FIXME: If I want the half_ambiguous lexicon to be generated with the ambiguous mappings chosen at random, I have to make sure that the majority_lex_hyp_indices and majority_composite_hyp_index are logged for each run separately

        if r == 0:
            correct_lex_hyp_indices = measur_for_eddie.find_correct_hyp_indices(hypothesis_space, perspective_hyps, lexicon_hyps, speaker.perspective, speaker.lexicon.lexicon, 'lexicon')
            correct_composite_hyp_indices = measur_for_eddie.find_correct_hyp_indices(hypothesis_space, perspective_hyps, lexicon_hyps, speaker.perspective, speaker.lexicon.lexicon, 'composite')
        multi_run_context_matrix[r] = data.contexts
        multi_run_utterances_matrix[r] = data.utterances
        multi_run_log_posterior_matrix[r] = log_posteriors_per_data_point_matrix

    run_time_mins = (time.clock()-t0)/60.

    results_dict = {'multi_run_context_matrix':multi_run_context_matrix, 'multi_run_utterances_matrix':multi_run_utterances_matrix, 'multi_run_log_posterior_matrix':multi_run_log_posterior_matrix, 'correct_p_hyp_indices':correct_p_hyp_indices, 'correct_lex_hyp_indices':correct_lex_hyp_indices, 'correct_composite_hyp_indices':correct_composite_hyp_indices, 'speaker_lexicon':speaker.lexicon.lexicon,
    'run_time_mins':run_time_mins}
    return results_dict
