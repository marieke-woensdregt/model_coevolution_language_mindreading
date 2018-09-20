__author__ = 'Marieke Woensdregt'

from scipy.special import logsumexp
from scipy.spatial.distance import pdist

from prior import *
from hypspace import *
from context import *




##############################################################################################
## MEASURES OF COMMUNICATIVE SUCCESS:


def calc_ca_lex_only(ca_measure_type, parent, learner, n_meanings, n_signals, production_error):
    # This method calculates communicative accuracy in both directions: How well the learner understands their cultural parent and also how well the learner can express meanings to their cultural parent. The method does NOT take perspective into account.
    if learner.pragmatic_level != 'literal':
        raise ValueError("lex_only method of CA calculation only works when pragmatic_level == 'literal'")
    production_probs_parent = np.zeros((n_meanings, n_signals))
    production_probs_learner = np.zeros((n_meanings, n_signals))
    reception_probs_parent = np.zeros((n_signals, n_meanings))
    reception_probs_learner = np.zeros((n_signals, n_meanings))
    #TODO: I added the uniform_intention below:
    uniform_intention = np.array([1./n_meanings for s in range(n_signals)])
    for m in range(n_meanings):
        signal_probs_parent = learner.calc_signal_probs(parent.lexicon.lexicon, m, production_error)
        production_probs_parent[m] = signal_probs_parent
        #TODO I added the line below where the production probs are multiplied with the topic probability:
        production_probs_parent[m] = production_probs_parent[m]*uniform_intention[m]
    for s in range(n_signals):
        meaning_probs_learner = learner.calc_meaning_probs(learner.lexicon.lexicon, s, production_error)
        reception_probs_learner[s] = meaning_probs_learner
        #TODO: I added the normalization step below!
        reception_probs_learner[s] = np.divide(reception_probs_learner[s], np.sum(reception_probs_learner[s]))
    reception_probs_learner = reception_probs_learner.T
    learner_comprehension_success = np.multiply(production_probs_parent, reception_probs_learner)
    success_per_meaning_comprehension = np.sum(learner_comprehension_success, axis=1)
    #TODO: Below I changed it to sum instead of mean!
    ca_comprehension = np.sum(success_per_meaning_comprehension)
    ca = ca_comprehension
    if ca_measure_type == "comp_n_prod":
        for m in range(n_meanings):
            signal_probs_learner = learner.calc_signal_probs(learner.lexicon.lexicon, m, production_error)
            production_probs_learner[m] = signal_probs_learner
            #TODO I added the line below where the production probs are multiplied with the topic probability:
            production_probs_learner[m] = production_probs_learner[m]*uniform_intention[m]
        for s in range(n_signals):
            meaning_probs_parent = learner.calc_meaning_probs(parent.lexicon.lexicon, s, production_error)
            reception_probs_parent[s] = meaning_probs_parent
            #TODO: I added the normalization step below!
            reception_probs_parent[s] = np.divide(reception_probs_parent[s], np.sum(reception_probs_parent[s]))
        reception_probs_parent = reception_probs_parent.T
        learner_production_success = np.multiply(production_probs_learner, reception_probs_parent)
        success_per_meaning_production = np.sum(learner_production_success, axis=1)
        #TODO: Below I changed it to sum instead of mean!
        ca_production = np.sum(success_per_meaning_production)
        ca = np.mean(np.array([ca_comprehension, ca_production]))
    return ca


def calc_ca_lex_n_context(context_matrix, ca_measure_type, n_interactions, parent, learner, n_utterances, n_meanings, n_signals, production_error):
    if learner.pragmatic_level != 'literal':
        raise ValueError("lex_only method of CA calculation only works when pragmatic_level == 'literal'")
    if ca_measure_type == 'comp_n_prod':
        n_parent_productions = n_interactions/2
    elif ca_measure_type == 'comp_only':
        n_parent_productions = n_interactions
    parent_data = parent.produce_data(n_meanings, n_signals, context_matrix[0:n_parent_productions], n_utterances, production_error)
    learner_interpretations = learner.interpret_data(n_meanings, parent_data, 'lex_n_context', production_error, perspective=None)
    score_list = np.zeros(n_interactions * n_utterances)
    counter = 0
    for i in range(len(parent_data.topics)):
        for j in range(len(parent_data.topics[0])):
            topic = parent_data.topics[i][j].astype(float)
            interpretation = learner_interpretations[i][j].astype(float)
            if topic == interpretation:
                score_list[counter] = 1.
            counter += 1
    if ca_measure_type == 'comp_n_prod':
        learner_data = learner.produce_data(n_meanings, n_signals, context_matrix[n_parent_productions:], n_utterances, production_error)
        parent_interpretations = parent.interpret_data(n_meanings, learner_data, 'lex_n_context', production_error, perspective=None)
        for i in range(len(learner_data.topics)):
            for j in range(len(learner_data.topics[0])):
                topic = learner_data.topics[i][j].astype(float)
                interpretation = parent_interpretations[i][j].astype(float)
                if topic == interpretation:
                    score_list[counter] = 1.
                counter += 1
    ca = np.mean(score_list)
    return ca


def calc_ca_lex_n_perspective(context_matrix, ca_measure_type, parent, learner, speaker_p_hyp, sal_alpha, production_error):
    if ca_measure_type != 'comp_only':
        raise ValueError("lex_n_perspective method of CA calculation only works when ca_measure_type == 'comp_only'")
    ca_per_context = np.zeros(len(context_matrix))
    for c in range(len(context_matrix)):
        context = context_matrix[c]
        s_prod_probs = parent.calc_literal_speaker(context, parent.lexicon.lexicon, production_error, parent.perspective, sal_alpha)
        log_s_prod_probs = np.log(s_prod_probs)
        l_rec_probs = learner.calc_pragmatic_listener(context, learner.lexicon.lexicon, speaker_p_hyp, 'literal', production_error, sal_alpha)
        l_rec_probs_transposed = l_rec_probs.T
        log_l_rec_probs_transposed = np.log(l_rec_probs_transposed)
        log_multiplied_matrices = np.add(log_s_prod_probs, log_l_rec_probs_transposed)
        multiplied_matrices = np.exp(log_multiplied_matrices)
        sum_over_signals = np.sum(multiplied_matrices, axis=1)
        ca = np.sum(sum_over_signals)
        ca_per_context[c] = ca
    mean_ca = np.mean(ca_per_context)
    return mean_ca


def calc_ca_pragmatic(context_matrix, ca_measure_type, speaker, listener, n_signals, sal_alpha, error, speaker_type, s_p_hyp, s_type_hyp, extra_error):
    if ca_measure_type != 'comp_only':
        raise ValueError("lex_n_perspective method of CA calculation only works when ca_measure_type == 'comp_only'")
    ca_per_context = np.zeros(len(context_matrix))
    for c in range(len(context_matrix)):
        context = context_matrix[c]
        parent_intention = speaker.calc_intention(speaker.perspective, context)
        log_parent_intention = np.log(parent_intention)
        l_rec_probs_according_to_speaker = speaker.calc_pragmatic_listener(context, speaker.lexicon.lexicon, speaker.perspective, 'literal', error, sal_alpha)
        s_prod_probs = speaker.calc_pragmatic_speaker(l_rec_probs_according_to_speaker)
        log_s_prod_probs = np.log(s_prod_probs)
        l_rec_probs = listener.calc_pragmatic_listener(context, listener.lexicon.lexicon, s_p_hyp, s_type_hyp, error, sal_alpha)
        l_rec_probs_transposed = l_rec_probs.T
        log_l_rec_probs_transposed = np.log(l_rec_probs_transposed)
        if extra_error == True:
            for m in range(len(s_prod_probs)):
                meaning_row = s_prod_probs[m]
                addition_matrix = np.zeros((n_signals, n_signals))
                for s in range(len(meaning_row)):
                    error_prob = meaning_row[s]*error
                    for t in range(len(addition_matrix[s])):
                        if t == s:
                            addition_matrix[s][t] = -error_prob
                        else:
                            addition_matrix[s][t] = error_prob/2.
                addition_matrix_summed = np.sum(addition_matrix, axis=0)
                new_meaning_row = np.add(meaning_row, addition_matrix_summed)
                s_prod_probs[m] = new_meaning_row
            log_s_prod_probs = np.log(s_prod_probs)
        log_multiplied_matrices = np.add(log_s_prod_probs, log_l_rec_probs_transposed)
        multiplied_matrices = np.exp(log_multiplied_matrices)
        sum_over_signals = np.sum(multiplied_matrices, axis=1)
        log_sum_over_signals = np.log(sum_over_signals)
        log_sum_over_signals_times_topic_probs = np.add(log_sum_over_signals, log_parent_intention)
        sum_over_signals_times_topic_probs = np.exp(log_sum_over_signals_times_topic_probs)
        ca = np.sum(sum_over_signals_times_topic_probs)
        ca_per_context[c] = ca
    mean_ca = np.mean(ca_per_context)
    return mean_ca


def calc_comm_acc(context_matrix, communication_type, ca_measure_type, n_interactions, n_utterances, speaker, listener, n_meanings, n_signals, sal_alpha, error, speaker_type, s_p_hyp=None, s_type_hyp=None, extra_error=None):
    if communication_type == 'lex_only':
        mean_ca = calc_ca_lex_only(ca_measure_type, speaker, listener, n_meanings, n_signals, error)
    elif communication_type == 'lex_n_context':
        mean_ca = calc_ca_lex_n_context(context_matrix, ca_measure_type, n_interactions, speaker, listener, n_utterances, n_meanings, n_signals, error)
    elif communication_type == 'lex_n_p':
        # ca_per_context, mean_ca = self.calc_ca_lex_n_perspective(n_interactions, speaker, listener)
        mean_ca = calc_ca_lex_n_perspective(context_matrix, ca_measure_type, speaker, listener, s_p_hyp, sal_alpha, error)
    elif communication_type == 'prag':
        # ca_per_context, mean_ca = self.calc_ca_pragmatic(n_interactions, speaker, listener)
        mean_ca = calc_ca_pragmatic(context_matrix, ca_measure_type, speaker, listener, n_signals, sal_alpha, error, speaker_type, s_p_hyp, s_type_hyp, extra_error)
    # return ca_per_context, mean_ca
    return mean_ca




##############################################################################################
## MEASURES OF LEARNING:

# The function below finds the indices of the correct (part) hypotheses:
def find_correct_hyp_indices(hypothesis_space, perspective_hyps, lexicon_hyps, speaker_perspectives, speaker_lexicon, which_hyps):
    """
    :param hypothesis_space: The full space of composite hypotheses (2D numpy matrix)
    :param speaker_perspectives: The speaker's perspective
    :param speaker_lexicon: The speaker's lexicon
    :param which_hyps: The hypotheses for which we want the index. This can be either 'perspective' for all hypotheses that contain the correct perspective, 'lexicon' for all hypotheses that contain the correct lexicon, or 'composite' for only the one correct composite hypothesis
    :return: A 1D numpy array of correct hypothesis indices
    """

    correct_hyp_indices = [] # I do this as a list rather than a numpy array because the indices have to be integeres, not floats

    if which_hyps == 'perspective':
        for i in range(len(hypothesis_space)):
            composite_hypothesis = hypothesis_space[i]
            p_hyp_indices = composite_hypothesis[0]
            #FIXME: again, calling perspective_hyps and lexicon_hyps as global variables from params_and_run module...
            p_hyps = np.zeros_like(p_hyp_indices, dtype=float)
            if isinstance(p_hyp_indices, list) or isinstance(p_hyp_indices, np.ndarray):
                for j in range(len(p_hyp_indices)):
                    p_hyp_index = p_hyp_indices[j]
                    p_hyp = perspective_hyps[p_hyp_index]
                    p_hyps[j] = p_hyp
            elif isinstance(p_hyp_indices, int):
                p_hyps = p_hyp_indices
            if np.array_equal(p_hyps, speaker_perspectives):
                correct_hyp_indices.append(i)

    elif which_hyps == 'lexicon':
        for i in range(len(hypothesis_space)):
            composite_hypothesis = hypothesis_space[i]
            lex_hyp_index = composite_hypothesis[1]
            #FIXME: again, calling perspective_hyps and lexicon_hyps as global variables from params_and_run module...
            lex_hyp = lexicon_hyps[lex_hyp_index]
            if np.array_equal(lex_hyp, speaker_lexicon):
                correct_hyp_indices.append(i)

    elif which_hyps == 'composite':
        for i in range(len(hypothesis_space)):
            composite_hypothesis = hypothesis_space[i]
            p_hyp_indices = composite_hypothesis[0]
            lex_hyp_index = composite_hypothesis[1]
            #FIXME: again, calling perspective_hyps and lexicon_hyps as global variables from params_and_run module...
            if isinstance(p_hyp_indices, list) or isinstance(p_hyp_indices, np.ndarray):
                p_hyps = np.zeros(len(p_hyp_indices))
                for j in range(len(p_hyp_indices)):
                    p_hyp_index = p_hyp_indices[j]
                    p_hyp = perspective_hyps[p_hyp_index]
                    p_hyps[j] = p_hyp
            elif isinstance(p_hyp_indices, int):
                p_hyps = p_hyp_indices
            lex_hyp = lexicon_hyps[lex_hyp_index]
            if np.array_equal(p_hyps, speaker_perspectives) and np.array_equal(lex_hyp, speaker_lexicon):
                correct_hyp_indices.append(i)
    return correct_hyp_indices




# The function below finds the indices of the correct (part) hypotheses:
def find_correct_hyp_indices_with_speaker_distinction(hypothesis_space, perspective_hyps, lexicon_hyps, speaker_perspectives, speaker_index, speaker_lexicon, which_hyps):
    """
    :param hypothesis_space: The full space of composite hypotheses (2D numpy matrix)
    :param speaker_perspectives: The speaker's perspective
    :param speaker_lexicon: The speaker's lexicon
    :param which_hyps: The hypotheses for which we want the index. This can be either 'perspective' for all hypotheses that contain the correct perspective, 'lexicon' for all hypotheses that contain the correct lexicon, or 'composite' for only the one correct composite hypothesis
    :return: A 1D numpy array of correct hypothesis indices
    """

    correct_hyp_indices = [] # I do this as a list rather than a numpy array because the indices have to be integeres, not floats

    if which_hyps == 'perspective':
        for i in range(len(hypothesis_space)):
            composite_hypothesis = hypothesis_space[i]
            p_hyp_indices = composite_hypothesis[0]
            #FIXME: again, calling perspective_hyps and lexicon_hyps as global variables from params_and_run module...
            p_hyps = np.zeros_like(p_hyp_indices, dtype=float)
            for j in range(len(p_hyp_indices)):
                p_hyp_index = p_hyp_indices[j]
                p_hyp = perspective_hyps[p_hyp_index]
                p_hyps[j] = p_hyp
            if np.array_equal(p_hyps[speaker_index], speaker_perspectives[speaker_index]):
                correct_hyp_indices.append(i)

    elif which_hyps == 'lexicon':
        for i in range(len(hypothesis_space)):
            composite_hypothesis = hypothesis_space[i]
            lex_hyp_index = composite_hypothesis[1]
            #FIXME: again, calling perspective_hyps and lexicon_hyps as global variables from params_and_run module...
            lex_hyp = lexicon_hyps[lex_hyp_index]
            if np.array_equal(lex_hyp, speaker_lexicon):
                correct_hyp_indices.append(i)

    elif which_hyps == 'composite':
        for i in range(len(hypothesis_space)):
            composite_hypothesis = hypothesis_space[i]
            p_hyp_indices = composite_hypothesis[0]
            lex_hyp_index = composite_hypothesis[1]
            #FIXME: again, calling perspective_hyps and lexicon_hyps as global variables from params_and_run module...
            p_hyps = np.zeros(len(p_hyp_indices))
            for j in range(len(p_hyp_indices)):
                p_hyp_index = p_hyp_indices[j]
                p_hyp = perspective_hyps[p_hyp_index]
                p_hyps[j] = p_hyp
            lex_hyp = lexicon_hyps[lex_hyp_index]
            if np.array_equal(p_hyps, speaker_perspectives) and np.array_equal(lex_hyp, speaker_lexicon):
                correct_hyp_indices.append(i)

    return correct_hyp_indices





def find_majority_lexicon(lexicon_types, population_lexicons, population_lexicon_type_probs):
    """
    :param population_lexicons: A list containing the different lexicons that can be present in a population. The indices in this list have to correspond to those of parameter 'population_lexicon_type_probs'
    :param population_lexicon_type_probs: A 1D numpy array containing the probabilities with which the different lexicons are present in the population
    :return: The lexicon that is used by the majority of speakers in the population (2D numpy array with meanings on the rows and signals on the columns)
    """
    majority_lex_type_index = np.argmax(population_lexicon_type_probs)
    #FIXME: calling lexicon_types as global variable from params_and_run module (but maybe that's ok)
    majority_lex_type = lexicon_types[majority_lex_type_index]
    if majority_lex_type == 'optimal_lex':
        majority_lexicon = population_lexicons[0].lexicon
    elif majority_lex_type == 'half_ambiguous_lex':
        majority_lexicon = population_lexicons[1].lexicon
    elif majority_lex_type == 'fully_ambiguous_lex':
        majority_lexicon = population_lexicons[2].lexicon
    return majority_lexicon


def find_majority_hyp_indices(hypothesis_space, perspective_hyps, lexicon_hyps, population_perspective_probs, population_lexicons, population_lexicon_type_probs, which_hyps):
    """
    :param hypothesis_space: The full space of composite hypotheses (2D numpy matrix)
    :param population_perspectives: The different perspectives that can be present in a population
    :param population_perspective_probs: The probabilities with which the different possible perspectives are present in the population (of which the indices have to correspond to those in parameter population_perspectives
    :param population_lexicons: A list of the different lexicons that can be present in a population
    :param population_lexicon_type_probs: The probabilities with which the different possible lexicons are present in the population (of which the indices have to correspond to those in parameter population_lexicons
    :param which_hyps: The hypothesis type for which we want to find the indices. This can be set to either 'perspective', 'lexicon' or 'composite'
    :return: A 1D numpy array of majority hypothesis indices
    """
    majority_perspective_index = np.argmax(population_perspective_probs)
    #FIXME: calling perspectives as global variable from params_and_run module (but maybe that's ok)
    majority_perspective = perspective_hyps[majority_perspective_index]
    majority_lexicon = find_majority_lexicon(population_lexicons, population_lexicons, population_lexicon_type_probs)
    majority_hyp_indices = find_correct_hyp_indices(hypothesis_space, perspective_hyps, lexicon_hyps, majority_perspective, majority_lexicon, which_hyps)
    return majority_hyp_indices







#######################################################################################################################

# The functions below calculated the performance of the learner:



# TODO: Write functions that are similar to the ones below, but calculate how much the learner's posterior probability distribution matches the actual probabilities with which lexicons and perspectives are present in the population, rather than just measuring whether the learner converges on the majority lexicon or perspective.


#TODO: Write this function:
def calc_posterior_pop_probs_match(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix, agent_type, pop_size, pop_perspective_probs, pop_lex_type_probs, pop_lexicons, hypothesis_space, perspective_hyps, lexicon_hyps):
    """
    :param multi_run_log_posterior_matrix: A 3D multi_run_log_posterior_matrix with shape (n_runs, n_contexts, n_hypotheses)
    :param pop_perspective_probs: The population's perspective probabilities
    :param pop_lex_type_probs: The population's lexicon type probabilities
    :param pop_lexicons: The population's lexicons
    :return: A list containing the first quartile, median and third quartile of approximation of the real perspective and lexicon probabilities over time (percentiles calculated over runs)
    """
    #FIXME: again, calling lexicon_hyps as global variables from params_and_run module...
    pop_lexicon_probs = np.zeros(len(lexicon_hyps))
    for i in range(len(lexicon_hyps)):
        lexicon = lexicon_hyps[i]
        for j in range(len(pop_lexicons)):
            if np.array_equal(lexicon, pop_lexicons[j].lexicon):
                pop_lexicon_probs[i] = pop_lex_type_probs[j]
    #FIXME: again, calling hypothesis_space as global variables from params_and_run module...
    real_log_probs_per_hyp_array = list_composite_log_priors(agent_type, pop_size, hypothesis_space, perspective_hyps, lexicon_hyps, pop_perspective_probs, pop_lexicon_probs)
    unlogged_real_probs_per_hyp_array = np.exp(real_log_probs_per_hyp_array)
    posterior_pop_probs_approximation_matrix = np.zeros((n_runs, ((n_contexts*n_utterances)+1)))
    for r in range(len(multi_run_log_posterior_matrix)):
        log_posteriors_accumulator = multi_run_log_posterior_matrix[r]
        for i in range(len(log_posteriors_accumulator)):
            log_posteriors = log_posteriors_accumulator[i]
            unlogged_posteriors = np.exp(log_posteriors)
            difference_posteriors_real_probs_array = np.absolute(np.subtract(unlogged_real_probs_per_hyp_array, unlogged_posteriors))
            summed_difference_posteriors_real_probs = np.sum(difference_posteriors_real_probs_array)
            # FIXME: Check whether it's really right to divide this by 2: (I think 2 is the maximum difference that could happen between two prior/posterior arrays, because both sum to 1, and it is the absolute difference, so the most that could happen is that all posterior is on some hypotheses (summing to 1) whereas the real probabilities are all on other hypotheses (also summing to 1), together making a total summed_difference of 2.
            normalized_difference_posteriors_real_probs = summed_difference_posteriors_real_probs/2.
            approximation_real_probs = np.subtract(1., normalized_difference_posteriors_real_probs)
            posterior_pop_probs_approximation_matrix[r][i] = approximation_real_probs
    percentile_25 = np.percentile(posterior_pop_probs_approximation_matrix, q=25, axis=0)
    percentile_50 = np.percentile(posterior_pop_probs_approximation_matrix, q=50, axis=0)
    percentile_75 = np.percentile(posterior_pop_probs_approximation_matrix, q=75, axis=0)
    return [percentile_25, percentile_50, percentile_75]







def calc_hyp_correct_posterior_mass(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix, correct_hyp_indices):
    """
    :param multi_run_log_posterior_matrix: A 3D multi_run_log_posterior_matrix with shape (n_runs, n_contexts, n_hypotheses)
    :param correct_hyp_indices: A 1D numpy array with all the indices of the correct hypotheses in the learner's full hypothesis_space
    :return:
    """
    posterior_correct_mass_accumulator = np.zeros((n_runs, ((n_contexts*n_utterances)+1)))
    for r in range(len(multi_run_log_posterior_matrix)):
        log_posteriors_accumulator = multi_run_log_posterior_matrix[r]
        for i in range(len(log_posteriors_accumulator)):
            log_posteriors = log_posteriors_accumulator[i]
            log_posterior_mass_correct = logsumexp(log_posteriors[correct_hyp_indices])
            posterior_mass_correct_exp = np.exp(log_posterior_mass_correct)
            posterior_correct_mass_accumulator[r][i] = posterior_mass_correct_exp
    return posterior_correct_mass_accumulator





def calc_hyp_correct_posterior_mass_percentiles(n_runs, n_contexts, n_utterances, multi_run_log_posterior_matrix, correct_hyp_indices):
    """
    :param multi_run_log_posterior_matrix: A 3D multi_run_log_posterior_matrix with shape (n_runs, n_contexts, n_hypotheses)
    :param correct_hyp_indices: A 1D numpy array with all the indices of the correct hypotheses in the learner's full hypothesis_space
    :return: A list containing the first quartile, median, third quartile, and mean of posterior probability mass assigned to the correct hypotheses over time (percentiles calculated over runs)
    """
    posterior_correct_mass_accumulator = np.zeros((n_runs, ((n_contexts*n_utterances)+1)))
    for r in range(len(multi_run_log_posterior_matrix)):
        log_posteriors_accumulator = multi_run_log_posterior_matrix[r]
        for i in range(len(log_posteriors_accumulator)):
            log_posteriors = log_posteriors_accumulator[i]
            log_posterior_mass_correct = logsumexp(log_posteriors[correct_hyp_indices])
            posterior_mass_correct_exp = np.exp(log_posterior_mass_correct)
            posterior_correct_mass_accumulator[r][i] = posterior_mass_correct_exp
    percentile_25 = np.percentile(posterior_correct_mass_accumulator, q=25, axis=0)
    percentile_50 = np.percentile(posterior_correct_mass_accumulator, q=50, axis=0)
    percentile_75 = np.percentile(posterior_correct_mass_accumulator, q=75, axis=0)
    mean = np.mean(posterior_correct_mass_accumulator, axis=0)
    return np.asarray([percentile_25, percentile_50, percentile_75, mean])




def calc_cumulative_belief_in_correct_hyps_per_speaker(n_runs, n_contexts, multi_run_log_posterior_matrix, hypothesis_space, perspective_hyps, lexicon_hyps, perspectives_per_speaker, speaker_index, lexicons_per_speaker, which_hyps):
    """
    :param n_runs: Number of runs
    :param n_contexts: Number of contexts per run
    :param n_utterances: Number of utterances per context
    :param multi_run_log_posterior_per_speaker_matrix: A 3D numpy array with axis 0 = runs, axis 1 = (contexts*utterances) and axis 2 = composite hypotheses
    :param perspectives_per_speaker: A 1D numpy array containing the perspectives for each speaker in the population (matching by index)
    :param lexicons_per_speaker: A 3D numpy array containing the perspectives for each speaker in the population (matching by index). With axis 0 = speakers, axis 1 = meanings and axis 2 = signals
    :param which_hyps: Determines for which part or composite hypothesis the percentiles should be calculated. Can be set to either 'perspective', 'lexicon' or 'composite'
    :return: A list containing where index 0 = first quartile (i.e. percentile 25), index 1 = median (i.e. percentile 50), index 2 = third quartile (i.e. percentile 75) and index 3 = mean.
    """
    posterior_correct_mass_accumulator = np.zeros((n_runs, (n_contexts+1))) # It's (n_contexts+1) here because the first set of posteriors that will be shown in the plot are the initial posteriors at context 0
    log_prior = multi_run_log_posterior_matrix[0][0]
    for run in range(n_runs):
        speaker_perspectives = perspectives_per_speaker[run]
        speaker_lexicon = lexicons_per_speaker[run][int(speaker_index)]
        correct_hyp_indices = find_correct_hyp_indices_with_speaker_distinction(hypothesis_space, perspective_hyps, lexicon_hyps, speaker_perspectives, speaker_index, speaker_lexicon, which_hyps)
        for context in range(len(multi_run_log_posterior_matrix[run])):
            log_posteriors = multi_run_log_posterior_matrix[run][context]
            log_posterior_mass_correct = logsumexp(log_posteriors[correct_hyp_indices])
            posterior_mass_correct_exp = np.exp(log_posterior_mass_correct)
            log_prior_mass_correct = logsumexp(log_prior[correct_hyp_indices])
            prior_mass_correct_exp = np.exp(log_prior_mass_correct)
            posterior_mass_correct_exp_minus_prior = np.subtract(posterior_mass_correct_exp, prior_mass_correct_exp)
            if context == 0:
                posterior_correct_mass_accumulator[run][context] = posterior_mass_correct_exp_minus_prior
            else:
                posterior_correct_mass_accumulator[run][context] = posterior_correct_mass_accumulator[run][context-1] + posterior_mass_correct_exp_minus_prior
    percentile_25 = np.percentile(posterior_correct_mass_accumulator, q=25, axis=0)
    percentile_50 = np.percentile(posterior_correct_mass_accumulator, q=50, axis=0)
    percentile_75 = np.percentile(posterior_correct_mass_accumulator, q=75, axis=0)
    mean = np.mean(posterior_correct_mass_accumulator, axis=0)
    std = np.std(posterior_correct_mass_accumulator, axis=0)
    return np.asarray([percentile_25, percentile_50, percentile_75, mean, std])



def calc_majority_hyp_convergence_time(multi_run_log_posterior_matrix, majority_hyp_indices, theta):
    """
    :param multi_run_log_posterior_matrix: A 3D multi_run_log_posterior_matrix with shape (n_runs, n_contexts, n_hypotheses)
    :param majority_hyp_indices: The indices of the hypothesis type for which we want to calculate the convergence time (i.e. perspective, lexicon or composite)
    :return: min_convergence_time, mean_convergence_time, median_convergence_time, max_convergence_time
    """
    majority_hyp_posterior_matrix = multi_run_log_posterior_matrix[:,:,majority_hyp_indices]
    total_majority_hyp_posterior = logsumexp(majority_hyp_posterior_matrix, axis=2)
    min_majority_hyp_posterior_over_time = np.amin(total_majority_hyp_posterior, axis=0)
    mean_majority_hyp_posterior_over_time = np.mean(total_majority_hyp_posterior, axis=0)
    median_majority_hyp_posterior_over_time = np.median(total_majority_hyp_posterior, axis=0)
    max_majority_hyp_posterior_over_time = np.amax(total_majority_hyp_posterior, axis=0)
    unlogged_min_majority_hyp_posterior_over_time = np.exp(min_majority_hyp_posterior_over_time)
    unlogged_mean_majority_hyp_posterior_over_time = np.exp(mean_majority_hyp_posterior_over_time)
    unlogged_median_majority_hyp_posterior_over_time = np.exp(median_majority_hyp_posterior_over_time)
    unlogged_max_majority_hyp_posterior_over_time = np.exp(max_majority_hyp_posterior_over_time)
    min_convergence_time = float('NaN')
    mean_convergence_time = float('NaN')
    median_convergence_time = float('NaN')
    max_convergence_time = float('NaN')
    for i in range(len(unlogged_min_majority_hyp_posterior_over_time)):
        if unlogged_min_majority_hyp_posterior_over_time[i] > (1.-theta):
            max_convergence_time = i
            break
    for i in range(len(unlogged_mean_majority_hyp_posterior_over_time)):
        if unlogged_mean_majority_hyp_posterior_over_time[i] > (1.-theta):
            mean_convergence_time = i
            break
    for i in range(len(unlogged_median_majority_hyp_posterior_over_time)):
        if unlogged_median_majority_hyp_posterior_over_time[i] > (1.-theta):
            median_convergence_time = i
            break
    for i in range(len(unlogged_max_majority_hyp_posterior_over_time)):
        if unlogged_max_majority_hyp_posterior_over_time[i] > (1.-theta):
            min_convergence_time = i
            break
    return min_convergence_time, mean_convergence_time, median_convergence_time, max_convergence_time


def calc_convergence_time_over_theta_range_percentiles(multi_run_log_posterior_matrix, majority_hyp_indices, theta_range):
    """
    :param multi_run_log_posterior_matrix: A 3D multi_run_log_posterior_matrix with shape (n_runs, n_contexts, n_hypotheses)
    :param majority_hyp_indices: The indices of the hypothesis type for which we want to calculate the convergence time (i.e. perspective, lexicon or composite)
    :param theta_range: The range of different theta-values that we want to be plotted (determined by global parameters theta_start, theta_stop and theta_step
    :return:
    """
    majority_hyp_posterior_matrix = multi_run_log_posterior_matrix[:,:,majority_hyp_indices]
    total_majority_hyp_posterior = logsumexp(majority_hyp_posterior_matrix, axis=2)
    majority_hyp_posterior_over_time_percentile_25 = np.percentile(total_majority_hyp_posterior, 25, axis=0)
    unlogged_majority_hyp_posterior_over_time_percentile_25 = np.exp(majority_hyp_posterior_over_time_percentile_25)
    majority_hyp_posterior_over_time_percentile_50 = np.percentile(total_majority_hyp_posterior, 50, axis=0)
    unlogged_majority_hyp_posterior_over_time_percentile_50 = np.exp(majority_hyp_posterior_over_time_percentile_50)
    majority_hyp_posterior_over_time_percentile_75 = np.percentile(total_majority_hyp_posterior, 75, axis=0)
    unlogged_majority_hyp_posterior_over_time_percentile_75 = np.exp(majority_hyp_posterior_over_time_percentile_75)
    convergence_time_array_percentile_25 = np.zeros(len(theta_range))
    convergence_time_array_percentile_50 = np.zeros(len(theta_range))
    convergence_time_array_percentile_75 = np.zeros(len(theta_range))
    for i in range(len(theta_range)):
        theta_value = theta_range[i]
        convergence_time_percentile_25 = float('NaN')
        convergence_time_percentile_50 = float('NaN')
        convergence_time_percentile_75 = float('NaN')
        for j in range(len(unlogged_majority_hyp_posterior_over_time_percentile_25)):
            if unlogged_majority_hyp_posterior_over_time_percentile_25[j] > (1.-theta_value):
                convergence_time_percentile_25 = j
                break
        convergence_time_array_percentile_25[i] = convergence_time_percentile_25
        for j in range(len(unlogged_majority_hyp_posterior_over_time_percentile_50)):
            if unlogged_majority_hyp_posterior_over_time_percentile_50[j] > (1.-theta_value):
                convergence_time_percentile_50 = j
                break
        convergence_time_array_percentile_50[i] = convergence_time_percentile_50
        for j in range(len(unlogged_majority_hyp_posterior_over_time_percentile_75)):
            if unlogged_majority_hyp_posterior_over_time_percentile_75[j] > (1.-theta_value):
                convergence_time_percentile_75 = j
                break
        convergence_time_array_percentile_75[i] = convergence_time_percentile_75
    return [convergence_time_array_percentile_25, convergence_time_array_percentile_50, convergence_time_array_percentile_75]



def lex_approximation(real_lexicon, lexicon_hyp):
    """
    :param real_lexicon: The real lexicon of which we want to calculate how much it is approximated
    :param lexicon_hyp: The hypothesised lexicon of which we want to calculate how much it approximates real_lexicon
    :return: A value between 0.0 and 1.0 constitutin the proportion of overlap between real_lexicon and lexicon_hyp
    """
    shape = real_lexicon.shape
    max_distance = shape[0]*shape[1]
    distance_matrix = np.subtract(real_lexicon, lexicon_hyp)
    summed_distance = np.sum(np.absolute(distance_matrix))
    prop_distance = summed_distance/max_distance
    approximation = 1.-prop_distance
    return approximation



def calc_lex_approximation_posterior_mass(multi_run_log_posterior_matrix, speaker_lexicon, lexicon_hyps):
    """
    :param multi_run_log_posterior_matrix: A 3D numpy matrix with axis 0 = runs; axis 1 = data_dict points; axis 2 = log posteriors for the different hypotheses
    :param speaker_lexicon: The speaker's lexicon (2D numpy array)
    :param lexicon_hyps: A 3D numpy array containing all the lexicon hypotheses that the learner has considered
    :return: A list containing the first quartile, median and third quartile of posterior probability mass assigned to PARTS of the correct lexicon hypothesis, over time. Percentiles calculated over runs.
    """
    lexicon_hyps_approximation_array = np.zeros(len(lexicon_hyps))
    for i in range(len(lexicon_hyps)):
        lexicon_hyp = lexicon_hyps[i]
        approximation = lex_approximation(speaker_lexicon, lexicon_hyp)
        lexicon_hyps_approximation_array[i] = approximation
    lexicon_hyps_approximation_array_double = np.append(lexicon_hyps_approximation_array, lexicon_hyps_approximation_array)
    unlogged_multi_run_posterior_matrix = np.exp(multi_run_log_posterior_matrix)
    multi_run_approximation_times_posterior_matrix = np.multiply(unlogged_multi_run_posterior_matrix, lexicon_hyps_approximation_array_double)
    summed_approximation_posterior_mass_matrix = np.sum(multi_run_approximation_times_posterior_matrix, axis=2)
    percentile_25 = np.percentile(summed_approximation_posterior_mass_matrix, q=25, axis=0)
    percentile_50 = np.percentile(summed_approximation_posterior_mass_matrix, q=50, axis=0)
    percentile_75 = np.percentile(summed_approximation_posterior_mass_matrix, q=75, axis=0)
    return [percentile_25, percentile_50, percentile_75]


def calc_hypotheses_percentiles(multi_run_log_posterior_matrix):
    """
    :param multi_run_log_posterior_matrix: A 3D numpy matrix with axis 0 = runs, axis 1 = data_dict points and axis 2 = log posteriors for the different composite hypotheses
    :return: A list containing the first quartile, median and third quartile of the posterior probability assigned to the different composite hypothese over time. Percentiles calculated over runs.
    """
    unlogged_multi_run_posterior_matrix = np.exp(multi_run_log_posterior_matrix)
    percentile_posteriors_25 = np.percentile(unlogged_multi_run_posterior_matrix, q=25, axis=0)
    hypotheses_percentiles_25 = np.transpose(percentile_posteriors_25)
    unlogged_multi_run_posterior_matrix = np.exp(multi_run_log_posterior_matrix)
    percentile_posteriors_50 = np.percentile(unlogged_multi_run_posterior_matrix, q=50, axis=0)
    hypotheses_percentiles_50 = np.transpose(percentile_posteriors_50)
    unlogged_multi_run_posterior_matrix = np.exp(multi_run_log_posterior_matrix)
    percentile_posteriors_75 = np.percentile(unlogged_multi_run_posterior_matrix, q=75, axis=0)
    hypotheses_percentiles_75 = np.transpose(percentile_posteriors_75)
    return [hypotheses_percentiles_25, hypotheses_percentiles_50, hypotheses_percentiles_75]


def find_correct_and_mirror_hyp_index(n_meanings, n_signals, lexicon_hyps, speaker_lexicon):
    """
    :param speaker_lexicon: The speaker's lexicon (2D numpy matrix)
    :return: The index of the correct composite hypothesis and the mirror image hypothesis in the learner's hypothesis space
    """
    mirror_lexicon = create_mirror_image_lexicon(n_meanings, n_signals)
    #FIXME: Calling lexicon_hyps as global variable from params_and_run module:
    for i in range(len(lexicon_hyps)):
        if np.array_equal(lexicon_hyps[i], speaker_lexicon):
            correct_lexicon_index = i
        if np.array_equal(lexicon_hyps[i], mirror_lexicon):
            mirror_lexicon_index = i
    correct_hypothesis_index = len(lexicon_hyps)+correct_lexicon_index
    if speaker_lexicon.shape == (n_meanings, n_signals):
        mirror_hypothesis_index = mirror_lexicon_index
    else:
        mirror_hypothesis_index = 'nan'
    return correct_hypothesis_index, mirror_hypothesis_index


def create_lex_posterior_matrix(n_runs, n_meanings, n_signals, hypothesis_space, perspective_hyps, lexicon_hyps, multi_run_log_posterior_matrix):
    """
    :param lexicon_hypotheses: A 3D numpy matrix containing all lexicon hypotheses
    :param multi_run_log_posterior_matrix: A 3D numpy matrix with axis 0 = runs, axis 1 = data_dict points and axis 2 = log posteriors for the different composite hypotheses
    :return: A matrix of lexicon shape that contains for each mapping the posterior probability that is assigned to that mapping on average at the END of each run
    """
    #FIXME: Calling hypothesis_space as global variable from params_and_run module
    final_log_posteriors_matrix = np.zeros((n_runs, len(hypothesis_space)))
    for r in range(len(multi_run_log_posterior_matrix)):
        final_log_posteriors_matrix[r] = multi_run_log_posterior_matrix[r][-1]
    average_final_log_posteriors = np.mean(final_log_posteriors_matrix, axis=0)
    unlogged_average_final_posteriors = np.exp(average_final_log_posteriors)
    normalized_unlogged_average_final_posteriors = np.divide(unlogged_average_final_posteriors, np.sum(unlogged_average_final_posteriors))
    #FIXME: Calling perspective_hyps and lexicon_hyps as global variable from params_and_run module
    posteriors_reshaped = normalized_unlogged_average_final_posteriors.reshape((len(perspective_hyps), len(lexicon_hyps)))
    posteriors_over_lexicons = np.sum(posteriors_reshaped, axis=0)
    matrix = np.zeros((n_meanings, n_signals))
    # TODO: Have a look if this for-loop can be replaced with simple matrix multiplication:
    for i in range(len(lexicon_hyps)):
        lexicon = lexicon_hyps[i]
        lexicon_posterior = posteriors_over_lexicons[i]
        mapping_sum = np.sum(lexicon)
        posterior_divided_over_mappings = np.divide(lexicon_posterior, mapping_sum)
        posterior_divided_over_mappings_matrix = np.multiply(lexicon, posterior_divided_over_mappings)
        matrix = np.add(matrix, posterior_divided_over_mappings_matrix)
    return matrix




#######################################################################################################################

# No. of observation measures for comparing learning rate in different input order conditions:



def calc_no_of_contexts_for_ceiling(correct_hyp_posterior_mass, ceiling):
    n_contexts_per_run = np.array([np.nan for x in range(len(correct_hyp_posterior_mass))])
    for r in range(len(correct_hyp_posterior_mass)):
        if len(np.argwhere(correct_hyp_posterior_mass[r]>ceiling)) != 0:
            n_contexts_per_run[r] = np.argwhere(correct_hyp_posterior_mass[r]>ceiling)[0][0]
    return n_contexts_per_run


#######################################################################################################################

# Lexicon similarity measures for iteration runs

def calc_pop_lexicons_distance(pop_lexicons_matrix):
    pop_lexicons_matrix_flattened = np.zeros((len(pop_lexicons_matrix), (len(pop_lexicons_matrix[0])*len(pop_lexicons_matrix[0][0]))))
    for i in range(len(pop_lexicons_matrix)):
        lexicon = pop_lexicons_matrix[i]
        lexicon = lexicon.flatten()
        pop_lexicons_matrix_flattened[i] = lexicon
    pop_lexicons_distance = pdist(pop_lexicons_matrix_flattened)
    summed_pop_lexicons_distance = np.sum(pop_lexicons_distance)
    return summed_pop_lexicons_distance


def calc_mean_of_final_posteriors(multi_run_log_posterior_matrix):
    final_log_posterior_matrix = multi_run_log_posterior_matrix[:,-1,:]
    unlogged_final_log_posterior_matrix = np.exp(final_log_posterior_matrix)
    mean_final_posteriors = np.mean(unlogged_final_log_posterior_matrix, axis=0)
    return mean_final_posteriors


def calc_std_of_final_posteriors(multi_run_log_posterior_matrix):
    final_log_posterior_matrix = multi_run_log_posterior_matrix[:,-1,:]
    unlogged_final_log_posterior_matrix = np.exp(final_log_posterior_matrix)
    std_final_posteriors = np.std(unlogged_final_log_posterior_matrix, axis=0)
    return std_final_posteriors



def check_convergence(multi_run_over_gens_matrix, window, bandwith):
    convergence_generation_per_run = np.zeros(len(multi_run_over_gens_matrix))
    for r in range(len(multi_run_over_gens_matrix)):
        difference = bandwith+1
        i = 0
        while difference >= bandwith:
            gens_window = multi_run_over_gens_matrix[r][i:window+i]
            max_gens_window = np.amax(gens_window)
            min_gens_window = np.amin(gens_window)
            difference = max_gens_window-min_gens_window
            if difference < bandwith:
                convergence_generation_per_run[r] = window+i
            if window+i == len(multi_run_over_gens_matrix[r]):
                convergence_generation_per_run[r] = "NaN"
                break
            i += 1
    return convergence_generation_per_run