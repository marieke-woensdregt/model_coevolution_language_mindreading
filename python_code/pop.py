__author__ = 'Marieke Woensdregt'


from scipy.misc import logsumexp

import prior
from hypspace import *
from data import Data, FixedContextsData, SpeakerAnnotatedData, convert_dataset_to_signal_counts_per_context
from lex import Lexicon
from context import *




def create_speaker_order_single_generation(population, speaker_order_type, n_contexts, first_input_stage_ratio):
    if speaker_order_type == 'random':
        speaker_order = np.random.choice(population.population, size=n_contexts, replace=True)
    elif speaker_order_type == 'same_first':
        first_stage = np.array([population.population[0] for x in range(int(n_contexts*first_input_stage_ratio))])
        second_stage = np.random.choice(population.population, size=(n_contexts-len(first_stage)), replace=True)
        speaker_order = np.append(first_stage, second_stage)
    elif speaker_order_type == 'opp_first':
        first_stage = np.array([population.population[1] for x in range(int(n_contexts*first_input_stage_ratio))])
        second_stage = np.random.choice(population.population, size=(n_contexts-len(first_stage)), replace=True)
        speaker_order = np.append(first_stage, second_stage)
    elif speaker_order_type == 'random_equal':
        first_stage = np.array([population.population[0] for x in range(int(n_contexts*first_input_stage_ratio))])
        second_stage = np.array([population.population[1] for x in range(int(n_contexts-len(first_stage)))])
        speaker_order = np.append(first_stage, second_stage)
        np.random.shuffle(speaker_order)
    elif speaker_order_type == 'same_first_equal':
        first_stage = np.array([population.population[0] for x in range(int(n_contexts*first_input_stage_ratio))])
        second_stage = np.array([population.population[1] for x in range(int(n_contexts-len(first_stage)))])
        speaker_order = np.append(first_stage, second_stage)
    elif speaker_order_type == 'opp_first_equal':
        first_stage = np.array([population.population[1] for x in range(int(n_contexts*first_input_stage_ratio))])
        second_stage = np.array([population.population[0] for x in range(int(n_contexts-len(first_stage)))])
        speaker_order = np.append(first_stage, second_stage)
    return speaker_order



def create_speaker_order_iteration(population, selection_type, parent_probs, parent_type, n_contexts):
    if parent_type == 'sng_teacher' and selection_type == 'none':
        speaker = np.random.choice(population.population)
        speaker_order = np.array([speaker for x in range(n_contexts)])
    elif parent_type == 'multi_teacher' and selection_type == 'none':
        speaker_order = np.random.choice(population.population, size=n_contexts, replace=True)
    elif parent_type == 'sng_teacher' and selection_type == 'p_taking' or selection_type == 'l_learning' or selection_type == 'ca_with_parent':
        speaker = np.random.choice(population.population, p=parent_probs)
        speaker_order = np.array([speaker for x in range(n_contexts)])
    elif parent_type == 'multi_teacher' and selection_type == 'p_taking' or selection_type == 'l_learning' or selection_type == 'ca_with_parent':
        speaker_order = np.random.choice(population.population, size=n_contexts, replace=True, p=parent_probs)
    return speaker_order



class Agent(object):
    """
    An agent object contains the agent's priors, hypothesis space, perspective and lexicon, and the necessary methods to do production, reception and learning (c.e. updating its posteriors and lexicon through Bayesian inference).
    """
    def __init__(self, perspective_hyps, lexicon_hyps, log_priors, log_posteriors, perspective, sal_alpha, lexicon, learning_type, pragmatic_level=None):
        """
        :param perspective_hyps: The agent's perspective hypotheses
        :param lexicon_hyps: The agent's lexicon hypotheses
        :param log_priors: The agent's prior probability distribution over all composite hypotheses IN LOG SPACE (1D numpy array)
        :param log_posteriors: The agent's posterior probability distribution over all composite hypotheses IN LOG SPACE (1D numpy array)
        :param perspective: The agent's OWN perspective (float)
        :param lexicon: The agent's OWN lexicon (2D numpy array)
        :param learning_type: The type of learning that the agent does. Either 'map' or 'sample'
        :return: Creates an agent with initial perspective and lexicon
        """
        self.perspective_hyps = perspective_hyps
        self.lexicon_hyps = lexicon_hyps
        self.hypothesis_space = list_hypothesis_space(perspective_hyps, lexicon_hyps)
        self.log_priors = log_priors
        self.log_posteriors = log_posteriors
        self.perspective = perspective
        self.sal_alpha = sal_alpha
        self.lexicon = lexicon
        if pragmatic_level != None:
            self.pragmatic_level = pragmatic_level
        else:
            self.pragmatic_level = 'literal'
        self.optimality_alpha = 1.0
        self.learning_type = learning_type
        self.id = 0

    def calc_distances(self, agent_perspective, context):
        """
        :param agent_perspective: The perspective of the agent for whom we want to calculate the distances (can be self.perspective or that of another agent) (float)
        :param context: The context for which we want to calculate the distances to agent_perspective (1D numpy array)
        :return: The distances between the meanings and agent_perspective in the context (1D numpy array)
        """
        distances = np.subtract(agent_perspective, context)
        # elif self.context_type == 'absolute':
        #     distances = np.subtract(1., context)
        abs_distances = np.absolute(distances)
        return abs_distances

    # TODO: The two methods below call previous methods from within the class. Is this a problem?
    def calc_saliencies(self, agent_perspective, context):
        """
        :param agent_perspective: The perspective of the agent for whom we want to calculate the saliencies (can be self.perspective or that of another agent) (float)
        :param context: The context for which we want to calculate the saliencies from agent_perspective (1D numpy array)
        :return: The saliency of the meanings based on distance from agent_perspective (1D numpy array)
        """
        distances = self.calc_distances(agent_perspective, context)
        inverse_distances = np.subtract(1, distances)
        saliencies = np.power(inverse_distances, self.sal_alpha)
        return saliencies

    def calc_intention(self, agent_perspective, context):
        """
        :param agent_perspective: The perspective of the agent for whom we want to calculate the intention distribution (can be self.perspective or that of another agent) (float)
        :param context: The context for which we want to calculate the intention distribution given agent_perspective (1D numpy array)
        :return: The intention distribution (c.e. probability of choosing meanings as topic meaning) given agent_perspective and context (1D numpy array) (This is simply a normalized version of the saliency array.)
        """
        saliencies = self.calc_saliencies(agent_perspective, context)
        intention = np.divide(saliencies, np.sum(saliencies))
        return intention

    def pick_topics(self, n_meanings, speaker_intention, n_utterances):
        """
        :param n_meanings: The number of meanings
        :param speaker_intention: (Hypothesized) intention distribution of the speaker (1D numpy array containing probabilities)
        :param n_utterances: Global variable determining the number of utterances produced per context (float)
        :return: A 1D numpy array of topic meanings chosen with the probabilities in speaker_intention.
        """
        topics = np.random.choice(np.arange(n_meanings), n_utterances, p=speaker_intention)
        return topics

    def calc_signal_probs(self, lexicon, meaning, error):
        """
        :param lexicon: (Hypothesized) lexicon for which we want to calculate the signal probabilities given a meaning
        :param meaning: The meaning (c.e. index in range 0, n_meanings) for which we want to calculate the signal probabilities
        :return: A 1D numpy array containing the probabilities with which each signal will be uttered for parameter meaning given parameter lexicon
        """
        meaning_row = lexicon[meaning]
        new_signal_probs = np.zeros_like(meaning_row)
        associated_signals_indices = np.where(meaning_row==1.)[0]
        unassociated_signals_indices = np.where(meaning_row==0.)[0]
        new_signal_probs[associated_signals_indices] = np.divide((1.-error), len(associated_signals_indices))
        new_signal_probs[unassociated_signals_indices] = np.divide(error, len(unassociated_signals_indices))
        if len(unassociated_signals_indices) == 0:
            new_signal_probs[associated_signals_indices] += np.divide(error, len(associated_signals_indices))
        return new_signal_probs

    def calc_meaning_probs(self, lexicon, signal, error):
        """
        :param lexicon: (Hypothesized) lexicon for which we want to calculate the meaning probabilities given a particular signal
        :param signal: The signal (c.e. index in range 0, n_signals) for which we want to calculate the meaning probabilities
        :return: A 1D numpy array containing the probabilities that each of the meanings is intended given signal and lexicon
        """
        lexicon_transposed = lexicon.T
        signal_column = lexicon_transposed[int(signal)]
        new_meaning_probs = np.zeros_like(signal_column)
        associated_meanings_indices = np.where(signal_column==1.)[0]
        unassociated_meanings_indices = np.where(signal_column==0.)[0]
        new_meaning_probs[associated_meanings_indices] = np.divide((1.-error), len(associated_meanings_indices))
        new_meaning_probs[unassociated_meanings_indices] = np.divide(error, len(unassociated_meanings_indices))
        if len(unassociated_meanings_indices) == 0:
            new_meaning_probs[associated_meanings_indices] += np.divide(error, len(associated_meanings_indices))
        return new_meaning_probs

    def produce_signal(self, n_signals, topic, error):
        """
        :param n_signals: The number of signals
        :param topic: The index of the topic meaning for which we want to produce a signal
        :return: A signal for topic meaning, chosen probabilistically based on self.lexicon
        """
        signal_probs = self.calc_signal_probs(self.lexicon.lexicon, topic, error)
        signal = np.random.choice(np.arange(n_signals), 1, p=signal_probs)
        return signal

    def interpret_signal(self, n_meanings, topic_probs, signal, error):
        """
        :param n_meanings: The number of meanings
        :param signal: The index of the signal that we want to interpret
        :return: An interpreted meaning for the received signal, chosen probabilistically based on self.lexicon
        """
        meaning_probs = self.calc_meaning_probs(self.lexicon.lexicon, signal, error)
        meaning_probs_weighted = np.multiply(meaning_probs, topic_probs)
        meaning_probs_weighted_normalized = np.divide(meaning_probs_weighted, np.sum(meaning_probs_weighted))
        interpretation = np.random.choice(np.arange(n_meanings), 1, p=meaning_probs_weighted_normalized)
        return interpretation


    def produce_data(self, n_meanings, n_signals, context_matrix, n_utterances, error, extra_error=None):
        """
        :param n_meanings: The number of meanings
        :param context_matrix: A 2D numpy array of contexts
        :param n_utterances: Global variable determining the number of utterances produced per context (float)
        :return: A data_dict object, consisting of two attributes: 1) context matrix, 2) a corresponding matrix of utterance arrays of length n_utterances
        """
        topic_matrix = np.zeros((len(context_matrix), n_utterances))
        utterance_matrix = np.zeros((len(context_matrix), n_utterances))
        for i in range(len(context_matrix)):
            context = context_matrix[i]
            intention = self.calc_intention(self.perspective, context)
            topics = self.pick_topics(n_meanings, intention, n_utterances)
            for j in range(len(topics)):
                topic = topics[j]
                topic_matrix[i][j] = topic
                utterance_matrix[i][j] = self.produce_signal(n_signals, topic, error)
        data = Data(context_matrix, topic_matrix, utterance_matrix)
        return data


    def calc_literal_speaker(self, context, lexicon, error, perspective, alpha):
        topic_dist = self.calc_intention(perspective, context)
        literal_speaker = np.zeros((lexicon.shape[1], lexicon.shape[0]))
        for m in range(lexicon.shape[0]):
            signal_probs = self.calc_signal_probs(lexicon, m, error)
            literal_speaker[m] = signal_probs
        topic_dist_reshaped = topic_dist.reshape(topic_dist.shape[0], -1)
        literal_speaker_times_topic_probs = np.multiply(literal_speaker, topic_dist_reshaped)
        literal_speaker_normalized = np.divide(literal_speaker_times_topic_probs, np.sum(literal_speaker_times_topic_probs))
        return literal_speaker_normalized

    def calc_pragmatic_speaker(self, listener_rec_probs):
        listener_utilities = np.log(listener_rec_probs)
        pragmatic_speaker = np.zeros((listener_rec_probs.shape[1], listener_rec_probs.shape[0]))
        signal_probs_pragmatic_speaker = np.exp(np.multiply(listener_utilities, self.optimality_alpha))
        for m in range(len(signal_probs_pragmatic_speaker.T)):
            pragmatic_speaker[m] = np.divide(signal_probs_pragmatic_speaker.T[m], np.sum(signal_probs_pragmatic_speaker.T[m]))
        return pragmatic_speaker

    def calc_pragmatic_listener_rec_probs(self, speaker_prod_probs):
        speaker_normalized = np.divide(speaker_prod_probs, np.sum(speaker_prod_probs))
        pragmatic_listener_rec_probs = np.zeros((speaker_normalized.shape[0], speaker_normalized.shape[1]))
        for s in range(len(speaker_prod_probs.T)):
            pragmatic_listener_rec_probs[s] = np.divide(speaker_normalized.T[s], np.sum(speaker_normalized.T[s]))
        return pragmatic_listener_rec_probs

    def calc_pragmatic_listener(self, context, speaker_lex, speaker_p, speaker_type, error, sal_alpha):
        speaker_prod_probs = self.calc_literal_speaker(context, speaker_lex, error, speaker_p, sal_alpha)
        pragmatic_listener = self.calc_pragmatic_listener_rec_probs(speaker_prod_probs)
        if speaker_type == 'literal':
            return pragmatic_listener
        elif speaker_type == 'prag':
            speaker_prod_probs = self.calc_pragmatic_speaker(pragmatic_listener)


            ### BELOW IS THE NEW STUFF ADDED ON 12 DEC 2018:
            topic_dist = self.calc_intention(speaker_p, context)
            topic_dist_reshaped = topic_dist.reshape(topic_dist.shape[0], -1)
            speaker_times_topic_probs = np.exp(np.add(np.log(speaker_prod_probs), np.log(topic_dist_reshaped)))
            pragmatic_speaker_normalized = np.divide(speaker_times_topic_probs, np.sum(speaker_times_topic_probs))


            pragmatic_listener_old = self.calc_pragmatic_listener_rec_probs(speaker_prod_probs)

            ### BELOW IS THE NEW STUFF ADDED ON 12 DEC 2018:
            pragmatic_listener = self.calc_pragmatic_listener_rec_probs(pragmatic_speaker_normalized)

            return pragmatic_listener




    def calc_topic_probs(self, comprehension_type, context, perspective=None):
        if comprehension_type == 'lex_only':
            topic_probs = np.array([(1./len(context)) for x in range(len(context))])
        elif comprehension_type == 'lex_n_context':
            topic_probs_per_perspective_matrix = np.zeros((len(self.perspective_hyps), len(context)))
            for p in range(len(self.perspective_hyps)):
                perspective_hyp = self.perspective_hyps[p]
                topic_probs_per_perspective = self.calc_intention(perspective_hyp, context)
                topic_probs_per_perspective_matrix[p] = topic_probs_per_perspective
            topic_probs_averaged = np.mean(topic_probs_per_perspective_matrix, axis=0)
            topic_probs = np.divide(topic_probs_averaged, np.sum(topic_probs_averaged))
        elif comprehension_type == 'lex_n_p':
            topic_probs = self.calc_intention(perspective, context)
        return topic_probs


    def interpret_data(self, n_meanings, data, comprehension_type, error, perspective=None):
        interpretation_matrix = np.zeros_like(data.utterances)
        for i in range(len(data.utterances)):
            context = data.contexts[i]
            for j in range(len(data.utterances[0])):
                topic_probs = self.calc_topic_probs(comprehension_type, context, perspective)
                utterance = data.utterances[i][j]
                interpretation = self.interpret_signal(n_meanings, topic_probs, utterance, error)
                interpretation_matrix[i][j] = interpretation
        return interpretation_matrix



    def inference(self, n_contexts, n_utterances, data, error):
        """
        :param data: A data_dict object with context matrix and corresponding utterances matrix
        :update: Updates self.log_posteriors after going through the whole set of data_dict
        :return: A 2D numpy array with each row representing an update in the log_posteriors over time, where there is one row for each utterance, and the columns represent the different composite hypotheses.
        """
        posteriors_per_data_point_matrix = np.zeros((((n_contexts*n_utterances)+1), len(self.hypothesis_space))) #+1 is added because the original prior distribution (c.e. the distribution of belief before making any observations) will be the first posterior to be saved to the matrix
        log_posteriors_per_data_point_matrix = np.log(posteriors_per_data_point_matrix)
        log_posteriors_per_data_point_matrix[0] = self.log_posteriors
        # 1) For each hypothesis in the hypothesis space:
        for i in range(len(self.hypothesis_space)):
            composite_hypothesis = self.hypothesis_space[i]
            persp_hyp_index = composite_hypothesis[0]
            lex_hyp_index = composite_hypothesis[1]
            perspective_hyp = self.perspective_hyps[persp_hyp_index]
            lexicon_hyp = self.lexicon_hyps[lex_hyp_index]
            counter = 1 # this starts at 1 since the first log_posteriors written to the matrix is the prior
            # 2) For each context in the data_dict set:
            for j in range(len(data.contexts)):
                context = data.contexts[j]
                utterances = data.utterances[j]
                speaker_intention = self.calc_intention(perspective_hyp, context)
                # 3) For each utterance in the context:
                for k in range(len(utterances)):
                    utterance = utterances[k]
                    initial_utterance_likelihood = 0.0 # This has to be zero because the probability values for the different meanings are ADDED rather than multiplied
                    utterance_log_likelihood = np.log(initial_utterance_likelihood)
                    utterance_log_likelihood = np.nan_to_num(utterance_log_likelihood) # turns -inf into a very large negative number
                    # 4) For each meaning in the speaker's intention distribution:
                    for l in range(len(speaker_intention)):
                        meaning = l

                        meaning_prob = speaker_intention[l]

                        log_meaning_prob = np.log(meaning_prob)
                        log_meaning_prob = np.nan_to_num(log_meaning_prob) # turns -inf into a very large negative number

                        #TODO: Check whether the signal_probs below are correct!
                        signal_probs = self.calc_signal_probs(lexicon_hyp, meaning, error)
                        log_signal_probs = np.log(signal_probs)
                        log_signal_probs = np.nan_to_num(log_signal_probs) # turns -inf into a very large negative number
                        # 5) Multiply the probability of that meaning being the intended meaning and the probability of the utterance signal being produced for that meaning (multiplication = addition in logspace):

                        log_likelihood_for_meaning = log_meaning_prob+log_signal_probs[int(utterance)]
                        # 6) And add this up for all meanings in the intention distribution to arrive at the full likelihood of the utterance:
                        utterance_log_likelihood = np.logaddexp(utterance_log_likelihood, log_likelihood_for_meaning)
                    # 7) Update the old posterior for this hypothesis by multiplying it with the utterance likelihood calculated at step 6
                    self.log_posteriors[i] += utterance_log_likelihood
                    log_posteriors_per_data_point_matrix[counter][i] = self.log_posteriors[i]
                    counter += 1
        # 8.c) After updating the posteriors for all hypotheses for the full data_dict set, normalize each posterior distribution that has been saved into the log_posteriors_per_data_point_matrix:


        # print ''
        # print ''
        # print 'This is the inference() method of the Agent class'
        #
        # #TODO: See below for method using logsumexp:
        # sums_matrix = logsumexp(log_posteriors_per_data_point_matrix, axis=1) # Creates a matrix that contains the sum over each row in the log_posteriors_per_data_point_matrix
        #
        # print ''
        # print ''
        # print ''
        # print "sums_matrix USING logsumexp is:"
        # print sums_matrix

        unlogged_posteriors_per_data_point_matrix = np.exp(log_posteriors_per_data_point_matrix)
        sums_matrix_new = np.sum(unlogged_posteriors_per_data_point_matrix, axis=1)
        sums_matrix = np.log(sums_matrix_new)

        # print ''
        # print ''
        # print ''
        # print "sums_matrix AVOIDING logsumexp is:"
        # print sums_matrix


        sums_matrix = sums_matrix[:, np.newaxis] # Reshapes the sums matrix so that it can be subtracted from the log_posteriors_per_data_point_matrix (subtraction in logspace = division in normal space)

        # print ''
        # print ''
        # print ''
        # print "sums_matrix AFTER RESHAPING is:"
        # print sums_matrix

        normalized_log_posteriors_per_data_point_matrix = np.subtract(log_posteriors_per_data_point_matrix, sums_matrix)

        # 9) Normalize the current posterior distribution:

        normalized_log_posteriors = np.subtract(self.log_posteriors, logsumexp(self.log_posteriors))

        # 10) And update the agent's attribute self.log_posteriors to the newly calculated log_posteriors:
        self.log_posteriors = normalized_log_posteriors
        return normalized_log_posteriors_per_data_point_matrix


    def log_likelihoods_on_signal_counts_data(self, signal_counts_data, error):
    # def log_likelihoods_on_signal_counts_data(self, signal_counts_data,dataset_array_pickle_file_name, log_likelihood_pickle_file_name, error):
        """
        This method uses memoization. It accesses and updates a pickle file ("log_likelihood_pickle_file") containing a 2D numpy array that stores an array of likelihoods for each possible dataset. This array has the number of possible datasets as the number of rows, and the length of the hypothesis space as the number of columns. It is indexed according to the order of another pickle file that has all the possible datasets stored in the signal_counts_per_context format (given n_helpful_contexts, n_observations and n_signals). In the log_likelihood_pickle_file, the log_likelihoods (per hypothesis) are stored and updated dynamically for each new dataset that is encountered. If the dataset has been observed before by another learner with the same hypothesis space, the likelihoods are simply loaded from the pickle file. If the dataset has not been encountered before, the likelihoods are calculated and written to the pickle file.
        """
        signal_counts_per_context_matrix = signal_counts_data.signal_counts_per_context_matrix
        # signal_counts_per_context_matrix_flattened = signal_counts_per_context_matrix.flatten()
        # dataset_array = np.asarray(pickle.load(open(dataset_array_pickle_file_name, "rb")))
        # dataset_index = np.argwhere((dataset_array == signal_counts_per_context_matrix_flattened).all(axis=1))[0][0]
        # log_likelihood_array = pickle.load(open(log_likelihood_pickle_file_name, "rb"))
        # if np.isnan(log_likelihood_array[dataset_index][0]):
        log_likelihoods = np.zeros(len(self.hypothesis_space))  # 0.0 in log-space is 1.0 in probability space. We have to initialise the likelihoods as 1's, because from here this number will be multiplied with the actual likelihoods of the observations.
        for h in range(len(self.hypothesis_space)):
            composite_hypothesis = self.hypothesis_space[h]
            persp_hyp_index = composite_hypothesis[0]
            lex_hyp_index = composite_hypothesis[1]
            perspective_hyp = self.perspective_hyps[persp_hyp_index]
            lexicon_hyp = self.lexicon_hyps[lex_hyp_index]
            for c in range(len(signal_counts_data.helpful_contexts)):
                context = signal_counts_data.helpful_contexts[c]
                speaker_intention = self.calc_intention(perspective_hyp, context)
                log_speaker_intention = np.log(speaker_intention)
                signal_counts = signal_counts_per_context_matrix[c]
                for signal in range(len(signal_counts)):
                    if signal_counts[signal] > 0.:
                        signal_likelihood = 0.
                        signal_log_likelihood = np.log(signal_likelihood)
                        for meaning in range(len(speaker_intention)):
                            signal_probs = self.calc_signal_probs(lexicon_hyp, meaning, error)
                            log_signal_probs = np.log(signal_probs)
                            # 5) Multiply the probability of that meaning being the intended meaning and the probability of the utterance signal being produced for that meaning (multiplication = addition in logspace):
                            log_likelihood_for_meaning = np.add(log_speaker_intention[meaning], log_signal_probs[signal])
                            # 6) And add this up for all meanings in the intention distribution to arrive at the full likelihood of the utterance:
                            signal_log_likelihood = np.logaddexp(signal_log_likelihood, log_likelihood_for_meaning)
                        for i in range(signal_counts[signal].astype(int)):
                            log_likelihoods[h] += signal_log_likelihood
        # log_likelihood_array[dataset_index] = log_likelihoods
        # pickle.dump(log_likelihood_array, open(log_likelihood_pickle_file_name, 'wb'))
        # else:
        #     log_likelihoods = log_likelihood_array[dataset_index]
        log_likelihoods = np.nan_to_num(log_likelihoods)
        return log_likelihoods


    def inference_on_signal_counts_data(self, signal_counts_data, error):
    # def inference_on_signal_counts_data(self, signal_counts_data, dataset_array_pickle_file_name, log_likelihood_pickle_file_name, error):
        log_likelihoods = self.log_likelihoods_on_signal_counts_data(signal_counts_data, error)
        self.log_posteriors += log_likelihoods
        self.log_posteriors = np.nan_to_num(self.log_posteriors)
        self.log_posteriors = np.subtract(self.log_posteriors, logsumexp(self.log_posteriors))
        return self.log_posteriors



    def pick_winning_hyp(self):
        """
        :return: The index of a winning composite hypothesis based on the agent's current log_posteriors attribute. Winning hypothesis is chosen either with 'map' (maximum a posteriori) learning method or 'sample' learning method, depending on the agent' agent.learning type.
        """
        if self.learning_type == 'map':
            winning_hyp_index = np.argmax(self.log_posteriors)
            return winning_hyp_index
        elif self.learning_type == 'sample':
            winning_hyp_index = np.random.choice(np.arange(len(self.log_posteriors)), p=np.exp(self.log_posteriors))
            return winning_hyp_index

            #TODO: See below for hand-implemented version of sampling from SimLang code:
            # r = np.log(np.random.rand())
            # accumulator = self.log_posteriors[0]
            # for hyp_index in range(len(self.log_posteriors)):
            #     if r < accumulator:
            #         return hyp_index
            #     accumulator = logsumexp([accumulator, self.log_posteriors[hyp_index+1]])

    def update_lexicon(self):
        """
        Updates the agent's lexicon by taking its current log_posteriors, picking the winning hypothesis by use of self.learning_type, and setting the agents lexicon to that of the winning hypothesis
        """
        lexicon_per_posterior_array = self.lexicon_hyps
        # The following for-loop creates an array of which the lexicons on the indices match the indices of the hypothesis_space (c.e. the full set of lexicons repeated for as many times as there are different perspective hypotheses
        for i in range((len(self.perspective_hyps)-1)):
            lexicon_per_posterior_array = np.append(lexicon_per_posterior_array, self.lexicon_hyps, axis=0)
        winning_hyp_index = self.pick_winning_hyp()
        winning_hyp = self.hypothesis_space[winning_hyp_index]
        winning_lex_hyp_index = winning_hyp[1]
        self.lexicon.lexicon = lexicon_per_posterior_array[winning_lex_hyp_index]
        return winning_hyp_index, winning_lex_hyp_index

    def print_agent(self):
        """
        Prints the agent's attributes. Doesn't return anything
        """
        print "This agent's id is:"
        print self.id
        print "This agent's perspective is:"
        print self.perspective
        print "This agent's lexicon is:"
        print self.lexicon.lexicon
        print "This agent's learning_type is:"
        print self.learning_type
        # print "This agent's hypothesis space is:"
        # print self.hypothesis_space
        # print "The size of this agent's hypothesis space is:"
        # print self.hypothesis_space.shape
        # print "This agent's prior probability distribution (exponentiated) is:"
        # print np.exp(self.log_priors)
        # print "This agent's posterior probability distribution (exponentiated) is:"
        # print np.exp(self.log_posteriors)





class PragmaticAgent(Agent):
    """
    This is an agent that can do higher levels of pragmatic reasoning. In the role of speaker they will reason about a listener (either literal or perspective-taking), and in the role of listener/learner they will reason about that pragmatic speaker. This class adds some new methods for predicting the behaviour of different levels of speaker and listeners. It also overwrites the 'calc_signal_probs', 'calc_meaning_probs' and 'interpret_signal' methods, for those have to be based on these higher-level pragmatic speakers and listeners. All the other methods (for picking a topic, producing a data set, inferring a lex and p hyp based on data and interpreting data) all build on these three basic methods, so do not need to be changed.
    """

    def __init__(self, perspective_hyps, lexicon_hyps, log_priors, log_posteriors, perspective, sal_alpha, lexicon, learning_type, speaker_type, listener_type, optimality_alpha, extra_error):
        """
        This initializes the same as for the Agent superclass, except that a pragmatic agent has a extra attribute 'listener_type' which specifies whether when they're in the role of speaker they will reason about a literal listener or a perspective-taking listener.
        :param perspective_hyps:
        :param lexicon_hyps:
        :param log_priors:
        :param log_posteriors:
        :param perspective:
        :param lexicon:
        :param learning_type:
        :param n_speakers:
        :return:
        """
        super(PragmaticAgent, self).__init__(perspective_hyps, lexicon_hyps, log_priors, log_posteriors, perspective, sal_alpha, lexicon, learning_type)
        self.speaker_type = speaker_type
        self.listener_type = listener_type
        self.pragmatic_level = listener_type
        self.optimality_alpha = optimality_alpha
        self.extra_error = extra_error

    def calc_literal_listener(self, lexicon, error):
        literal_listener = np.zeros((lexicon.shape[0], lexicon.shape[1]))
        for s in range(lexicon.shape[0]):
            meaning_probs = self.calc_meaning_probs(lexicon, s, error)
            literal_listener[s] = meaning_probs
        return literal_listener


    def calc_signal_probs_pragmatic(self, context, speaker_type, s_perspective, sal_alpha, topic, lexicon, error):
        if speaker_type == 'literal' or speaker_type == 'perspective-taking':
            s_prod_probs = self.calc_literal_speaker(context, lexicon, error, s_perspective, sal_alpha)
        elif speaker_type == 'prag':
            l_rec_probs_according_to_speaker = self.calc_pragmatic_listener(context, lexicon, s_perspective, 'literal', error, sal_alpha)
            s_prod_probs = self.calc_pragmatic_speaker(l_rec_probs_according_to_speaker)
            if self.extra_error == True:
                for m in range(len(s_prod_probs)):
                    meaning_row = s_prod_probs[m]
                    addition_matrix = np.zeros((len(meaning_row), len(meaning_row)))
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
        s_prod_probs_topic = np.divide(s_prod_probs[topic].flatten(), np.sum(s_prod_probs[topic].flatten()))
        return s_prod_probs_topic


    def calc_meaning_probs_pragmatic(self, context, listener_type, s_p_hyp, sal_alpha, signal, lexicon, error):
        if listener_type == 'perspective-taking':
            l_rec_probs = self.calc_pragmatic_listener(context, lexicon, s_p_hyp, 'literal', error, sal_alpha)
        elif listener_type == 'prag':
            l_rec_probs = self.calc_pragmatic_listener(context, lexicon, s_p_hyp, 'prag', error, sal_alpha)
        l_rec_probs_signal = l_rec_probs[signal].flatten()
        return l_rec_probs_signal


    def produce_signal_without_error(self, context, n_signals, topic, error, speaker_type, s_perspective):
        if speaker_type == 'literal' or speaker_type == 'perspective-taking':
            signal_probs = self.calc_signal_probs(self.lexicon.lexicon, topic, error)
        elif speaker_type == 'prag':
            signal_probs = self.calc_signal_probs_pragmatic(context, speaker_type, s_perspective, self.sal_alpha, topic, self.lexicon.lexicon, error)
        signal = np.random.choice(np.arange(n_signals), 1, p=signal_probs)
        return signal


    def produce_signal(self, context, n_signals, topic, error, speaker_type, s_perspective):
        signal_without_error = self.produce_signal_without_error(context, n_signals, topic, error, speaker_type, s_perspective)
        remaining_signals = range(n_signals)
        remaining_signals.remove(signal_without_error[0])
        random_float = np.random.uniform()
        if random_float < error:
            signal = np.random.choice(np.array(remaining_signals), 1)
        else:
            signal = signal_without_error
        return signal


    def interpret_signal(self, context, n_meanings, topic_probs, signal, error, s_p_hyp):
        meaning_probs = self.calc_meaning_probs_pragmatic(context, self.listener_type, s_p_hyp, self.sal_alpha, signal, self.lexicon.lexicon, error)
        interpretation = np.random.choice(np.arange(n_meanings), 1, p=meaning_probs)
        return interpretation

    def produce_data(self, n_meanings, n_signals, context_matrix, n_utterances, error, extra_error):
        """
        :param n_meanings: The number of meanings
        :param context_matrix: A 2D numpy array of contexts
        :param n_utterances: Global variable determining the number of utterances produced per context (float)
        :return: A data_dict object, consisting of two attributes: 1) context matrix, 2) a corresponding matrix of utterance arrays of length n_utterances
        """
        topic_matrix = np.zeros((len(context_matrix), n_utterances))
        utterance_matrix = np.zeros((len(context_matrix), n_utterances))
        for i in range(len(context_matrix)):
            context = context_matrix[i]
            intention = self.calc_intention(self.perspective, context)
            topics = self.pick_topics(n_meanings, intention, n_utterances)
            for j in range(len(topics)):
                topic = topics[j]
                topic_matrix[i][j] = topic
                if self.extra_error == True:
                    utterance_matrix[i][j] = self.produce_signal(context, n_signals, topic, error, self.speaker_type, self.perspective)
                else:
                    utterance_matrix[i][j] = self.produce_signal_without_error(context, n_signals, topic, error, self.speaker_type, self.perspective)
        data = Data(context_matrix, topic_matrix, utterance_matrix)
        return data


    def inference(self, n_contexts, n_utterances, data, error):
        """
        :param data: A data_dict object with context matrix and corresponding utterances matrix
        :update: Updates self.log_posteriors after going through the whole set of data_dict
        :return: A 2D numpy array with each row representing an update in the log_posteriors over time, where there is one row for each utterance, and the columns represent the different composite hypotheses.
        """
        posteriors_per_data_point_matrix = np.zeros((((n_contexts*n_utterances)+1), len(self.hypothesis_space))) #+1 is added because the original prior distribution (c.e. the distribution of belief before making any observations) will be the first posterior to be saved to the matrix
        log_posteriors_per_data_point_matrix = np.log(posteriors_per_data_point_matrix)
        log_posteriors_per_data_point_matrix[0] = self.log_posteriors
        # 1) For each hypothesis in the hypothesis space:
        for i in range(len(self.hypothesis_space)):
            composite_hypothesis = self.hypothesis_space[i]
            persp_hyp_index = composite_hypothesis[0]
            lex_hyp_index = composite_hypothesis[1]
            perspective_hyp = self.perspective_hyps[persp_hyp_index]
            lexicon_hyp = self.lexicon_hyps[lex_hyp_index]
            counter = 1  # this starts at 1 since the first log_posteriors written to the matrix is the prior
            # 2) For each context in the data_dict set:
            for j in range(len(data.contexts)):
                context = data.contexts[j]
                utterances = data.utterances[j]
                speaker_intention = self.calc_intention(perspective_hyp, context)
                log_speaker_intention = np.log(speaker_intention)
                # 3) For each utterance in the context:
                for k in range(len(utterances)):
                    utterance = utterances[k]
                    initial_utterance_likelihood = 0.0 # This has to be zero because the probability values for the different meanings are ADDED rather than multiplied
                    utterance_log_likelihood = np.log(initial_utterance_likelihood)
                    utterance_log_likelihood = np.nan_to_num(utterance_log_likelihood) # turns -inf into a very large negative number
                    # 4) For each meaning in the speaker's intention distribution:
                    for l in range(len(speaker_intention)):
                        meaning = l
                        signal_probs = self.calc_signal_probs_pragmatic(context, self.speaker_type, perspective_hyp, self.sal_alpha, meaning, lexicon_hyp, error)
                        log_signal_probs = np.log(signal_probs)
                        log_signal_probs = np.nan_to_num(log_signal_probs) # turns -inf into a very large negative number
                        # 5) Multiply the probability of that meaning being the intended meaning and the probability of the utterance signal being produced for that meaning (multiplication = addition in logspace):
                        log_likelihood_for_meaning = np.add(log_speaker_intention[int(l)], log_signal_probs[int(utterance)])
                        # 6) And add this up for all meanings in the intention distribution to arrive at the full likelihood of the utterance:
                        utterance_log_likelihood = np.logaddexp(utterance_log_likelihood, log_likelihood_for_meaning)
                    # 7) Update the old posterior for this hypothesis by multiplying it with the utterance likelihood calculated at step 6
                    self.log_posteriors[i] += utterance_log_likelihood
                    log_posteriors_per_data_point_matrix[counter][i] = self.log_posteriors[i]
                    counter += 1
        # 8.c) After updating the posteriors for all hypotheses for the full data_dict set, normalize each posterior distribution that has been saved into the log_posteriors_per_data_point_matrix:


        # print ''
        # print ''
        # print 'This is the inference() method of the PragmaticAgent class:'
        #
        # #TODO: See below for method using logsumexp:
        # sums_matrix = logsumexp(log_posteriors_per_data_point_matrix, axis=1) # Creates a matrix that contains the sum over each row in the log_posteriors_per_data_point_matrix

        # print ''
        # print ''
        # print 'This is the sums_matrix using logsumexp'
        # print sums_matrix

        unlogged_posteriors_per_data_point_matrix = np.exp(log_posteriors_per_data_point_matrix)
        sums_matrix_new = np.sum(unlogged_posteriors_per_data_point_matrix, axis=1)
        sums_matrix = np.log(sums_matrix_new)

        # print ''
        # print ''
        # print 'This is the sums_matrix OLD method avoiding logsumexp:'
        # print sums_matrix

        sums_matrix = sums_matrix[:, np.newaxis] # Reshapes the sums matrix so that it can be subtracted from the log_posteriors_per_data_point_matrix (subtraction in logspace = division in normal space)



        normalized_log_posteriors_per_data_point_matrix = np.subtract(log_posteriors_per_data_point_matrix, sums_matrix)

        # 9) Normalize the current posterior distribution:


        # print ''
        # print ''
        # print 'This is the inference() method of the PragmaticAgent class:'
        #
        #
        # # TODO: See below for method using logsumexp:
        # normalized_log_posteriors = np.subtract(self.log_posteriors, logsumexp(self.log_posteriors))
        #
        # print ''
        # print "np.exp(normalized_log_posteriors) using logsumexp are:"
        # print np.exp(normalized_log_posteriors)

        unlogged_posteriors = np.exp(self.log_posteriors)
        sum_unlogged_posteriors = np.sum(unlogged_posteriors)
        log_sum_posteriors = np.log(sum_unlogged_posteriors)
        normalized_log_posteriors = np.subtract(self.log_posteriors, log_sum_posteriors)
        #
        # print ''
        # print "np.exp(normalized_log_posteriors) OLD method avoiding logsumexp are:"
        # print np.exp(normalized_log_posteriors)

        # 10) And update the agent's attribute self.log_posteriors to the newly calculated log_posteriors:
        self.log_posteriors = normalized_log_posteriors
        return normalized_log_posteriors_per_data_point_matrix


    def log_likelihoods_on_signal_counts_data(self, signal_counts_data, error):
    # def log_likelihoods_on_signal_counts_data(self, signal_counts_data,dataset_array_pickle_file_name, log_likelihood_pickle_file_name, error):
        """
        This method uses memoization. It accesses and updates a pickle file ("log_likelihood_pickle_file") containing a 2D numpy array that stores an array of likelihoods for each possible dataset. This array has the number of possible datasets as the number of rows, and the length of the hypothesis space as the number of columns. It is indexed according to the order of another pickle file that has all the possible datasets stored in the signal_counts_per_context format (given n_helpful_contexts, n_observations and n_signals). In the log_likelihood_pickle_file, the log_likelihoods (per hypothesis) are stored and updated dynamically for each new dataset that is encountered. If the dataset has been observed before by another learner with the same hypothesis space, the likelihoods are simply loaded from the pickle file. If the dataset has not been encountered before, the likelihoods are calculated and written to the pickle file.
        """
        signal_counts_per_context_matrix = signal_counts_data.signal_counts_per_context_matrix
        # signal_counts_per_context_matrix_flattened = signal_counts_per_context_matrix.flatten()
        # dataset_array = np.asarray(pickle.load(open(dataset_array_pickle_file_name, "rb")))
        # dataset_index = np.argwhere((dataset_array == signal_counts_per_context_matrix_flattened).all(axis=1))[0][0]
        # log_likelihood_array = pickle.load(open(log_likelihood_pickle_file_name, "rb"))
        # if np.isnan(log_likelihood_array[dataset_index][0]):
        log_likelihoods = np.zeros(len(self.hypothesis_space))  # 0.0 in log-space is 1.0 in probability space. We have to initialise the likelihoods as 1's, because from here this number will be multiplied with the actual likelihoods of the observations.
        for h in range(len(self.hypothesis_space)):
            composite_hypothesis = self.hypothesis_space[h]
            persp_hyp_index = composite_hypothesis[0]
            lex_hyp_index = composite_hypothesis[1]
            perspective_hyp = self.perspective_hyps[persp_hyp_index]
            lexicon_hyp = self.lexicon_hyps[lex_hyp_index]
            for c in range(len(signal_counts_data.helpful_contexts)):
                context = signal_counts_data.helpful_contexts[c]
                speaker_intention = self.calc_intention(perspective_hyp, context)
                log_speaker_intention = np.log(speaker_intention)
                signal_counts = signal_counts_per_context_matrix[c]
                for signal in range(len(signal_counts)):
                    if signal_counts[signal] > 0.:
                        signal_likelihood = 0.
                        signal_log_likelihood = np.log(signal_likelihood)
                        for meaning in range(len(speaker_intention)):
                            signal_probs = self.calc_signal_probs_pragmatic(context, self.speaker_type, perspective_hyp, self.sal_alpha, meaning, lexicon_hyp, error)
                            log_signal_probs = np.log(signal_probs)
                            # 5) Multiply the probability of that meaning being the intended meaning and the probability of the utterance signal being produced for that meaning (multiplication = addition in logspace):
                            log_likelihood_for_meaning = np.add(log_speaker_intention[meaning], log_signal_probs[signal])
                            # 6) And add this up for all meanings in the intention distribution to arrive at the full likelihood of the utterance:
                            signal_log_likelihood = np.logaddexp(signal_log_likelihood, log_likelihood_for_meaning)
                        for i in range(signal_counts[signal].astype(int)):
                            log_likelihoods[h] += signal_log_likelihood
        # log_likelihood_array[dataset_index] = log_likelihoods
        # pickle.dump(log_likelihood_array, open(log_likelihood_pickle_file_name, 'wb'))
        # else:
        #     log_likelihoods = log_likelihood_array[dataset_index]
        return log_likelihoods



#############################################################################

class DistinctionAgent(Agent):
    """
    This is an agent that can distinguish between different speakers (c.e. has separate probability distributions for the different speakers that it encounters)
    """

    def __init__(self, perspective_hyps, lexicon_hyps, log_priors, log_posteriors, perspective, alpha, lexicon, learning_type, n_speakers):
        """
        This initializes the same as for the Agent superclass, with the only difference that an extra attribute self.all_speakers_log_posteriors is added, containing a 2D numpy array that has an array of posteriors for each speaker in the population (where the index in the posteriors array matches the index of the speaker in the Population object).
        :param perspective_hyps:
        :param lexicon_hyps:
        :param log_priors:
        :param log_posteriors:
        :param perspective:
        :param lexicon:
        :param learning_type:
        :param n_speakers:
        :return:
        """
        super(DistinctionAgent, self).__init__(perspective_hyps, lexicon_hyps, log_priors, log_posteriors, perspective, alpha, lexicon, learning_type)
        self.hypothesis_space = list_hypothesis_space_with_speaker_distinction(perspective_hyps, lexicon_hyps, n_speakers)
        self.n_speakers = n_speakers


    def inference(self, n_contexts, n_utterances, speaker_annotated_data, error):
        """
        :param n_contexts: The number of contexts
        :param n_utterances: The number of utterances observed in each context
        :param speaker_annotated_data: A SpeakerAnnotatedData object, which has attributes .contexts, .utterances and .speaker_id_matrix. So for each utterance we can reconstruct which agent produced it.
        :return: A 3D numpy array with axis 0 = speakers, axis 1 = (contexts*utterances)+1 and axis 2 = log posteriors (with length = number of perspective hypotheses * number of lexicon hypotheses)
        """

        # 1) First the matrix is created that will save the log posteriors for each data_dict point the learner observes:
        posteriors_per_data_point_matrix = np.zeros((((n_contexts*n_utterances)+1), len(self.hypothesis_space))) #+1 is added because the original prior distribution (c.e. the distribution of belief before making any observations) will be the first posterior to be saved to the matrix
        log_posteriors_per_data_point_matrix = np.log(posteriors_per_data_point_matrix)
        ## 1.2) The first posterior distribution is the prior:
        log_posteriors_per_data_point_matrix[0] = self.log_posteriors


        counter = 1 # this starts at 1 since the first log_posteriors written to the matrix is the prior

        # 2) Then the learner goes through each data_dict point and for each data_dict point updates the posterior probability of the different hypotheses:
        # 2.1) For each context in the data_dict set:
        for i in range(len(speaker_annotated_data.contexts)):
            context = speaker_annotated_data.contexts[i]
            utterances = speaker_annotated_data.utterances[i]
            speaker_id = speaker_annotated_data.speaker_id_matrix[i]


            # 2.2) For each utterance j that was produced in context c:
            for j in range(len(utterances)):
                utterance = utterances[j]
                # 2.3) For each hypothesis k in the hypothesis space:
                for k in range(len(self.hypothesis_space)):
                    composite_hypothesis = self.hypothesis_space[k]
                    persp_hyp_index = composite_hypothesis[0][int(speaker_id)]
                    lex_hyp_index = composite_hypothesis[1]
                    perspective_hyp = self.perspective_hyps[persp_hyp_index]
                    lexicon_hyp = self.lexicon_hyps[lex_hyp_index]
                    speaker_intention = self.calc_intention(perspective_hyp, context)
                    initial_utterance_likelihood = 0.0 # This has to be zero because the probability values for the different meanings are ADDED rather than multiplied
                    utterance_log_likelihood = np.log(initial_utterance_likelihood)
                    utterance_log_likelihood = np.nan_to_num(utterance_log_likelihood) # turns -inf into a very large negative number

                    # 2.4) For each meaning l in the speaker's intention distribution:
                    for l in range(len(speaker_intention)):
                        meaning = l
                        meaning_prob = speaker_intention[l]
                        log_meaning_prob = np.log(meaning_prob)
                        log_meaning_prob = np.nan_to_num(log_meaning_prob) # turns -inf into a very large negative number
                        signal_probs = self.calc_signal_probs(lexicon_hyp, meaning, error)
                        log_signal_probs = np.log(signal_probs)
                        log_signal_probs = np.nan_to_num(log_signal_probs) # turns -inf into a very large negative number

                        # 5) Multiply the probability of that meaning being the intended meaning and the probability of the utterance signal being produced for that meaning (multiplication = addition in logspace):
                        log_likelihood_for_meaning = log_meaning_prob+log_signal_probs[int(utterance)]

                        # 6) And add this up for all meanings in the intention distribution to arrive at the full likelihood of the utterance:
                        utterance_log_likelihood = np.logaddexp(utterance_log_likelihood, log_likelihood_for_meaning)

                    # 7) Update the old posterior for this hypothesis by multiplying it with the utterance likelihood calculated at step 6 (addition in logspace is multiplication in normal probability space)

                    self.log_posteriors[k] += utterance_log_likelihood

                # 8) After updating the posterior for each hypothesis c based on utterance k, we normalize the posterior distribution of the current speaker:

                self.log_posteriors = np.subtract(self.log_posteriors, logsumexp(self.log_posteriors))

                # 9) Then we add the new posterior distribution to the matrix that saves the timecourse data_dict:
                log_posteriors_per_data_point_matrix[counter] = self.log_posteriors
                counter += 1
        return log_posteriors_per_data_point_matrix










class Population(object):
    """
    A Population object consists of a list of Agent objects
    """
    def __init__(self, size, n_meanings, n_signals, hypothesis_space, perspective_hyps, lexicon_hyps, learner_perspective, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_strength, perspectives, perspective_probs, sal_alpha, lexicon_probs, production_error, extra_error, pragmatic_level, optimality_alpha, n_contexts, context_type, context_generation, context_size, helpful_contexts, n_utterances, learning_types, learning_type_probs):
        """
        :param size: The size of the population (integer)
        :param agent_type: The type of agents (can be set to either 'p_distinction' or 'no_p_distinction')
        :param perspectives: The perspectives that the agents can have
        :param perspective_probs: The probabilities with which the different perspectives will be present in the population
        :param lexicons: The lexicons that the agents can have
        :param lexicon_probs: The probabilities with which the different lexicons will be present in the population
        :param learning_types: The learning types that the agents can have
        :param learning_type_probs: The probabilities with which the different learning types will be present in the population
        :return: Creates a population with the specified attributes
        """
        self.pop_type = 'singular'
        self.size = size
        self.agent_type = 'no_p_distinction'
        self.n_meanings = n_meanings
        self.n_signals = n_signals
        self.hypothesis_space = hypothesis_space
        self.perspective_hyps = perspective_hyps
        self.learner_perspective = learner_perspective
        self.perspective_prior_type = perspective_prior_type
        self.perspective_prior_strength = perspective_prior_strength
        self.lexicon_prior_type = lexicon_prior_type
        self.lexicon_prior_strength = lexicon_prior_strength
        self.perspectives = perspectives #TODO: Change this so that this is just the same attribute as self.perspective_hyps
        self.perspective_probs = perspective_probs
        self.perspectives_per_agent = []
        self.sal_alpha = sal_alpha
        self.lexicon_hyps = lexicon_hyps
        self.lexicon_probs = lexicon_probs
        self.error = production_error
        self.extra_error = extra_error
        self.pragmatic_level = pragmatic_level
        self.optimality_alpha = optimality_alpha
        self.n_contexts = n_contexts
        self.context_type = context_type
        self.context_generation = context_generation
        self.context_size = context_size
        self.helpful_contexts = helpful_contexts
        self.n_utterances = n_utterances
        self.lexicons_per_agent = []
        # The initial generation all gets the same lexicon index, of the final lexicon in the lexicon hypothesis space, which is the lexicon that maps all meanings to all signals. Those lexicon indices are used to calculate the communicative accuracy of the first generation of agents with their non-existent parent. (So we assume that the 'parents' of generation 0 all have the same fully ambiguous all-to-all lexicon.)
        self.lex_indices_per_agent = np.array([(len(lexicon_hyps)-1) for x in range(size)])
        self.parent_index_per_learner = np.arange(size)
        self.parent_generation = []
        self.parent_lex_indices = np.zeros(size)
        self.learning_types = learning_types
        self.learning_type_probs = learning_type_probs
        self.learning_types_per_agent = []
        self.lexicons = []
        self.population = self.create_pop()

    def create_pop(self):
        """
        :return: A list containing all the Agent objects in the population
        """
        population = []

        pop_perspectives = np.random.choice(self.perspectives, size=self.size, p=self.perspective_probs)

        self.perspectives_per_agent = pop_perspectives
        pop_lex_indices = np.random.choice(np.arange(len(self.lexicon_hyps)), size=self.size, p=self.lexicon_probs)
        self.lexicons_per_agent = np.array([self.lexicon_hyps[l] for l in pop_lex_indices])
        self.learning_types_per_agent = np.random.choice(self.learning_types, size=self.size, p=self.learning_type_probs)
        perspective_prior = prior.create_perspective_prior(self.perspective_hyps, self.lexicon_hyps, self.perspective_prior_type, self.learner_perspective, self.perspective_prior_strength)
        lexicon_prior = prior.create_lexicon_prior(self.lexicon_hyps, self.lexicon_prior_type, self.lexicon_prior_strength, self.error)
        composite_log_priors_population = prior.list_composite_log_priors(self.agent_type, self.size, self.hypothesis_space, self.perspective_hyps, self.lexicon_hyps, perspective_prior, lexicon_prior) # The full set of composite priors on a LOG SCALE (1D numpy array)
        for i in range(self.size):
            perspective = pop_perspectives[i]
            lexicon = Lexicon('specified_lexicon', self.n_meanings, self.n_signals, specified_lexicon=self.lexicons_per_agent[i])
            learning_type = self.learning_types_per_agent[i]
            #FIXME: A bit strange that the default attributes for creating an agent have to be specified as parameters in the params_and_run module, is it not..? (see all three if and elif statements below)

            if self.pragmatic_level == 'literal' or self.pragmatic_level == 'perspective-taking':
                agent = Agent(self.perspective_hyps, self.lexicon_hyps, composite_log_priors_population, composite_log_priors_population, perspective, self.sal_alpha, lexicon, learning_type)
            elif self.pragmatic_level == 'prag':
                agent = PragmaticAgent(self.perspective_hyps, self.lexicon_hyps, composite_log_priors_population, composite_log_priors_population, perspective, self.sal_alpha, lexicon, learning_type, self.pragmatic_level, self.pragmatic_level, self.optimality_alpha, self.extra_error)

            agent.id = int(i)
            population.append(agent)
        return population


    def calc_ca_lex_only(self, ca_measure_type, parent, learner):
        # This method calculates communicative accuracy in both directions: How well the learner understands their cultural parent and also how well the learner can express meanings to their cultural parent. The method does NOT take perspective into account.
        if learner.pragmatic_level != 'literal':
            raise ValueError("lex_only method of CA calculation only works when pragmatic_level == 'literal'")
        production_probs_parent = np.zeros((self.n_meanings, self.n_signals))
        production_probs_learner = np.zeros((self.n_meanings, self.n_signals))
        reception_probs_parent = np.zeros((self.n_signals, self.n_meanings))
        reception_probs_learner = np.zeros((self.n_signals, self.n_meanings))
        for m in range(self.n_meanings):
            signal_probs_parent = learner.calc_signal_probs(parent.lexicon.lexicon, m, self.error)
            production_probs_parent[m] = signal_probs_parent
        for s in range(self.n_signals):
            meaning_probs_learner = learner.calc_meaning_probs(learner.lexicon.lexicon, s, self.error)
            reception_probs_learner[s] = meaning_probs_learner
        reception_probs_learner = reception_probs_learner.T
        learner_comprehension_success = np.multiply(production_probs_parent, reception_probs_learner)
        success_per_meaning_comprehension = np.sum(learner_comprehension_success, axis=1)
        ca_comprehension = np.mean(success_per_meaning_comprehension)
        ca = ca_comprehension
        if ca_measure_type == "comp_n_prod":
            for m in range(self.n_meanings):
                signal_probs_learner = learner.calc_signal_probs(learner.lexicon.lexicon, m, self.error)
                production_probs_learner[m] = signal_probs_learner
            for s in range(self.n_signals):
                meaning_probs_parent = learner.calc_meaning_probs(parent.lexicon.lexicon, s, self.error)
                reception_probs_parent[s] = meaning_probs_parent
            reception_probs_parent = reception_probs_parent.T
            learner_production_success = np.multiply(production_probs_learner, reception_probs_parent)
            success_per_meaning_production = np.sum(learner_production_success, axis=1)
            ca_production = np.mean(success_per_meaning_production)
            ca = np.mean(np.array([ca_comprehension, ca_production]))
        return ca

    def calc_ca_lex_n_context(self, ca_measure_type, n_interactions, parent, learner):
        if learner.pragmatic_level != 'literal':
            raise ValueError("lex_only method of CA calculation only works when pragmatic_level == 'literal'")
        context_matrix = gen_context_matrix('random', self.n_meanings, self.n_meanings, n_interactions)
        if ca_measure_type == 'comp_n_prod':
            n_parent_productions = n_interactions/2
        elif ca_measure_type == 'comp_only':
            n_parent_productions = n_interactions
        parent_data = parent.produce_data(self.n_meanings, self.n_signals, context_matrix[0:n_parent_productions], self.n_utterances, self.error, self.extra_error)
        learner_interpretations = learner.interpret_data(self.n_meanings, parent_data, 'lex_n_context', self.error, perspective=None)
        score_list = np.zeros(n_interactions * self.n_utterances)
        counter = 0
        for i in range(len(parent_data.topics)):
            for j in range(len(parent_data.topics[0])):
                topic = parent_data.topics[i][j].astype(float)
                interpretation = learner_interpretations[i][j].astype(float)
                if topic == interpretation:
                    score_list[counter] = 1.
                counter += 1
        if ca_measure_type == 'comp_n_prod':
            learner_data = learner.produce_data(self.n_meanings, self.n_signals, context_matrix[n_parent_productions:], self.n_utterances, self.error, self.extra_error)
            parent_interpretations = parent.interpret_data(self.n_meanings, learner_data, 'lex_n_context', self.error, perspective=None)
            for i in range(len(learner_data.topics)):
                for j in range(len(learner_data.topics[0])):
                    topic = learner_data.topics[i][j].astype(float)
                    interpretation = parent_interpretations[i][j].astype(float)
                    if topic == interpretation:
                        score_list[counter] = 1.
                    counter += 1
        ca = np.mean(score_list)
        return ca

    def calc_ca_lex_n_perspective(self, ca_measure_type, n_interactions, parent, learner):
        # if learner.pragmatic_level != 'perspective-taking':
        #     raise ValueError("lex_n_perspective method of CA calculation only works when pragmatic_level == 'perspective-taking'")
        if ca_measure_type != 'comp_only':
            raise ValueError("lex_n_perspective method of CA calculation only works when ca_measure_type == 'comp_only'")
        # Below the learner picks a perspective hypothesis from their posterior distribution to assign to their parent:
        winning_hyp_index = learner.pick_winning_hyp()
        winning_hyp = learner.hypothesis_space[winning_hyp_index]
        learner_p_hyp_index = winning_hyp[0]
        learner_p_hyp = self.perspective_hyps[learner_p_hyp_index]
        context_matrix = gen_context_matrix('random', self.n_meanings, self.n_meanings, n_interactions)
        ca_per_context = np.zeros(len(context_matrix))
        for c in range(len(context_matrix)):
            context = context_matrix[c]
            s_prod_probs = parent.calc_literal_speaker(context, parent.lexicon.lexicon, self.error, parent.perspective, self.sal_alpha)
            log_s_prod_probs = np.log(s_prod_probs)
            l_rec_probs = learner.calc_pragmatic_listener(context, learner.lexicon.lexicon, learner_p_hyp, 'literal', self.error, self.sal_alpha)
            l_rec_probs_transposed = l_rec_probs.T
            log_l_rec_probs_transposed = np.log(l_rec_probs_transposed)
            log_multiplied_matrices = np.add(log_s_prod_probs, log_l_rec_probs_transposed)
            multiplied_matrices = np.exp(log_multiplied_matrices)
            sum_over_signals = np.sum(multiplied_matrices, axis=1)
            ca = np.sum(sum_over_signals)
            ca_per_context[c] = ca
        mean_ca = np.mean(ca_per_context)
        # return ca_per_context, mean_ca
        return mean_ca


    def calc_ca_pragmatic(self, ca_measure_type, n_interactions, parent, learner, extra_error):
        if ca_measure_type != 'comp_only':
            raise ValueError("lex_n_perspective method of CA calculation only works when ca_measure_type == 'comp_only'")
        # Below the learner picks a perspective hypothesis from their posterior distribution to assign to their parent:
        winning_hyp_index = learner.pick_winning_hyp()
        winning_hyp = learner.hypothesis_space[winning_hyp_index]
        learner_p_hyp_index = winning_hyp[0]
        learner_p_hyp = self.perspective_hyps[learner_p_hyp_index]
        context_matrix = gen_context_matrix('random', self.n_meanings, self.n_meanings, n_interactions)
        ca_per_context = np.zeros(len(context_matrix))
        for c in range(len(context_matrix)):
            context = context_matrix[c]
            parent_intention = parent.calc_intention(parent.perspective, context)
            log_parent_intention = np.log(parent_intention)
            l_rec_probs_according_to_speaker = parent.calc_pragmatic_listener(context, parent.lexicon.lexicon, parent.perspective, 'literal', self.error, self.sal_alpha)
            s_prod_probs = parent.calc_pragmatic_speaker(l_rec_probs_according_to_speaker)
            log_s_prod_probs = np.log(s_prod_probs)
            l_rec_probs = learner.calc_pragmatic_listener(context, learner.lexicon.lexicon, learner_p_hyp, 'prag', self.error, self.sal_alpha)
            l_rec_probs_transposed = l_rec_probs.T
            log_l_rec_probs_transposed = np.log(l_rec_probs_transposed)
            if self.extra_error == True:
                for m in range(len(s_prod_probs)):
                    meaning_row = s_prod_probs[m]
                    addition_matrix = np.zeros((self.n_signals, self.n_signals))
                    for s in range(len(meaning_row)):
                        error_prob = meaning_row[s]*self.error
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


    def calc_comm_acc(self, communication_type, ca_measure_type, n_interactions, parent, learner):
        if communication_type == 'lex_only':
            mean_ca = self.calc_ca_lex_only(ca_measure_type, parent, learner)
        elif communication_type == 'lex_n_context':
            mean_ca = self.calc_ca_lex_n_context(ca_measure_type, n_interactions, parent, learner)
        elif communication_type == 'lex_n_p':
            # ca_per_context, mean_ca = self.calc_ca_lex_n_perspective(n_interactions, parent, learner)
            mean_ca = self.calc_ca_lex_n_perspective(ca_measure_type, n_interactions, parent, learner)
        elif communication_type == 'prag':
            # ca_per_context, mean_ca = self.calc_ca_pragmatic(n_interactions, parent, learner)
            mean_ca = self.calc_ca_pragmatic(ca_measure_type, n_interactions, parent, learner, self.extra_error)
        # return ca_per_context, mean_ca
        return mean_ca


    def calc_fitness(self, selection_type, selection_weighting, communication_type, ca_measure_type, n_interactions, parent_index_per_learner, parents, parent_lex_indices):
        """
        :param selection_type: This can be set to either 'p_taking', 'l_learning' or 'ca_with_parent'
        :param selection_weighting: This determines the strength of the selection (the larger the stronger)
        :return:
        """
        pop_log_posteriors = self.get_all_log_posteriors_matrix()
        fitness_per_agent = np.ones(self.size)
        for a in range(self.size):
            learner_log_posteriors = pop_log_posteriors[a]
            learner_log_posteriors_split_on_p_hyps = np.array(np.split(learner_log_posteriors, len(self.perspective_hyps)))
            learner = self.population[a]
            parent_index = int(parent_index_per_learner[a])
            if len(parents) == 0:
                parent = learner
            else:
                parent = parents[parent_index]
            if selection_type == 'p_taking':
                parent_perspective = parent.perspective
                parent_p_index = np.where(self.perspective_hyps == parent_perspective)[0][0]
                p_log_posteriors = learner_log_posteriors_split_on_p_hyps[parent_p_index, :]
                learner_fitness = np.exp(logsumexp(p_log_posteriors))
                fitness_per_agent[a] = learner_fitness

            elif selection_type == 'l_learning':
                parent_lex_index = int(parent_lex_indices[a])
                l_log_posteriors = learner_log_posteriors_split_on_p_hyps[:, parent_lex_index]
                learner_fitness = np.exp(logsumexp(l_log_posteriors))
                fitness_per_agent[a] = learner_fitness

            elif selection_type == 'ca_with_parent':
                learner_fitness = self.calc_comm_acc(communication_type, ca_measure_type, n_interactions, parent, learner)
                fitness_per_agent[a] = learner_fitness

        return fitness_per_agent



    def produce_pop_data(self, context_matrix, n_utterances, speaker_order):
        """
        :param context_matrix: A 2D numpy matrix of contexts
        :param n_utterances: Global variable determining the number of utterances produced per context (float)
        :return: A data_dict object produced by the population, for which speakers have been chosen from the population with uniform probability
        """
        pop_topic_matrix = np.zeros((len(context_matrix), n_utterances))
        pop_utterance_matrix = np.zeros((len(context_matrix), n_utterances))
        for c in range(len(context_matrix)):
            context = context_matrix[c]
            speaker = speaker_order[c]
            speaker_data = speaker.produce_data(self.n_meanings, self.n_signals, np.array([context]), self.n_utterances, self.error, self.extra_error)
            speaker_topics = speaker_data.topics[0]
            speaker_utterances = speaker_data.utterances[0]
            pop_topic_matrix[c] = speaker_topics
            pop_utterance_matrix[c] = speaker_utterances
            #TODO: Note that at the moment there is only one speaker per context
        pop_data = Data(context_matrix, pop_topic_matrix, pop_utterance_matrix)
        return pop_data



    def produce_pop_data_fixed_contexts(self, context_matrix, n_utterances, speaker_order, helpful_contexts, n_signals):
        """
        :param context_matrix: A 2D numpy matrix of contexts
        :param n_utterances: Global variable determining the number of utterances produced per context (float)
        :return: A data_dict object produced by the population, for which speakers have been chosen from the population with uniform probability
        """
        pop_topic_matrix = np.zeros((len(context_matrix), n_utterances)).astype(int)
        signal_counts_per_context_matrix = np.zeros((len(helpful_contexts), n_signals)).astype(int)
        for c in range(len(context_matrix)):
            context_index = c % len(helpful_contexts)
            context = context_matrix[c]
            speaker = speaker_order[c]
            speaker_data = speaker.produce_data(self.n_meanings, self.n_signals, np.array([context]), self.n_utterances, self.error, self.extra_error)
            speaker_topics = speaker_data.topics[0]
            speaker_utterances = speaker_data.utterances[0].astype(int)
            pop_topic_matrix[c] = speaker_topics
            signal_counts_per_context_matrix[context_index, speaker_utterances] += 1
        pop_data = FixedContextsData(context_matrix, pop_topic_matrix, signal_counts_per_context_matrix, helpful_contexts)
        return pop_data


    def pop_update(self, recording, context_generation, helpful_contexts, n_meanings, n_signals, error, turnover_type, selection_type, selection_weighting, communication_type, ca_measure_type, n_interactions, teacher_type, perspectives_per_agent=None):
    # def pop_update(self, dataset_array_pickle_file_name, log_likelihood_pickle_file_name, recording, context_generation, helpful_contexts, n_meanings, n_signals, error, turnover_type, selection_type, selection_weighting, communication_type, ca_measure_type, n_interactions, teacher_type, perspectives_per_agent=None):
        """
        :param turnover_type: 'chain' for one agent at a time, or 'whole_pop' for the whole population at once
        :return: Doesn't return anything, but changes self.population to the new population
        """
        parent_fitness_array = self.calc_fitness(selection_type, selection_weighting, communication_type, ca_measure_type, n_interactions, self.parent_index_per_learner, self.parent_generation, self.parent_lex_indices)
        avg_fitness = np.mean(parent_fitness_array)

        if selection_weighting == 'none':
            parent_probs = np.divide(parent_fitness_array, np.sum(parent_fitness_array))
        elif isinstance(selection_weighting, float):
            parent_fitness_array_weighted = np.multiply(parent_fitness_array, selection_weighting)
            parent_fitness_array_weighted_exp = np.exp(parent_fitness_array_weighted)
            parent_probs = np.divide(parent_fitness_array_weighted_exp, np.sum(parent_fitness_array_weighted_exp))

        if turnover_type == 'chain':
            n_agents_to_be_replaced = 1
        elif turnover_type == 'whole_pop':
            n_agents_to_be_replaced = self.size

        # 1) For the number of agents that need to be replaced in the population, we do the following:
        # FIXME: Figure out why the prior and posterior of ALL agents is updated in the chain method, when it should update only those of the new agent

        data_per_agent = []
        # 1.1) First we generate data_dict from the current population:
        for i in range(n_agents_to_be_replaced):
            if self.context_generation == 'random':
                context_matrix = gen_context_matrix(self.context_type, self.n_meanings, self.context_size, self.n_contexts)
            elif self.context_generation == 'only_helpful':
                context_matrix = gen_helpful_context_matrix(self.n_meanings, self.n_contexts, self.helpful_contexts)
            elif self.context_generation == 'optimal':
                context_matrix = gen_helpful_context_matrix_fixed_order(self.n_meanings, self.n_contexts, self.helpful_contexts)

            speaker_order = create_speaker_order_iteration(self, selection_type, parent_probs, teacher_type, self.n_contexts)
            self.parent_index_per_learner[i] = speaker_order[0].id

            if context_generation == 'random':
                old_pop_data = self.produce_pop_data(context_matrix, self.n_utterances, speaker_order)
            elif context_generation == 'optimal':
                old_pop_data = self.produce_pop_data_fixed_contexts(context_matrix, self.n_utterances, speaker_order, helpful_contexts, n_signals)
            data_per_agent.append(old_pop_data)
        self.parent_generation = self.population
        self.parent_lex_indices = np.zeros(self.size)
        for a in range(self.size):
            parent_index = self.parent_index_per_learner[a]
            parent = self.population[parent_index]
            self.parent_lex_indices[a] = self.lex_indices_per_agent[parent_index]
        selected_hyp_per_agent_matrix = np.zeros(n_agents_to_be_replaced)
        # normalized_log_posteriors_per_data_point_per_agent_matrix = np.zeros((n_agents_to_be_replaced, (self.n_contexts+1), len(self.hypothesis_space)))
        normalized_log_posteriors_per_agent_matrix = np.zeros((n_agents_to_be_replaced, len(self.hypothesis_space)))
        for i in range(n_agents_to_be_replaced):
            # 1.2) We choose the new agent's perspective and learning_type with uniform probability from the attributes self.perspective_probs and self.learning_probs:
            new_agent_perspective = np.random.choice(self.perspectives, size=1, p=self.perspective_probs)
            new_agent_learning_type = np.random.choice(self.learning_types, size=1, p=self.learning_type_probs)
            new_agent_lexicon = Lexicon('empty_lex', self.n_meanings, self.n_signals)

            # 1.3) Then we initialize the new agent with that perspective and learning_type and with an empty lexicon
            #FIXME: Again: the new agent is initialized with attributes that are globally defined in the params_and_run module ()
            perspective_prior = prior.create_perspective_prior(self.perspective_hyps, self.lexicon_hyps, self.perspective_prior_type, self.learner_perspective, self.perspective_prior_strength)
            lexicon_prior = prior.create_lexicon_prior(self.lexicon_hyps, self.lexicon_prior_type, self.lexicon_prior_strength, self.error)
            composite_log_priors = prior.list_composite_log_priors(self.agent_type, self.size, self.hypothesis_space, self.perspective_hyps, self.lexicon_hyps, perspective_prior, lexicon_prior)

            if self.pragmatic_level == 'literal' or self.pragmatic_level == 'perspective-taking':
                new_agent = Agent(self.perspective_hyps, self.lexicon_hyps, composite_log_priors, composite_log_priors, new_agent_perspective[0], self.sal_alpha, new_agent_lexicon, new_agent_learning_type[0])
            elif self.pragmatic_level == 'prag':
                new_agent = PragmaticAgent(self.perspective_hyps, self.lexicon_hyps, composite_log_priors, composite_log_priors, new_agent_perspective[0], self.sal_alpha, new_agent_lexicon, new_agent_learning_type[0], self.pragmatic_level, self.pragmatic_level, self.optimality_alpha, self.extra_error)


            # 1.4) Then we get the new agent's parent data from the old population:
            agent_data = data_per_agent[i]

            # 1.5) We subsequently let the new agent learn from the annotated_pop_data of the population and update its lexicon accordingly:


            normalized_log_posteriors_per_agent_matrix[i] = new_agent.inference_on_signal_counts_data(agent_data, error)

            selected_hyp, selected_lex_hyp = new_agent.update_lexicon()
            selected_hyp_per_agent_matrix[i] = selected_hyp
            self.lex_indices_per_agent[i] = selected_lex_hyp

            # 1.6) Then, we remove the oldest agent (c.e. agent with index 0) from the population, and append the new agent at the end:
            self.population = np.delete(self.population, 0)
            self.population = np.append(self.population, new_agent)  # Appends the new agent to the end of the population
            self.perspectives_per_agent = np.delete(self.perspectives_per_agent, 0)
            self.perspectives_per_agent = np.append(self.perspectives_per_agent, new_agent.perspective)
            self.lexicons_per_agent = np.delete(self.lexicons_per_agent, 0)
            self.lexicons_per_agent = np.append(self.population, new_agent.lexicon)
            self.learning_types_per_agent = np.delete(self.learning_types_per_agent, 0)
            self.learning_types_per_agent = np.append(self.population, new_agent.learning_type)

        # 1.7) Finally, the agent id numbers are updated:
        for i in range(self.size):
            agent = self.population[i]
            agent.id = i
        if recording == 'minimal':
            return selected_hyp_per_agent_matrix, avg_fitness, parent_probs, self.parent_index_per_learner, self.parent_lex_indices
        elif recording == 'everything':
            return selected_hyp_per_agent_matrix, avg_fitness, parent_probs, self.parent_index_per_learner, self.parent_lex_indicesnormalized_log_posteriors_per_data_point_per_agent_matrix


    def calc_population_average_lexicon(self):
        """
        :return: A 2D numpy array of size lexicon, that is the average of all the lexicons present in the population
        """
        pop_lexicons_matrix = np.zeros((len(self.population), self.n_meanings, self.n_signals))
        for i in range(len(self.population)):
            agent = self.population[i]
            lexicon = agent.lexicon.lexicon
            pop_lexicons_matrix[i] = lexicon
        average_lexicon = np.mean(pop_lexicons_matrix, axis=0)
        return average_lexicon


    def get_all_lexicons_matrix(self):
        """
        :return: A matrix containing all the lexicon of each agent in the population
        """
        all_lexicons_matrix = np.zeros((len(self.population), self.n_meanings, self.n_signals))
        for i in range(len(self.population)):
            agent = self.population[i]
            agent_lexicon = agent.lexicon.lexicon
            all_lexicons_matrix[i] = agent_lexicon
        return all_lexicons_matrix


    def get_all_log_posteriors_matrix(self):
        all_log_posteriors_matrix = np.zeros((self.size, len(self.hypothesis_space)))
        for i in range(len(self.population)):
            agent = self.population[i]
            agent_log_posteriors = agent.log_posteriors
            all_log_posteriors_matrix[i] = agent_log_posteriors
        return all_log_posteriors_matrix


    def print_population(self):
        """
        Prints each agent (c.e. the agent's attributes) with it's respective index number on a new line
        """
        for i in range(len(self.population)):
            print 
            agent = self.population[i]
            print "This is agent number "+str(i)
            agent.print_agent()





class MixedPopulation(Population):
    """
    A MixedPopulation can contain agents of different pragmatic levels
    """
    def __init__(self, size, n_meanings, n_signals, hypothesis_space, perspective_hyps, lexicon_hyps, learner_perspective, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_strength, perspectives, perspective_probs, sal_alpha, lex_indices_per_agent, lexicons_per_agent, production_error, extra_error, pragmatic_level_initial_pop, optimality_alpha_initial_pop, pragmatic_level_mutants, optimality_alpha_mutants, pragmatic_level_parent_hyp, n_contexts, context_type, context_generation, context_size, helpful_contexts, n_utterances, learning_types, learning_type_probs):
        """
        :param size: The size of the population (integer)
        :param agent_type: The type of agents (can be set to either 'p_distinction' or 'no_p_distinction')
        :param perspectives: The perspectives that the agents can have
        :param perspective_probs: The probabilities with which the different perspectives will be present in the population
        :param lexicons: The lexicons that the agents can have
        :param lexicon_probs: The probabilities with which the different lexicons will be present in the population
        :param learning_types: The learning types that the agents can have
        :param learning_type_probs: The probabilities with which the different learning types will be present in the population
        :return: Creates a population with the specified attributes
        """
        self.pop_type = 'mixed'
        self.size = size
        self.agent_type = 'no_p_distinction'
        self.n_meanings = n_meanings
        self.n_signals = n_signals
        self.hypothesis_space = hypothesis_space
        self.perspective_hyps = perspective_hyps
        self.learner_perspective = learner_perspective
        self.perspective_prior_type = perspective_prior_type
        self.perspective_prior_strength = perspective_prior_strength
        self.lexicon_prior_type = lexicon_prior_type
        self.lexicon_prior_strength = lexicon_prior_strength
        self.perspectives = perspectives #TODO: Change this so that this is just the same attribute as self.perspective_hyps
        self.perspective_probs = perspective_probs
        self.perspectives_per_agent = []
        self.sal_alpha = sal_alpha
        self.lexicon_hyps = lexicon_hyps
        self.error = production_error
        self.extra_error = extra_error
        self.pragmatic_level_initial_pop = pragmatic_level_initial_pop
        self.optimality_alpha_initial_pop = optimality_alpha_initial_pop
        self.pragmatic_level_mutants = pragmatic_level_mutants
        self.optimality_alpha_mutants = optimality_alpha_mutants
        self.pragmatic_level_parent_hyp = pragmatic_level_parent_hyp
        self.n_contexts = n_contexts
        self.context_type = context_type
        self.context_generation = context_generation
        self.context_size = context_size
        self.helpful_contexts = helpful_contexts
        self.n_utterances = n_utterances
        self.lexicons_per_agent = lexicons_per_agent
        # The initial generation all gets the same lexicon index, of the final lexicon in the lexicon hypothesis space, which is the lexicon that maps all meanings to all signals. Those lexicon indices are used to calculate the communicative accuracy of the first generation of agents with their non-existent parent. (So we assume that the 'parents' of generation 0 all have the same fully ambiguous all-to-all lexicon.)
        self.lex_indices_per_agent = lex_indices_per_agent
        self.parent_index_per_learner = np.arange(size)
        self.parent_generation = []
        self.parent_lex_indices = np.zeros(size)
        self.learning_types = learning_types
        self.learning_type_probs = learning_type_probs
        self.learning_types_per_agent = []
        self.lexicons = []
        self.population = self.create_pop()



    def create_pop(self):
        """
        :return: A list containing all the Agent objects in the population
        """
        population = []

        pop_perspectives = np.random.choice(self.perspectives, size=self.size, p=self.perspective_probs)

        self.perspectives_per_agent = pop_perspectives
        self.learning_types_per_agent = np.random.choice(self.learning_types, size=self.size, p=self.learning_type_probs)
        perspective_prior = prior.create_perspective_prior(self.perspective_hyps, self.lexicon_hyps, self.perspective_prior_type, self.learner_perspective, self.perspective_prior_strength)
        lexicon_prior = prior.create_lexicon_prior(self.lexicon_hyps, self.lexicon_prior_type, self.lexicon_prior_strength, self.error)
        composite_log_priors_population = prior.list_composite_log_priors(self.agent_type, self.size, self.hypothesis_space, self.perspective_hyps, self.lexicon_hyps, perspective_prior, lexicon_prior) # The full set of composite priors on a LOG SCALE (1D numpy array)
        for i in range(self.size):
            perspective = pop_perspectives[i]
            lexicon = Lexicon('specified_lexicon', self.n_meanings, self.n_signals, specified_lexicon=self.lexicons_per_agent[i])
            learning_type = self.learning_types_per_agent[i]
            #FIXME: A bit strange that the default attributes for creating an agent have to be specified as parameters in the params_and_run module, is it not..? (see all three if and elif statements below)

            if self.pragmatic_level_initial_pop == 'literal' or self.pragmatic_level_initial_pop == 'perspective-taking':
                agent = Agent(self.perspective_hyps, self.lexicon_hyps, composite_log_priors_population, composite_log_priors_population, perspective, self.sal_alpha, lexicon, learning_type, self.pragmatic_level_initial_pop)
            elif self.pragmatic_level_initial_pop == 'prag':
                agent = PragmaticAgent(self.perspective_hyps, self.lexicon_hyps, composite_log_priors_population, composite_log_priors_population, perspective, self.sal_alpha, lexicon, learning_type, self.pragmatic_level_parent_hyp, self.pragmatic_level_initial_pop, self.optimality_alpha, self.extra_error)

            agent.id = int(i)
            population.append(agent)
        return population


    def calc_fitness_old(self, selection_type, selection_weighting, communication_type_initial_pop, ca_measure_type_initial_pop, communication_type_mutants, ca_measure_type_mutants, n_interactions, learners, parent_index_per_learner, parents, parent_lex_indices):
        """
        :param selection_type: This can be set to either 'p_taking', 'l_learning' or 'ca_with_parent'
        :param selection_weighting: This determines the strength of the selection (the larger the stronger)
        :return:
        """
        pop_log_posteriors = self.get_all_log_posteriors_matrix()
        pop_posteriors_unlogged = np.exp(pop_log_posteriors)
        fitness_per_agent = np.ones(len(learners))
        for a in range(len(learners)):
            learner_posteriors = pop_posteriors_unlogged[a]
            learner_posteriors_split_on_p_hyps = np.split(learner_posteriors, len(self.perspective_hyps))
            learner_fitness = 0.
            learner = learners[a]
            parent_index = parent_index_per_learner[a]
            if len(parents) == 0:
                parent = learner
            else:
                parent = parents[parent_index.astype(int)]
            if selection_type == 'p_taking':
                #TODO: Write this part of the function to be the same as the 'l_learning' part?
                for p in range(len(self.perspective_hyps)):
                    p_frequency = self.perspective_probs[p]
                    p_posteriors = learner_posteriors_split_on_p_hyps[p]
                    p_posteriors_times_freq = np.multiply(p_posteriors, p_frequency)
                    p_posteriors_times_freq_sum = np.sum(p_posteriors_times_freq)
                    learner_fitness += p_posteriors_times_freq_sum
                fitness_per_agent[a] = learner_fitness
            elif selection_type == 'l_learning':
                parent_lex_index = parent_lex_indices[a]
                learner_posteriors_split_on_p_hyps = np.array(learner_posteriors_split_on_p_hyps)
                l_posteriors = learner_posteriors_split_on_p_hyps[:, parent_lex_index]
                learner_fitness = np.sum(l_posteriors)
                fitness_per_agent[a] = learner_fitness
            elif selection_type == 'ca_with_parent':
                if self.pragmatic_level_initial_pop == 'literal' or self.pragmatic_level_initial_pop == 'perspective-taking' and learner.pragmatic_level == 'literal' or learner.pragmatic_level == 'perspective-taking':
                    learner_fitness = self.calc_comm_acc(communication_type_initial_pop, ca_measure_type_initial_pop, n_interactions, parent, learner)
                elif self.pragmatic_level_mutants == 'prag' and learner.pragmatic_level == 'prag':
                    learner_fitness = self.calc_comm_acc(communication_type_mutants, ca_measure_type_mutants, n_interactions, parent, learner)
                fitness_per_agent[a] = learner_fitness
        return fitness_per_agent




    def calc_fitness(self, selection_type, selection_weighting, communication_type_initial_pop, ca_measure_type_initial_pop, communication_type_mutants, ca_measure_type_mutants, n_interactions, learners, parent_index_per_learner, parents, parent_lex_indices):

    #def calc_fitness(self, selection_type, selection_weighting, communication_type, ca_measure_type, n_interactions, parent_index_per_learner, parents, parent_lex_indices):
        """
        :param selection_type: This can be set to either 'p_taking', 'l_learning' or 'ca_with_parent'
        :param selection_weighting: This determines the strength of the selection (the larger the stronger)
        :return:
        """

        pop_log_posteriors = self.get_all_log_posteriors_matrix()
        fitness_per_agent = np.ones(self.size)

        for a in range(self.size):
            learner_log_posteriors = pop_log_posteriors[a]
            learner_log_posteriors_split_on_p_hyps = np.array(np.split(learner_log_posteriors, len(self.perspective_hyps)))
            learner = self.population[a]
            parent_index = int(parent_index_per_learner[a])
            if len(parents) == 0:
                parent = learner
            else:
                parent = parents[parent_index]

            if selection_type == 'p_taking':
                parent_perspective = parent.perspective
                parent_p_index = np.where(self.perspective_hyps == parent_perspective)[0][0]
                p_log_posteriors = learner_log_posteriors_split_on_p_hyps[parent_p_index, :]
                learner_fitness = np.exp(logsumexp(p_log_posteriors))
                fitness_per_agent[a] = learner_fitness

            elif selection_type == 'l_learning':
                parent_lex_index = int(parent_lex_indices[a])
                l_log_posteriors = learner_log_posteriors_split_on_p_hyps[:, parent_lex_index]
                learner_fitness = np.exp(logsumexp(l_log_posteriors))
                fitness_per_agent[a] = learner_fitness

            elif selection_type == 'ca_with_parent':
                if self.pragmatic_level_initial_pop == 'literal' or self.pragmatic_level_initial_pop == 'perspective-taking' and learner.pragmatic_level == 'literal' or learner.pragmatic_level == 'perspective-taking':
                    learner_fitness = self.calc_comm_acc(communication_type_initial_pop, ca_measure_type_initial_pop, n_interactions, parent, learner)
                elif self.pragmatic_level_mutants == 'prag' and learner.pragmatic_level == 'prag':
                    learner_fitness = self.calc_comm_acc(communication_type_mutants, ca_measure_type_mutants, n_interactions, parent, learner)
                fitness_per_agent[a] = learner_fitness

        return fitness_per_agent







    def insert_mutant(self, context_generation, helpful_contexts, n_signals, error, turnover_type, selection_type, selection_weighting, communication_type_initial_pop, ca_measure_type_initial_pop, communication_type_mutants, ca_measure_type_mutants, n_interactions, teacher_type, n_mutants):
        """
        :param turnover_type: 'chain' for one agent at a time, or 'whole_pop' for the whole population at once
        :return: Doesn't return anything, but changes self.population to the new population
        """
        pragmatic_level_per_agent = ['' for a in range(self.size)]

        parent_fitness_array = self.calc_fitness(selection_type, selection_weighting, communication_type_initial_pop, ca_measure_type_initial_pop, communication_type_mutants, ca_measure_type_mutants, n_interactions, self.population, self.parent_index_per_learner, self.parent_generation, self.parent_lex_indices)
        avg_fitness = np.mean(parent_fitness_array)

        if selection_weighting == 'none':
            parent_probs = np.divide(parent_fitness_array, np.sum(parent_fitness_array))
        elif isinstance(selection_weighting, float):
            parent_fitness_array_weighted = np.multiply(parent_fitness_array, selection_weighting)
            parent_fitness_array_weighted_exp = np.exp(parent_fitness_array_weighted)
            parent_probs = np.divide(parent_fitness_array_weighted_exp, np.sum(parent_fitness_array_weighted_exp))

        if turnover_type == 'chain':
            n_agents_to_be_replaced = 1
        elif turnover_type == 'whole_pop':
            n_agents_to_be_replaced = self.size

        # 1) For the number of agents that need to be replaced in the population, we do the following:
        # FIXME: Figure out why the prior and posterior of ALL agents is updated in the chain method, when it should update only those of the new agent

        data_per_agent = []
        # 1.1) First we generate data_dict from the current population:
        for i in range(n_agents_to_be_replaced):
            if self.context_generation == 'random':
                context_matrix = gen_context_matrix(self.context_type, self.n_meanings, self.context_size, self.n_contexts)
            elif self.context_generation == 'only_helpful':
                context_matrix = gen_helpful_context_matrix(self.n_meanings, self.n_contexts, self.helpful_contexts)
            elif self.context_generation == 'optimal':
                context_matrix = gen_helpful_context_matrix_fixed_order(self.n_meanings, self.n_contexts, self.helpful_contexts)

            speaker_order = create_speaker_order_iteration(self, selection_type, parent_probs, teacher_type, self.n_contexts)
            if len(speaker_order) > 0:
                self.parent_index_per_learner[i] = speaker_order[0].id
            else:
                self.parent_index_per_learner[i] = 0

            if context_generation == 'random':
                old_pop_data = self.produce_pop_data(context_matrix, self.n_utterances, speaker_order)
            elif context_generation == 'optimal':
                old_pop_data = self.produce_pop_data_fixed_contexts(context_matrix, self.n_utterances, speaker_order, helpful_contexts, n_signals)
            data_per_agent.append(old_pop_data)
        self.parent_generation = self.population
        self.parent_lex_indices = np.zeros(self.size)
        for a in range(self.size):
            parent_index = self.parent_index_per_learner[a]
            self.parent_lex_indices[a] = self.lex_indices_per_agent[parent_index]
        selected_hyp_per_agent_matrix = np.zeros(n_agents_to_be_replaced)
        normalized_log_posteriors_per_agent_matrix = np.zeros((n_agents_to_be_replaced, len(self.hypothesis_space)))
        for i in range(n_agents_to_be_replaced):
            # 1.2) Then we choose the new agent's perspective and learning_type with uniform probability from the attributes self.perspective_probs and self.learning_probs:
            new_agent_perspective = np.random.choice(self.perspectives, size=1, p=self.perspective_probs)
            new_agent_learning_type = np.random.choice(self.learning_types, size=1, p=self.learning_type_probs)
            new_agent_lexicon = Lexicon('empty_lex', self.n_meanings, self.n_signals)

            # 1.3) Then we initialize the new agent with that perspective and learning_type and with an empty lexicon
            #FIXME: Again: the new agent is initialized with attributes that are globally defined in the params_and_run module ()
            perspective_prior = prior.create_perspective_prior(self.perspective_hyps, self.lexicon_hyps, self.perspective_prior_type, self.learner_perspective, self.perspective_prior_strength)
            lexicon_prior = prior.create_lexicon_prior(self.lexicon_hyps, self.lexicon_prior_type, self.lexicon_prior_strength, self.error)
            composite_log_priors = prior.list_composite_log_priors(self.agent_type, self.size, self.hypothesis_space, self.perspective_hyps, self.lexicon_hyps, perspective_prior, lexicon_prior)

            if i < n_agents_to_be_replaced - n_mutants:
                if self.pragmatic_level_initial_pop == 'literal' or self.pragmatic_level_initial_pop == 'perspective-taking':
                    new_agent = Agent(self.perspective_hyps, self.lexicon_hyps, composite_log_priors, composite_log_priors, new_agent_perspective[0], self.sal_alpha, new_agent_lexicon, new_agent_learning_type[0], self.pragmatic_level_initial_pop)
                elif self.pragmatic_level == 'prag':
                    new_agent = PragmaticAgent(self.perspective_hyps, self.lexicon_hyps, composite_log_priors, composite_log_priors, new_agent_perspective[0], self.sal_alpha, new_agent_lexicon, new_agent_learning_type[0], self.pragmatic_level, self.pragmatic_level, self.optimality_alpha, self.extra_error)
            elif i >= n_agents_to_be_replaced - n_mutants:
                if self.pragmatic_level_mutants == 'literal' or self.pragmatic_level_mutants == 'perspective-taking':
                    new_agent = Agent(self.perspective_hyps, self.lexicon_hyps, composite_log_priors, composite_log_priors, new_agent_perspective[0], self.sal_alpha, new_agent_lexicon, new_agent_learning_type[0], self.pragmatic_level_mutants)
                elif self.pragmatic_level_mutants == 'prag':
                    new_agent = PragmaticAgent(self.perspective_hyps, self.lexicon_hyps, composite_log_priors, composite_log_priors, new_agent_perspective[0], self.sal_alpha, new_agent_lexicon, new_agent_learning_type[0], self.pragmatic_level_mutants, self.pragmatic_level_mutants, self.optimality_alpha_mutants, self.extra_error)

            # 1.4) Then we get the new agent's parent data from the old population:
            agent_data = data_per_agent[i]

            # 1.5) We subsequently let the new agent learn from the annotated_pop_data of the population and update its lexicon accordingly:
            normalized_log_posteriors_per_agent_matrix[i] = new_agent.inference_on_signal_counts_data(agent_data, error)

            selected_hyp, selected_lex_hyp = new_agent.update_lexicon()
            selected_hyp_per_agent_matrix[i] = selected_hyp
            self.lex_indices_per_agent[i] = selected_lex_hyp

            # 1.6) Then, we remove the oldest agent (c.e. agent with index 0) from the population, and append the new agent at the end:
            self.population = np.delete(self.population, 0)
            self.population = np.append(self.population, new_agent)  # Appends the new agent to the end of the population
            self.perspectives_per_agent = np.delete(self.perspectives_per_agent, 0)
            self.perspectives_per_agent = np.append(self.perspectives_per_agent, new_agent.perspective)
            self.lexicons_per_agent = np.delete(self.lexicons_per_agent, 0)
            self.lexicons_per_agent = np.append(self.population, new_agent.lexicon)
            self.learning_types_per_agent = np.delete(self.learning_types_per_agent, 0)
            self.learning_types_per_agent = np.append(self.population, new_agent.learning_type)

        # 1.7) Finally, the agent id numbers are updated:
        for i in range(self.size):
            agent = self.population[i]
            agent.id = i
            pragmatic_level_per_agent[i] = agent.pragmatic_level
        return selected_hyp_per_agent_matrix, avg_fitness, parent_probs, self.parent_index_per_learner, self.parent_lex_indices, pragmatic_level_per_agent



    def pop_update(self, context_generation, helpful_contexts, n_meanings, n_signals, error, turnover_type, selection_type, selection_weighting, communication_type_initial_pop, ca_measure_type_initial_pop, communication_type_mutants, ca_measure_type_mutants, n_interactions, teacher_type, perspectives_per_agent=None, decoupled=None):
        """
        :param turnover_type: 'chain' for one agent at a time, or 'whole_pop' for the whole population at once
        :return: Doesn't return anything, but changes self.population to the new population
        """

        pragmatic_level_per_agent = ['' for a in range(self.size)]

        parent_fitness_array = self.calc_fitness(selection_type, selection_weighting, communication_type_initial_pop, ca_measure_type_initial_pop, communication_type_mutants, ca_measure_type_mutants, n_interactions, self.population, self.parent_index_per_learner, self.parent_generation, self.parent_lex_indices)

        avg_fitness = np.mean(parent_fitness_array)

        if selection_weighting == 'none':
            parent_probs = np.divide(parent_fitness_array, np.sum(parent_fitness_array))
        elif isinstance(selection_weighting, float):
            parent_fitness_array_weighted = np.multiply(parent_fitness_array, selection_weighting)
            parent_fitness_array_weighted_exp = np.exp(parent_fitness_array_weighted)
            parent_probs = np.divide(parent_fitness_array_weighted_exp, np.sum(parent_fitness_array_weighted_exp))

        if turnover_type == 'chain':
            n_agents_to_be_replaced = 1
        elif turnover_type == 'whole_pop':
            n_agents_to_be_replaced = self.size

        # 1) For the number of agents that need to be replaced in the population, we do the following:
        # FIXME: Figure out why the prior and posterior of ALL agents is updated in the chain method, when it should update only those of the new agent

        # 1.1) First we generate a list of biological parents in case the decoupled parameter (which decouples biological from cultural inheritance) is set to True:
        if decoupled == True:
            bio_parent_per_learner = np.random.choice(self.population, size=n_agents_to_be_replaced, p=parent_probs)
        else:
            bio_parent_per_learner = self.population[self.parent_index_per_learner]
        pragmatic_level_bio_parent_per_learner = [bio_parent.pragmatic_level for bio_parent in bio_parent_per_learner]
        optimality_alpha_bio_parent_per_learner = [bio_parent.optimality_alpha for bio_parent in bio_parent_per_learner]

        # 1.2) Then we generate data_dict from the current population:
        data_per_agent = []
        for i in range(n_agents_to_be_replaced):
            if self.context_generation == 'random':
                context_matrix = gen_context_matrix(self.context_type, self.n_meanings, self.context_size, self.n_contexts)
            elif self.context_generation == 'only_helpful':
                context_matrix = gen_helpful_context_matrix(self.n_meanings, self.n_contexts, self.helpful_contexts)
            elif self.context_generation == 'optimal':
                context_matrix = gen_helpful_context_matrix_fixed_order(self.n_meanings, self.n_contexts, self.helpful_contexts)

            speaker_order = create_speaker_order_iteration(self, selection_type, parent_probs, teacher_type, self.n_contexts)

            if len(speaker_order) > 0:
                self.parent_index_per_learner[i] = speaker_order[0].id
            else:
                self.parent_index_per_learner[i] = 0

            if context_generation == 'random':
                old_pop_data = self.produce_pop_data(context_matrix, self.n_utterances, speaker_order)
            elif context_generation == 'optimal':
                old_pop_data = self.produce_pop_data_fixed_contexts(context_matrix, self.n_utterances, speaker_order, helpful_contexts, n_signals)
            data_per_agent.append(old_pop_data)
        self.parent_generation = self.population
        self.parent_lex_indices = np.zeros(self.size)
        for a in range(self.size):
            parent_index = self.parent_index_per_learner[a]
            self.parent_lex_indices[a] = self.lex_indices_per_agent[parent_index]
        selected_hyp_per_agent_matrix = np.zeros(n_agents_to_be_replaced)
        normalized_log_posteriors_per_agent_matrix = np.zeros((n_agents_to_be_replaced, len(self.hypothesis_space)))

        for i in range(n_agents_to_be_replaced):
            # 1.3) We choose the new agent's perspective and learning_type with uniform probability from the attributes self.perspective_probs and self.learning_probs:
            new_agent_perspective = np.random.choice(self.perspectives, size=1, p=self.perspective_probs)
            new_agent_learning_type = np.random.choice(self.learning_types, size=1, p=self.learning_type_probs)
            new_agent_lexicon = Lexicon('empty_lex', self.n_meanings, self.n_signals)

            # 1.4) Then we initialize the new agent with that perspective and learning_type and with an empty lexicon
            #FIXME: Again: the new agent is initialized with attributes that are globally defined in the params_and_run module ()
            perspective_prior = prior.create_perspective_prior(self.perspective_hyps, self.lexicon_hyps, self.perspective_prior_type, self.learner_perspective, self.perspective_prior_strength)
            lexicon_prior = prior.create_lexicon_prior(self.lexicon_hyps, self.lexicon_prior_type, self.lexicon_prior_strength, self.error)
            composite_log_priors = prior.list_composite_log_priors(self.agent_type, self.size, self.hypothesis_space, self.perspective_hyps, self.lexicon_hyps, perspective_prior, lexicon_prior)

            if pragmatic_level_bio_parent_per_learner[i] == 'literal' or pragmatic_level_bio_parent_per_learner[i] == 'perspective-taking':
                new_agent = Agent(self.perspective_hyps, self.lexicon_hyps, composite_log_priors, composite_log_priors, new_agent_perspective[0], self.sal_alpha, new_agent_lexicon, new_agent_learning_type[0])
            elif pragmatic_level_bio_parent_per_learner[i] == 'prag':
                new_agent = PragmaticAgent(self.perspective_hyps, self.lexicon_hyps, composite_log_priors, composite_log_priors, new_agent_perspective[0], self.sal_alpha, new_agent_lexicon, new_agent_learning_type[0], pragmatic_level_bio_parent_per_learner[i], pragmatic_level_bio_parent_per_learner[i], optimality_alpha_bio_parent_per_learner[i], self.extra_error)

            # 1.5) Then we get the new agent's parent data from the old population:
            agent_data = data_per_agent[i]

            # 1.6) We subsequently let the new agent learn from the annotated_pop_data of the population and update its lexicon accordingly:


            normalized_log_posteriors_per_agent_matrix[i] = new_agent.inference_on_signal_counts_data(agent_data, error)

            selected_hyp, selected_lex_hyp = new_agent.update_lexicon()
            selected_hyp_per_agent_matrix[i] = selected_hyp
            self.lex_indices_per_agent[i] = selected_lex_hyp


            # 1.7) Then, we remove the oldest agent (c.e. agent with index 0) from the population, and append the new agent at the end:
            self.population = np.delete(self.population, 0)
            self.population = np.append(self.population, new_agent)  # Appends the new agent to the end of the population
            self.perspectives_per_agent = np.delete(self.perspectives_per_agent, 0)
            self.perspectives_per_agent = np.append(self.perspectives_per_agent, new_agent.perspective)
            self.lexicons_per_agent = np.delete(self.lexicons_per_agent, 0)
            self.lexicons_per_agent = np.append(self.population, new_agent.lexicon)
            self.learning_types_per_agent = np.delete(self.learning_types_per_agent, 0)
            self.learning_types_per_agent = np.append(self.population, new_agent.learning_type)

        # 1.8) Finally, the agent id numbers are updated:
        for i in range(self.size):
            agent = self.population[i]
            agent.id = i
            pragmatic_level_per_agent[i] = agent.pragmatic_level

        return selected_hyp_per_agent_matrix, avg_fitness, parent_probs, self.parent_index_per_learner, self.parent_lex_indices, pragmatic_level_per_agent






class DistinctionPopulation(Population):
    """
    A Population object consists of a list of Agent objects
    """
    def __init__(self, size, n_meanings, n_signals, hypothesis_space, perspective_hyps, lexicon_hyps, learner_perspective, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_strength, composite_log_prior, perspectives, perspectives_per_agent, perspective_probs, alpha, lexicons, lexicons_per_agent, production_error, n_contexts, context_type, context_generation, context_size, helpful_contexts, n_utterances, learning_types, learning_types_per_agent, learning_type_probs):
        """
        :param size: The size of the population (integer)
        :param agent_type: The type of agents (can be set to either 'p_distinction' or 'no_p_distinction')
        :param perspectives: The perspectives that the agents can have
        :param perspectives_per_agent: A 1D numpy array of length size, with for each agent index the perspective that that agent will have
        :param lexicons: The lexicons that the agents can have
        :param lexicons_per_agent: A 1D numpy array of length size, with for each agent index the lexicon that that agent will have
        :param learning_types: The learning types that the agents can have
        :param learning_types_per_agent: A 1D numpy array of length size, with for each agent index the learning type that that agent will have
        :return: Creates a population with the specified attributes
        """
        self.pop_type = 'singular'
        self.size = size
        self.agent_type = 'p_distinction'
        self.n_meanings = n_meanings
        self.n_signals = n_signals
        self.hypothesis_space = hypothesis_space
        self.perspective_hyps = perspective_hyps
        self.lexicon_hyps = lexicon_hyps
        self.learner_perspective = learner_perspective
        self.perspective_prior_type = perspective_prior_type
        self.perspective_prior_strength = perspective_prior_strength
        self.lexicon_prior_type = lexicon_prior_type
        self.lexicon_prior_strength = lexicon_prior_strength
        self.composite_log_prior = composite_log_prior
        self.perspectives = perspectives
        self.perspectives_per_agent = perspectives_per_agent
        self.perspective_probs = perspective_probs
        self.alpha = alpha
        self.lexicons = lexicons
        self.lexicons_per_agent = lexicons_per_agent
        self.lex_indices_per_agent = np.zeros(size)
        self.error = production_error
        self.n_contexts = n_contexts
        self.context_type = context_type
        self.context_generation = context_generation
        self.context_size = context_size
        self.helpful_contexts = helpful_contexts
        self.n_utterances = n_utterances
        self.parent_index_per_learner = np.zeros(size)
        self.parent_generation = []
        self.parent_lex_indices = np.zeros(size)
        self.learning_types = learning_types
        self.learning_types_per_agent = learning_types_per_agent
        self.learning_type_probs = learning_type_probs
        self.population = self.create_pop()

    def create_pop(self):
        """
        :return: A list containing all the Agent objects in the population
        """
        population = []
        for i in range(self.size):
            perspective = self.perspectives_per_agent[i]
            lexicon = self.lexicons_per_agent[i]
            learning_type = self.learning_types_per_agent[i]
            #FIXME: A bit strange that the default attributes for creating an agent have to be specified as parameters in the params_and_run module, is it not..? (see all three if and elif statements below)
            agent = DistinctionAgent(self.perspective_hyps, self.lexicon_hyps, self.composite_log_prior, self.composite_log_prior, perspective, self.alpha, lexicon, learning_type, self.size)
            agent.id = int(i)
            population.append(agent)
        return population


    def produce_pop_data(self, n_meanings, n_signals, error, context_matrix, n_utterances, speaker_order):
        """
        :param context_matrix: A 2D numpy matrix of contexts
        :param n_utterances: Global variable determining the number of utterances produced per context (float)
        :param parent_type: This can be set to either 'single_parent' or 'multi_parent'
        :param speaker_order: A 1D numpy array containing speaker indices of length n_contexts. For each context, the agent with the ID on the corresponding index in speaker_order_same_first will be the speaker. (The reason for this is that the order of speakers has to be fixed over runs. Only when the speaker is the same on each context index over the whole set of runs can the mean, median and quartiles be calculated accurately. Another reason is that fixing the speaker_order_same_first allows for manipulating the staging of the input data_dict.)
        :return: A data_dict object produced by the population, for which speakers have been chosen from the population with uniform probability
        """
        pop_topic_matrix = np.zeros((len(context_matrix), n_utterances))
        pop_utterance_matrix = np.zeros((len(context_matrix), n_utterances))
        speaker_id_matrix = np.zeros(len(context_matrix))
        for c in range(len(context_matrix)):
            context = context_matrix[c]
            speaker = speaker_order[c]
            speaker_data = speaker.produce_data(self.n_meanings, self.n_signals, np.array([context]), self.n_utterances, self.error)
            speaker_topics = speaker_data.topics[0]
            speaker_utterances = speaker_data.utterances[0]
            pop_topic_matrix[c] = speaker_topics
            pop_utterance_matrix[c] = speaker_utterances
            #TODO: Note that at the moment there is only one speaker per context
            #TODO: The speaker_id_matrix can probably be generated more efficiently now that we feed a fixed speaker_order_same_first array into this method.
            speaker_id_matrix[c] = speaker.id
            pop_data = SpeakerAnnotatedData(context_matrix, pop_topic_matrix, pop_utterance_matrix, speaker_id_matrix)
        return pop_data


    def pop_update(self, recording, n_meanings, n_signals, error, turnover_type, selection_type, selection_weighting, communication_type, ca_measure_type, n_interactions, teacher_type, speaker_order_type, first_input_stage_ratio, perspectives_per_agent=None):
        """
        :param turnover_type: 'chain' for one agent at a time, or 'whole_pop' for the whole population at once
        :return: Doesn't return anything, but changes self.population to the new population
        """
        #TODO: Check that this is equivalent to the method in the Population object!

        parent_fitness_array = self.calc_fitness(selection_type, selection_weighting, communication_type, ca_measure_type, n_interactions, self.parent_index_per_learner, self.parent_generation, self.parent_lex_indices)
        avg_fitness = np.mean(parent_fitness_array)

        if selection_weighting == 'none':
            parent_probs = np.divide(parent_fitness_array, np.sum(parent_fitness_array))
        elif isinstance(selection_weighting, float):
            parent_fitness_array_weighted = np.multiply(parent_fitness_array, selection_weighting)
            parent_fitness_array_weighted_exp = np.exp(parent_fitness_array_weighted)
            parent_probs = np.divide(parent_fitness_array_weighted_exp, np.sum(parent_fitness_array_weighted_exp))


        if turnover_type == 'chain':
            n_agents_to_be_replaced = 1
        elif turnover_type == 'whole_pop':
            n_agents_to_be_replaced = self.size

        # 1) For the number of agents that need to be replaced in the population, we do the following:
        # FIXME: Figure out why the prior and posterior of ALL agents is updated in the chain method, when it should update only those of the new agent

        data_per_agent = []
        for i in range(n_agents_to_be_replaced):

            # 1.1) Then we generate a data_dict from the current population:
            if self.context_generation == 'random':
                context_matrix = gen_context_matrix(self.context_type, self.n_meanings, self.context_size, self.n_contexts)
            elif self.context_generation == 'only_helpful':
                context_matrix = gen_helpful_context_matrix(self.n_meanings, self.n_contexts, self.helpful_contexts)
            elif self.context_generation == 'optimal':
                context_matrix = gen_helpful_context_matrix_fixed_order(self.n_meanings, self.n_contexts, self.helpful_contexts)

            speaker_order = create_speaker_order_iteration(self, selection_type, parent_probs, teacher_type, self.n_contexts)
            if teacher_type == 'single_parent':
                self.parent_index_per_learner[i] = speaker_order[0].id

            old_pop_data = self.produce_pop_data(n_meanings, n_signals, error, context_matrix, self.n_utterances, speaker_order)
            data_per_agent.append(old_pop_data)

        self.parent_generation = self.population
        self.parent_lex_indices = np.zeros(self.size)
        for a in range(self.size):
            parent_index = self.parent_index_per_learner[a]
            self.parent_lex_indices[a] = self.lex_indices_per_agent[parent_index]

        selected_hyp_per_agent_matrix = np.zeros(n_agents_to_be_replaced)
        normalized_log_posteriors_per_data_point_per_agent_matrix = np.zeros((n_agents_to_be_replaced, (self.n_contexts + 1), len(self.hypothesis_space)))
        for i in range(n_agents_to_be_replaced):

            # 1.2) We choose the new agent's perspective and learning_type with uniform probability from the attributes self.perspective_probs and self.learning_probs:
            new_agent_perspective = np.random.choice(self.perspectives, size=1, p=self.perspective_probs)
            new_agent_learning_type = np.random.choice(self.learning_types, size=1, p=self.learning_type_probs)
            new_agent_lexicon = Lexicon('empty_lex', self.n_meanings, self.n_signals)

            # 1.3) Then we initialize the new agent with that perspective and learning_type and with an empty lexicon
            # FIXME: Again: the new agent is initialized with attributes that are globally defined in the params_and_run module ()
            perspective_prior = prior.create_perspective_prior(self.perspective_hyps, self.lexicon_hyps, self.perspective_prior_type, self.learner_perspective, self.perspective_prior_strength)
            lexicon_prior = prior.create_lexicon_prior(self.lexicon_hyps, self.lexicon_prior_type, self.lexicon_prior_strength, self.error)
            composite_log_priors = prior.list_composite_log_priors_with_speaker_distinction(self.hypothesis_space, self.perspective_hyps, self.lexicon_hyps, perspective_prior, lexicon_prior, self.size)
            new_agent = DistinctionAgent(self.perspective_hyps, self.lexicon_hyps, composite_log_priors, composite_log_priors, new_agent_perspective[0], self.alpha, new_agent_lexicon, new_agent_learning_type[0], self.size)

            # 1.4) Then we get the new agent's parent data from the old population:
            agent_data = data_per_agent[i]

            # 1.5) We subsequently let the new agent learn from the annotated_pop_data of the population and update its lexicon accordingly:
            normalized_log_posteriors_per_data_point_per_agent_matrix[i] = new_agent.inference(self.n_contexts, self.n_utterances, old_pop_data, error)
            selected_hyp, selected_lex_hyp = new_agent.update_lexicon()
            selected_hyp_per_agent_matrix[i] = selected_hyp
            self.lex_indices_per_agent[i] = selected_lex_hyp

            # 1.6) Then we remove the oldest agent (c.e. agent with index 0) from the population, and append the new agent at the end:
            self.population = np.delete(self.population, 0)
            self.population = np.append(self.population, new_agent)  # Appends the new agent to the end of the population
            self.perspectives_per_agent = np.delete(self.perspectives_per_agent, 0)
            self.perspectives_per_agent = np.append(self.perspectives_per_agent, new_agent.perspective)
            self.lexicons_per_agent = np.delete(self.lexicons_per_agent, 0)
            self.lexicons_per_agent = np.append(self.population, new_agent.lexicon)
            self.learning_types_per_agent = np.delete(self.learning_types_per_agent, 0)
            self.learning_types_per_agent = np.append(self.population, new_agent.learning_type)

        # 1.7) Finally, the agent id numbers are updated:
        for i in range(self.size):
            agent = self.population[i]
            agent.id = i

        if recording == 'minimal':
            return selected_hyp_per_agent_matrix, avg_fitness, parent_probs, self.parent_index_per_learner, self.parent_lex_indices
        elif recording == 'everything':
            return selected_hyp_per_agent_matrix, avg_fitness, parent_probs, self.parent_index_per_learner, self.parent_lex_indices, normalized_log_posteriors_per_data_point_per_agent_matrix


