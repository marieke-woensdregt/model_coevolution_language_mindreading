__author__ = 'Marieke Woensdregt'

import numpy as np
from scipy.misc import logsumexp
import string



#TODO: If the hypothesis space over perspectives increases, this function should perhaps work with logs instead of raw probabilities
def create_perspective_prior(perspective_hyps, lexicon_hyps, perspective_prior_type, learner_perspective, perspective_prior_strength):
    """
    :param perspective_hyps: A 1D numpy array containing the perspective hypotheses.
    :param lexicon_hyps: A 3D numpy array containing the lexicon hypotheses.
    :param perspective_prior_type: The type of perspective prior. This can be set to either 'neutral', 'egocentric' or 'same_as_lexicon' (in which case it is also egocentric, and the prior over the opposite perspective is equal to the prior of a particular lexicon given a neutral lexicon prior).
    :learner_perspective: The perspective of the learner agent
    :param perspective_prior_strength: The strength of the egocentric prior. (This is only used in case the first parameter is set to 'egocentric'.)
    :return: A 1D numpy array containing the prior probabilities corresponding by index to the perspective hypotheses listed in 'perspective_hyps'.
    """
    if perspective_prior_type == 'neutral':
        # full_perspective_prior = np.full(len(perspective_hyps), (1./len(perspective_hyps)), dtype=float) # This creates a neutral prior over all perspective hypotheses (1D numpy array)
        full_perspective_prior = [(1. / len(perspective_hyps)) for x in range(len(perspective_hyps))]
        full_perspective_prior = np.asarray(full_perspective_prior)
    elif perspective_prior_type == 'egocentric':
        full_perspective_prior = np.zeros(len(perspective_hyps))
        for i in range(len(perspective_hyps)):
            perspective = perspective_hyps[i]
            if perspective == learner_perspective:
                perspective_prior = perspective_prior_strength
            else:
                perspective_prior = ((1.-perspective_prior_strength)/(len(perspective_hyps)-1))
            full_perspective_prior[i] = perspective_prior
    elif perspective_prior_type == 'same_as_lexicon':
        perspective_prior_same_as_lexicon = 1./len(lexicon_hyps)
        perspective_1_prior = perspective_prior_same_as_lexicon
        perspective_0_prior = 1.-perspective_prior_same_as_lexicon
        full_perspective_prior = np.array([perspective_0_prior, perspective_1_prior])
    elif perspective_prior_type == 'zero_order_tom':
        full_perspective_prior = np.zeros(len(perspective_hyps))
        for i in range(len(perspective_hyps)):
            perspective = perspective_hyps[i]
            if perspective == learner_perspective:
                perspective_prior = 1.
            else:
                perspective_prior = 0.
            full_perspective_prior[i] = perspective_prior
    return full_perspective_prior



def calc_expressivity_prior(lexicon_hyps, lexicon_prior_constant, error_term):
    mutual_info_per_lexicon = np.zeros(len(lexicon_hyps))
    for l in range(len(lexicon_hyps)):
        lexicon = lexicon_hyps[l]
        signal_probs = np.zeros_like(lexicon)
        for m in range(len(signal_probs)):
            meaning_row = lexicon[m]
            new_signal_probs = np.zeros_like(meaning_row)
            associated_signals_indices = np.where(meaning_row == 1.)[0]
            unassociated_signals_indices = np.where(meaning_row == 0.)[0]
            new_signal_probs[associated_signals_indices] = np.divide((1. - error_term), len(associated_signals_indices))
            new_signal_probs[unassociated_signals_indices] = np.divide(error_term, len(unassociated_signals_indices))
            if len(unassociated_signals_indices) == 0:
                new_signal_probs[associated_signals_indices] += np.divide(error_term, len(associated_signals_indices))
            signal_probs[m] = new_signal_probs
        joint_probs = np.divide(signal_probs, np.sum(signal_probs))
        column_sums = joint_probs.sum(axis=0)
        row_sums = joint_probs.sum(axis=1)
        mutual_info_per_signal = np.zeros_like(lexicon)
        for m in range(len(joint_probs)):
            for s in range(len(meaning_row)):
                joint_prob_s = joint_probs[m][s]
                mutual_info_s = joint_prob_s * np.log2(joint_prob_s / (row_sums[m] * column_sums[s]))
                mutual_info_per_signal[m][s] = mutual_info_s
        mutual_info_lex = np.sum(mutual_info_per_signal)
        mutual_info_per_lexicon[l] = mutual_info_lex
    mi_plus_constant_per_lexicon = np.add(mutual_info_per_lexicon, lexicon_prior_constant)
    expressivity_prior = np.divide(mi_plus_constant_per_lexicon, np.sum(mi_plus_constant_per_lexicon))
    return expressivity_prior



def calc_compressibility_prior(lexicon_hyps, lexicon_prior_constant):
    meaning_symbols = list(string.ascii_lowercase)
    all_lex_bits = []
    all_lex_bits_summed = np.zeros(len(lexicon_hyps))
    for l in range(len(lexicon_hyps)):
        lex_description = ''
        lexicon = lexicon_hyps[l]
        lexicon_transposed = lexicon.T
        sentence_description = ''
        for s in range(len(lexicon_transposed)):
            signal_column = lexicon_transposed[s]
            meaning_string = ''
            for m in range(len(signal_column)):
                meaning = signal_column[m]
                if meaning == 1:
                    if len(meaning_string)==0:
                        meaning_string = meaning_string+meaning_symbols[m]
                    elif len(meaning_string)>0:
                        meaning_string = meaning_string+','+meaning_symbols[m]
            mapping_description = meaning_string+str(s)
            if sum(signal_column) > 0:
                if len(sentence_description) == 0:
                    sentence_description = 'S'+mapping_description
                else:
                    sentence_description = sentence_description+'.S'+mapping_description
        lex_description = lex_description+sentence_description
        symbol_counts = []
        for symbol in lex_description:
            symbol_count = lex_description.count(symbol)
            symbol_counts.append(symbol_count)
        symbol_probabilities = np.divide(np.array(symbol_counts).astype(float), float(len(lex_description)))
        lex_bits = -np.log2(symbol_probabilities)
        all_lex_bits.append(lex_bits)
        all_lex_bits_summed[l] = sum(lex_bits)
    all_lex_bare_prior_p = np.power(2, -all_lex_bits_summed)
    all_lex_prior_plus_constant = np.add(all_lex_bare_prior_p, lexicon_prior_constant)
    lex_prior_normalized = np.divide(all_lex_prior_plus_constant, np.sum(all_lex_prior_plus_constant))
    return lex_prior_normalized





#TODO: If the hypothesis space over lexicons increases, this function should perhaps work with logs instead of raw probabilities
def create_lexicon_prior(lexicon_hyps, lexicon_prior_type, lexicon_prior_constant, error_term):
    """
    :param lexicon_hyps: The agent's lexicon hypotheses (3D numpy array)
    :param lexicon_prior_type: The type of lexicon prior. This can be set to either 'neutral', 'ambiguous_fixed', 'half_ambiguous_fixed' or 'one_to_one_bias'.
    :param lexicon_prior_constant: The larger the constant, the more equal the prior probabilities of lexicons with different mutual information values will be (with c = 1000 giving an almost uniform distribution)
    :param error_term: probability that speaker will make error in production
    :return: A 1D numpy array with indices corresponding to lexicons in 'lexicon_hyps' array, with a prior probability value on each index
    """
    if lexicon_prior_type == 'neutral':
        # lexicon_prior = np.full(len(lexicon_hyps), (1./len(lexicon_hyps)), dtype=float) # This creates a neutral prior over all perspective hypotheses (1D numpy array)
        lexicon_prior = [(1./len(lexicon_hyps)) for x in range(len(lexicon_hyps))]
        lexicon_prior = np.asarray(lexicon_prior)
    elif lexicon_prior_type == 'ambiguous_fixed':
        lexicon_prior = np.zeros(len(lexicon_hyps))
        lexicon_prior[-1] = 1. # This creates a fixed prior where ALL prior probability is fixed upon the fully ambiguous lexicon
    elif lexicon_prior_type == 'half_ambiguous_fixed':
        lexicon_prior = np.zeros(len(lexicon_hyps))
        lexicon_prior[2] = 1. # This creates a fixed prior where ALL prior probability is fixed upon one of the 'half ambiguous' lexicons
    elif lexicon_prior_type == 'expressivity_bias':
        lexicon_prior = calc_expressivity_prior(lexicon_hyps, lexicon_prior_constant, error_term)
    elif lexicon_prior_type == 'compressibility_bias':
        lexicon_prior = calc_compressibility_prior(lexicon_hyps, lexicon_prior_constant)
    return lexicon_prior







def list_composite_log_priors(agent_type, pop_size, hypothesis_space, perspective_hyps, lexicon_hyps, perspective_prior, lexicon_prior):
    """
    :param hypothesis_space: The full space of composite hypotheses (2D numpy matrix)
    :param perspective_prior: The prior probability distribution over perspective hypotheses (1D numpy array)
    :param lexicon_prior: The prior probability distribution over lexicon hypotheses (1D numpy array)
    :return: A 1D numpy array that contains the LOG prior for each composite hypothesis (c.e. log(perspective_prior*lexicon_prior))
    """
    priors = np.zeros(len(hypothesis_space))
    counter = 0
    for i in range(len(hypothesis_space)):
        composite_hypothesis = hypothesis_space[i]
        if agent_type == 'no_p_distinction':
            persp_hyp_index = composite_hypothesis[0]
            lex_hyp_index = composite_hypothesis[1]
            prior = perspective_prior[persp_hyp_index]*lexicon_prior[lex_hyp_index]
            log_prior = np.log(prior)
            priors[counter] = log_prior
            counter += 1
        elif agent_type == 'p_distinction':
            for j in range(pop_size):
                persp_hyp_index = composite_hypothesis[0][j]
                lex_hyp_index = composite_hypothesis[1]
            prior = perspective_prior[persp_hyp_index]*lexicon_prior[lex_hyp_index]
            log_prior = np.log(prior)
            priors[counter] = log_prior
            counter += 1
    return priors



def list_composite_log_priors_with_speaker_distinction(hypothesis_space, perspective_hyps, lexicon_hyps, perspective_prior, lexicon_prior, n_speakers):
    """
    :param hypothesis_space: The full space of composite hypotheses (2D numpy matrix)
    :param perspective_hyps: 1D numpy array containing floats which specify the possible perspectives
    :param lexicon_hyps: 3D numpy array containing all lexicon hypotheses that are considered by the learner
    :param perspective_prior: The prior probability distribution over perspective hypotheses (1D numpy array)
    :param lexicon_prior: The prior probability distribution over lexicon hypotheses (1D numpy array)
    :param n_speakers: The number of speakers
    :return: A 1D numpy array that contains the LOG prior for each composite hypothesis (c.e. log(perspective_prior*lexicon_prior))
    """
    n_perspective_combinations = np.power(len(perspective_hyps), n_speakers)
    n_composite_hyp_combinations = n_perspective_combinations*len(lexicon_hyps)
    log_prior_distribution = np.zeros(n_composite_hyp_combinations)
    counter = 0
    for i in range(len(hypothesis_space)):
        composite_hypothesis = hypothesis_space[i]
        log_perspective_prior_for_all_speakers = 0.
        for speaker_index in range(n_speakers):
            if n_speakers > 1:
                speaker_persp_hyp_index = composite_hypothesis[0][speaker_index]
            else:
                speaker_persp_hyp_index = composite_hypothesis[0]
            log_perspective_prior_for_all_speakers += np.log(perspective_prior[speaker_persp_hyp_index])
        lex_hyp_index = composite_hypothesis[1]
        composite_log_prior = log_perspective_prior_for_all_speakers+np.log(lexicon_prior[lex_hyp_index])
        log_prior_distribution[counter] = composite_log_prior
        counter += 1
    log_prior_distribution = np.subtract(log_prior_distribution, logsumexp(log_prior_distribution))
    return log_prior_distribution


