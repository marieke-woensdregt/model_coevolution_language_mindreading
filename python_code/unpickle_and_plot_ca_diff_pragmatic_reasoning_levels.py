__author__ = 'Marieke Woensdregt'


import numpy as np
from hypspace import create_all_lexicons
import pickle
import plots
import matplotlib.pyplot as plt
import seaborn as sns




############   PARAMETER SETTINGS:
n_meanings = 3
n_signals = 3
lexicon_hyps = create_all_lexicons(n_meanings, n_signals) # The lexicon hypotheses
error = 0.05
# lexicon = np.array([[1., 1., 0.], [0., 1., 0.], [0., 0., 1.]])
# context = np.array([0.1, 0.2, 0.9])
sal_alpha = 1.0
optimality_alpha = 3.0  # Goodman & Stuhlmuller (2013) fitted sal_alpha = 3.4 to participant data with 4x3 lexicon of three number words ('one', 'two', and 'three') that could be mapped to four different world states (0, 1, 2, and 3)
perspective_hyps = np.array([0., 1.])
context_generation = 'random'
helpful_contexts = np.array([[0.1, 0.2, 0.9], [0.1, 0.8, 0.9],
                                 [0.1, 0.9, 0.2], [0.1, 0.9, 0.8],
                                 [0.2, 0.1, 0.9], [0.8, 0.1, 0.9],
                                 [0.2, 0.9, 0.1], [0.8, 0.9, 0.1],
                                 [0.9, 0.1, 0.2], [0.9, 0.1, 0.8],
                                 [0.9, 0.2, 0.1], [0.9, 0.8, 0.1]])
n_utterances = 1
n_interactions = 10000
lex_measure = 'ca'  # This can be set to either 'ca' or 'mi'
ca_calculation_type = 'numerical'  # This can be set to either 'numerical' or 'simulation'
results_directory = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Pickles/Learner_Speaker/'
#################################################################







############   PLOTTING:

plot_file_path = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Plots/Learner_Speaker/'

plot_file_title = 'Plot_diff_pragmatic_reasoning_levels'

plot_title = 'Mean ca for different levels of pragmatic reasoning'

def plot_ca_diff_pragmatic_reasoning_levels(plot_file_path, plot_file_title, plot_title, lexicon_matrix, key_values_sorted, optimality_alpha, ca_dict_list, labels, text_size, context_generation, n_interactions):
    sns.set_style("whitegrid")
    sns.set(font_scale=text_size)
    sns.set_palette("deep", n_colors=5)
    colors = [sns.color_palette()[0], sns.color_palette()[2], sns.color_palette()[1], sns.color_palette()[3], sns.color_palette()[4]]
    with sns.axes_style("whitegrid", {'xtick.major.size': 8,
 'xtick.minor.size': 8}):
        fig = plt.figure(figsize=(11, 4.8))
        ax = fig.add_subplot(111)
        for i in range(len(ca_dict_list)):
            ax.plot(np.arange(len(key_values_sorted)), ca_dict_list[i], 'o', color=colors[i], label=labels[i])
        ## Set x_ticks and y_ticks:
        # ax.set_xlim(0.3, 0.935)
        ax.set_xlim(-0.5, len(key_values_sorted) - 0.5)
        ax.set_xticks(np.arange(len(key_values_sorted)))
        xticklabels = [str(key)[1:4] for key in key_values_sorted]
        ax.set_xticklabels(xticklabels)
        if lex_measure == 'ca':
            ax.set_ylim(0.0, 1.05)
            ax.set_yticks(np.arange(0.0, 1.05, 0.1))
        # elif lex_measure == 'mi':
            # ax.set_ylim(-0.05, 1.3)
            # ax.set_yticks(np.arange(0.0, 1.3, 0.1))
        plt.gcf().subplots_adjust(bottom=0.15)  # This makes room for the xlabel which would otherwise be cut off because of the xtick labels
        # Shrink current axis by 88%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.88, box.height])
        # Put a legend to the right of the current axis
        legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, fontsize=12)
        legend.get_frame().set_linewidth(1.5)
    sns.despine()
    plt.xlabel('informativeness of lexicon')
    plt.ylabel(lex_measure+' between speaker and listener')
    plt.title(plot_title)
    # Then each bar gets the corresponding lexicon plotted above it:
    factor = len(lexicon_matrix)  # I found this factor through trial and error, I have no idea why it works.
    lex_height = 1. / factor
    lex_width = lex_height*0.55
    for l in range(len(lexicon_matrix)):
        plt.axes([(0.081 + ((lex_width+0.0094)*(l+1))), 0.15, lex_width, lex_height])
        plt.pcolor(lexicon_matrix[l], cmap=plt.cm.GnBu, edgecolors='k', linewidths=1, vmin=0)
        plt.gca().invert_yaxis()
        plt.xticks([])
        plt.yticks([])
    plt.savefig(plot_file_path+plot_file_title+'_'+context_generation+'_'+str(n_interactions)+'_interactions_'+lex_measure+'_'+'_alpha_'+str(optimality_alpha)+'.png')
    plt.show()




if __name__ == "__main__":

    lexicon_matrix_pickle = results_directory+'lexicon_matrix_'+lex_measure+'.p'
    lexicon_matrix = pickle.load(open(lexicon_matrix_pickle, 'rb'))
    print ' '
    print "lexicon_matrix is:"
    print lexicon_matrix



    key_list_pickle = results_directory+'key_list_sorted_'+lex_measure+'.p'
    key_list_sorted = pickle.load(open(key_list_pickle, 'rb'))
    print ''
    print "key_list_sorted is:"
    print key_list_sorted



    ca_s0p0_l0_per_inf_level_array_pickle = results_directory+'ca_s0p0_l0_per_inf_level_array_'+context_generation+'_'+str(n_interactions)+'_interactions_'+lex_measure+'_'+ca_calculation_type+'.p'
    ca_s0p0_l0_per_inf_level_array = pickle.load(open(ca_s0p0_l0_per_inf_level_array_pickle, 'rb'))
    print ' '
    print "ca_s0p0_l0_per_inf_level_array is:"
    print ca_s0p0_l0_per_inf_level_array


    ca_s0p1_l0_per_inf_level_array_pickle = results_directory+'ca_s0p1_l0_per_inf_level_array_'+context_generation+'_'+str(n_interactions)+'_interactions_'+lex_measure+'_'+ca_calculation_type+'.p'
    ca_s0p1_l0_per_inf_level_array = pickle.load(open(ca_s0p1_l0_per_inf_level_array_pickle, 'rb'))
    print ' '
    print "ca_s0p1_l0_per_inf_level_array is:"
    print ca_s0p1_l0_per_inf_level_array




    ca_s0p1_l1p0_per_inf_level_array_pickle = results_directory+'ca_s0p1_l1p0_per_inf_level_array_'+context_generation+'_'+str(n_interactions)+'_interactions_'+lex_measure+'_'+ca_calculation_type+'.p'
    ca_s0p1_l1p0_per_inf_level_array = pickle.load(open(ca_s0p1_l1p0_per_inf_level_array_pickle, 'rb'))
    print ' '
    print "ca_s0p1_l1p0_per_inf_level_array is:"
    print ca_s0p1_l1p0_per_inf_level_array



    ca_s0p1_l1p1_per_inf_level_array_pickle = results_directory+'ca_s0p1_l1p1_per_inf_level_array_'+context_generation+'_'+str(n_interactions)+'_interactions_'+lex_measure+'_'+ca_calculation_type+'.p'
    ca_s0p1_l1p1_per_inf_level_array = pickle.load(open(ca_s0p1_l1p1_per_inf_level_array_pickle, 'rb'))
    print ' '
    print "ca_s0p1_l1p1_per_inf_level_array is:"
    print ca_s0p1_l1p1_per_inf_level_array


    #
    # ca_s1p1_l2p0_per_inf_level_array_alpha_1_pickle = output_pickle_file_directory+'ca_s1p1_l2p0_per_inf_level_array_alpha_1_'+context_generation+'_'+str(n_interactions)+'_interactions_'+lex_measure+'_'+ca_calculation_type+'.p'
    # ca_s1p1_l2p0_per_inf_level_array_alpha_1 = pickle.load(open(ca_s1p1_l2p0_per_inf_level_array_alpha_1_pickle, 'rb'))
    # print ' '
    # print "ca_s1p1_l2p0_per_inf_level_array_alpha_1 is:"
    # print ca_s1p1_l2p0_per_inf_level_array_alpha_1
    #
    #
    #
    #
    # ca_s1p1_l2p1_per_inf_level_array_alpha_1_pickle = output_pickle_file_directory+'ca_s1p1_l2p1_per_inf_level_array_alpha_1_'+context_generation+'_'+str(n_interactions)+'_interactions_'+lex_measure+'_'+ca_calculation_type+'.p'
    # ca_s1p1_l2p1_per_inf_level_array_alpha_1 = pickle.load(open(ca_s1p1_l2p1_per_inf_level_array_alpha_1_pickle, 'rb'))
    # print ' '
    # print "ca_s1p1_l2p1_per_inf_level_array_alpha_1 is:"
    # print ca_s1p1_l2p1_per_inf_level_array_alpha_1
    #
    #
    #
    #
    #
    # ca_s1p1_l2p0_per_inf_level_array_alpha_3_pickle = output_pickle_file_directory+'ca_s1p1_l2p0_per_inf_level_array_alpha_3_'+context_generation+'_'+str(n_interactions)+'_interactions_'+lex_measure+'_'+ca_calculation_type+'.p'
    # ca_s1p1_l2p0_per_inf_level_array_alpha_3 = pickle.load(open(ca_s1p1_l2p0_per_inf_level_array_alpha_3_pickle, 'rb'))
    # print ' '
    # print "ca_s1p1_l2p0_per_inf_level_array_alpha_3 is:"
    # print ca_s1p1_l2p0_per_inf_level_array_alpha_3
    #
    #
    #
    #
    # ca_s1p1_l2p1_per_inf_level_array_alpha_3_pickle = output_pickle_file_directory+'ca_s1p1_l2p1_per_inf_level_array_alpha_3_'+context_generation+'_'+str(n_interactions)+'_interactions_'+lex_measure+'_'+ca_calculation_type+'.p'
    # ca_s1p1_l2p1_per_inf_level_array_alpha_3 = pickle.load(open(ca_s1p1_l2p1_per_inf_level_array_alpha_3_pickle, 'rb'))
    # print ' '
    # print "ca_s1p1_l2p1_per_inf_level_array_alpha_3 is:"
    # print ca_s1p1_l2p1_per_inf_level_array_alpha_3
    #



    ca_s1p1_l2p0_per_inf_level_array_alpha_0_pickle = results_directory+'ca_s1p1_l2p0_per_inf_level_array_alpha_0_'+context_generation+'_'+str(n_interactions)+'_interactions_'+lex_measure+'_'+ca_calculation_type+'.p'
    ca_s1p1_l2p0_per_inf_level_array_alpha_0 = pickle.load(open(ca_s1p1_l2p0_per_inf_level_array_alpha_0_pickle, 'rb'))
    print ' '
    print "ca_s1p1_l2p0_per_inf_level_array_alpha_0 is:"
    print ca_s1p1_l2p0_per_inf_level_array_alpha_0




    ca_s1p1_l2p1_per_inf_level_array_alpha_0_pickle = results_directory+'ca_s1p1_l2p1_per_inf_level_array_alpha_0_'+context_generation+'_'+str(n_interactions)+'_interactions_'+lex_measure+'_'+ca_calculation_type+'.p'
    ca_s1p1_l2p1_per_inf_level_array_alpha_0 = pickle.load(open(ca_s1p1_l2p1_per_inf_level_array_alpha_0_pickle, 'rb'))
    print ' '
    print "ca_s1p1_l2p1_per_inf_level_array_alpha_0 is:"
    print ca_s1p1_l2p1_per_inf_level_array_alpha_0





    ca_s1p1_l2p0_per_inf_level_array_alpha_3_pickle = results_directory+'ca_s1p1_l2p0_per_inf_level_array_alpha_3_'+context_generation+'_'+str(n_interactions)+'_interactions_'+lex_measure+'_'+ca_calculation_type+'.p'
    ca_s1p1_l2p0_per_inf_level_array_alpha_3 = pickle.load(open(ca_s1p1_l2p0_per_inf_level_array_alpha_3_pickle, 'rb'))
    print ' '
    print "ca_s1p1_l2p0_per_inf_level_array_alpha_3 is:"
    print ca_s1p1_l2p0_per_inf_level_array_alpha_3




    ca_s1p1_l2p1_per_inf_level_array_alpha_3_pickle = results_directory+'ca_s1p1_l2p1_per_inf_level_array_alpha_3_'+context_generation+'_'+str(n_interactions)+'_interactions_'+lex_measure+'_'+ca_calculation_type+'.p'
    ca_s1p1_l2p1_per_inf_level_array_alpha_3 = pickle.load(open(ca_s1p1_l2p1_per_inf_level_array_alpha_3_pickle, 'rb'))
    print ' '
    print "ca_s1p1_l2p1_per_inf_level_array_alpha_3 is:"
    print ca_s1p1_l2p1_per_inf_level_array_alpha_3




    ############   PLOTTING:

    plot_file_path = '/Users/pplsuser/Documents/PhD_Edinburgh/My_Modelling/Bayesian_Lang_n_ToM/Results/Plots/Learner_Speaker/'

    plot_file_title = 'Plot_diff_pragmatic_reasoning_levels_sp_literal_lr_pragmatic_'

    plot_title = 'Mean ca for different levels of pragmatic reasoning'



    labels = ['s0p1 + l0p1', 's0p1 + l1p0', 's0p1 + l1p1', 's0p1 + l2p0', 's0p1 + l2p1']


    plot_ca_diff_pragmatic_reasoning_levels(plot_file_path, plot_file_title, plot_title, lexicon_matrix, key_list_sorted, 3.0, ca_s0p1_l0_per_inf_level_array, ca_s0p1_l1p0_per_inf_level_array, ca_s0p1_l1p1_per_inf_level_array, ca_s1p1_l2p0_per_inf_level_array_alpha_3,  ca_s1p1_l2p1_per_inf_level_array_alpha_3, labels, 1.7, context_generation, n_interactions)



