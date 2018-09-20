__author__ = 'pplsuser'


import itertools
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from hypspace import get_sorted_lex_hyp_order
import prior


#############################################################################
# STEP 11: The functions below are used to create plots:


def create_one_to_one_bias_data_for_plotting():
    lexicon_hyps = np.array([[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]], [[1., 1.], [1., 0.]], [[1., 1.], [1., 1.]]])
    diff_bias_strengths_array = np.zeros((6, 4))
    lexicon_prior_bias_0 = prior.create_lexicon_prior(lexicon_hyps, 'one_to_one_bias', 0.0)
    diff_bias_strengths_array[0] = lexicon_prior_bias_0
    lexicon_prior_bias_025 = prior.create_lexicon_prior(lexicon_hyps, 'one_to_one_bias', 0.25)
    diff_bias_strengths_array[1] = lexicon_prior_bias_025
    lexicon_prior_bias_05 = prior.create_lexicon_prior(lexicon_hyps, 'one_to_one_bias', 0.5)
    diff_bias_strengths_array[2] = lexicon_prior_bias_05
    lexicon_prior_bias_075 = prior.create_lexicon_prior(lexicon_hyps, 'one_to_one_bias', 0.75)
    diff_bias_strengths_array[3] = lexicon_prior_bias_075
    lexicon_prior_bias_1 = prior.create_lexicon_prior(lexicon_hyps, 'one_to_one_bias', 1.0)
    diff_bias_strengths_array[4] = lexicon_prior_bias_1
    lexicon_prior_bias_2 = prior.create_lexicon_prior(lexicon_hyps, 'one_to_one_bias', 2.0)
    diff_bias_strengths_array[5] = lexicon_prior_bias_2
    beta_range = np.arange(0., 1.3, 0.25)
    return diff_bias_strengths_array, beta_range


def plot_lex_distribution(plot_file_path, plot_file_title, plot_title, hypothesis_count_proportions, conf_intervals, cut_off_point, text_size):
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    sns.set(font_scale=text_size)
    lex_type_labels = ['Optimal', 'Partly informative', 'Uninformative']
    ind = np.arange(len(lex_type_labels))
    width = 0.25
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots()
        ax.bar(ind+0.125, hypothesis_count_proportions, width, yerr=conf_intervals, error_kw=dict(ecolor='black', lw=1, capsize=5, capthick=1))
    ax.axhline((1./float(len(lex_type_labels))), color='0.6', linestyle='--')
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0.25, 1.25, 2.25])
    ax.set_xticklabels(lex_type_labels)
    ax.set_xlabel('Language types', labelpad=10)
    ax.set_ylabel('Proportion over generations', labelpad=10)
    ax.set_title(plot_title, y=1.05, fontweight='bold')
    plt.gcf().subplots_adjust(bottom=0.15, top=0.85)  # This makes room for the xlabel and title
    plt.savefig(plot_file_path+'Plot_Prop_Hyps'+plot_file_title+'_cutoff_'+str(cut_off_point)+'.png')
    plt.show()



def plot_timecourse_scores_percentiles(plot_title, plot_file_path, plot_file_title, n_data_points, percentiles_composite_hyp_score, percentiles_perspective_score, percentiles_lexicon_hyp_score, x_axis_steps):
    """
    :param plot_title: The title that will be displayed at the top of the graph (i.e. on the plot itself)
    :param plot_file_title: The title that the plot FILE should get (string without file extension)
    :param n_data_points: The number of data_dict points that the learner has received (integer)
    :param percentiles_composite_hyp_score: The percentiles of the composite hypothesis score, calculated over all the runs
    :param percentiles_perspective_score: The percentiles of the perspective hypothesis score, calculated over all the runs
    :param percentiles_lexicon_hyp_score: The percentiles of the lexicon hypothesis score, calculated over all the runs
    :param percentiles_lexicon_approximation: The percentiles of the lexicon APPROXIMATION score, calculated over all the runs
    :return: Creates a plot with the number of observations on the x-axis and the four different performance scores on the y-axis. Saves the plot and then opens it.
    """
    sns.set_style("darkgrid", {"xtick.major.size":4, "ytick.major.size":4})
    sns.set_palette("deep")
    color_composite_hyp = sns.xkcd_rgb["purplish"] #/  "grape purple"/ "darker purple" / "grape"
    color_perspective_hyp = sns.xkcd_rgb["macaroni and cheese"] #/ "desert" / "sand" / "dusty orange" / "orangish" / "light mustard" / "faded orange"
    color_lexicon_hyp = sns.xkcd_rgb["salmon pink"] #sns.xkcd_rgb["teal"] #/ "turquoise blue / "blue green" / "greenish blue"
    color_lexicon_approximation = sns.xkcd_rgb["salmon pink"]
    fig, ax = plt.subplots(1)
    ax.plot(np.arange(n_data_points), percentiles_composite_hyp_score[1], label='Composite hyp.', color=color_composite_hyp)
    ax.fill_between(np.arange(n_data_points), percentiles_composite_hyp_score[0], percentiles_composite_hyp_score[2], facecolor=color_composite_hyp, alpha=0.3)
    ax.plot(np.arange(n_data_points), percentiles_perspective_score[1], label='Perspective hyp.', color=color_perspective_hyp)
    ax.fill_between(np.arange(n_data_points), percentiles_perspective_score[0], percentiles_perspective_score[2], facecolor=color_perspective_hyp, alpha=0.3)
    ax.plot(np.arange(n_data_points), percentiles_lexicon_hyp_score[1], label='Lexicon hyp.', color=color_lexicon_hyp)
    ax.fill_between(np.arange(n_data_points), percentiles_lexicon_hyp_score[0], percentiles_lexicon_hyp_score[2], facecolor=color_lexicon_hyp, alpha=0.3)
    # ax.plot(np.arange(n_data_points), percentiles_lexicon_approximation[1], label='Lexicon approx.', color=color_lexicon_approximation)
    # ax.fill_between(np.arange(n_data_points), percentiles_lexicon_approximation[0], percentiles_lexicon_approximation[2], facecolor=color_lexicon_approximation, sal_alpha=0.3)
    plt.ylim((-0.05, 1.05))
    ax.set_yticks(np.arange(0.05, 1.05, 0.05))
    ax.set_xticks(np.arange(0, n_data_points, x_axis_steps))
    ax.set_title(plot_title)
    ax.set_xlabel("No. observations")
    ax.set_ylabel("Posterior probability")
    ax.legend(loc='best')
    plt.savefig(plot_file_path+'/Timecrse_Scores_Med_Perc_'+plot_file_title+'.png')
    plt.show()




def plot_timecourse_scores_percentiles_without_error_median(plot_title, plot_file_path, plot_file_title, n_data_points, percentiles_composite_hyp_score, percentiles_perspective_score, percentiles_lexicon_hyp_score):
    """
    :param plot_title: The title that will be displayed at the top of the graph (i.e. on the plot itself)
    :param plot_file_title: The title that the plot FILE should get (string without file extension)
    :param n_data_points: The number of data_dict points that the learner has received (integer)
    :param percentiles_composite_hyp_score: The percentiles of the composite hypothesis score, calculated over all the runs
    :param percentiles_perspective_score: The percentiles of the perspective hypothesis score, calculated over all the runs
    :param percentiles_lexicon_hyp_score: The percentiles of the lexicon hypothesis score, calculated over all the runs
    :param percentiles_lexicon_approximation: The percentiles of the lexicon APPROXIMATION score, calculated over all the runs
    :return: Creates a plot with the number of observations on the x-axis and the four different performance scores on the y-axis. Saves the plot and then opens it.
    """
    sns.set_style("darkgrid", {"xtick.major.size":4, "ytick.major.size":4})
    sns.set_palette("deep")
    color_composite_hyp = sns.xkcd_rgb["purplish"] #/  "grape purple"/ "darker purple" / "grape"
    color_perspective_hyp = sns.xkcd_rgb["macaroni and cheese"] #/ "desert" / "sand" / "dusty orange" / "orangish" / "light mustard" / "faded orange"
    color_lexicon_hyp = sns.xkcd_rgb["salmon pink"] #sns.xkcd_rgb["teal"] #/ "turquoise blue / "blue green" / "greenish blue"
    color_lexicon_approximation = sns.xkcd_rgb["salmon pink"]
    fig, ax = plt.subplots(1)
    ax.plot(np.arange(n_data_points), percentiles_composite_hyp_score[1], label='Composite hyp.', color=color_composite_hyp)
    ax.fill_between(np.arange(n_data_points), percentiles_composite_hyp_score[0], percentiles_composite_hyp_score[2], facecolor=color_composite_hyp, alpha=0.3)
    ax.plot(np.arange(n_data_points), percentiles_perspective_score[1], label='Perspective hyp.', color=color_perspective_hyp)
    ax.plot(np.arange(n_data_points), percentiles_lexicon_hyp_score[1], label='Lexicon hyp.', color=color_lexicon_hyp)
    # ax.plot(np.arange(n_data_points), percentiles_lexicon_approximation[1], label='Lexicon approx.', color=color_lexicon_approximation)
    # ax.fill_between(np.arange(n_data_points), percentiles_lexicon_approximation[0], percentiles_lexicon_approximation[2], facecolor=color_lexicon_approximation, sal_alpha=0.3)
    plt.ylim((-0.05, 1.05))
    ax.set_title(plot_title)
    ax.set_xlabel("No. observations")
    ax.set_ylabel("Posterior probability")
    ax.legend(loc='center right')
    plt.savefig(plot_file_path+'/Timecrse_Scores_Med_'+plot_file_title+'.png')
    plt.show()




def plot_timecourse_scores_percentiles_without_error_mean(plot_title, plot_file_path, plot_file_title, n_data_points, percentiles_composite_hyp_score, percentiles_perspective_score, percentiles_lexicon_hyp_score):
    """
    :param plot_title: The title that will be displayed at the top of the graph (i.e. on the plot itself)
    :param plot_file_title: The title that the plot FILE should get (string without file extension)
    :param n_data_points: The number of data_dict points that the learner has received (integer)
    :param percentiles_composite_hyp_score: The percentiles of the composite hypothesis score, calculated over all the runs
    :param percentiles_perspective_score: The percentiles of the perspective hypothesis score, calculated over all the runs
    :param percentiles_lexicon_hyp_score: The percentiles of the lexicon hypothesis score, calculated over all the runs
    :param percentiles_lexicon_approximation: The percentiles of the lexicon APPROXIMATION score, calculated over all the runs
    :return: Creates a plot with the number of observations on the x-axis and the four different performance scores on the y-axis. Saves the plot and then opens it.
    """
    sns.set_style("darkgrid", {"xtick.major.size":4, "ytick.major.size":4})
    sns.set_palette("deep")
    color_composite_hyp = sns.xkcd_rgb["purplish"] #/  "grape purple"/ "darker purple" / "grape"
    color_perspective_hyp = sns.xkcd_rgb["macaroni and cheese"] #/ "desert" / "sand" / "dusty orange" / "orangish" / "light mustard" / "faded orange"
    color_lexicon_hyp = sns.xkcd_rgb["salmon pink"] #sns.xkcd_rgb["teal"] #/ "turquoise blue / "blue green" / "greenish blue"
    color_lexicon_approximation = sns.xkcd_rgb["salmon pink"]
    fig, ax = plt.subplots(1)
    ax.plot(np.arange(n_data_points), percentiles_composite_hyp_score[1], label='Composite hyp.', color=color_composite_hyp)
    ax.fill_between(np.arange(n_data_points), percentiles_composite_hyp_score[0], percentiles_composite_hyp_score[2], facecolor=color_composite_hyp, alpha=0.3)
    ax.plot(np.arange(n_data_points), percentiles_perspective_score[3], label='Perspective hyp.', color=color_perspective_hyp)
    ax.plot(np.arange(n_data_points), percentiles_lexicon_hyp_score[3], label='Lexicon hyp.', color=color_lexicon_hyp)
    # ax.plot(np.arange(n_data_points), percentiles_lexicon_approximation[1], label='Lexicon approx.', color=color_lexicon_approximation)
    # ax.fill_between(np.arange(n_data_points), percentiles_lexicon_approximation[0], percentiles_lexicon_approximation[2], facecolor=color_lexicon_approximation, sal_alpha=0.3)
    plt.ylim((-0.05, 1.05))
    ax.set_title(plot_title)
    ax.set_xlabel("No. observations")
    ax.set_ylabel("Posterior probability")
    ax.legend(loc='center right')
    plt.savefig(plot_file_path+'/Timecrse_Scores_Mean_'+plot_file_title+'.png')
    plt.show()



def plot_timecourse_scores_percentiles_with_speaker_distinction(plot_title, plot_file_path, plot_file_title, n_data_points, percentiles_composite_hyp_score, percentiles_perspective_score_per_speaker, percentiles_lexicon_hyp_score):
    """
    :param plot_title: The title that will be displayed at the top of the graph (i.e. on the plot itself)
    :param plot_file_title: The title that the plot FILE should get (string without file extension)
    :param n_data_points: The number of data_dict points that the learner has received (integer)
    :param percentiles_composite_hyp_score: The percentiles of the composite hypothesis score, calculated over all the runs
    :param percentiles_perspective_score: The percentiles of the perspective hypothesis score, calculated over all the runs
    :param percentiles_lexicon_hyp_score: The percentiles of the lexicon hypothesis score, calculated over all the runs
    :param percentiles_lexicon_approximation: The percentiles of the lexicon APPROXIMATION score, calculated over all the runs
    :return: Creates a plot with the number of observations on the x-axis and the four different performance scores on the y-axis. Saves the plot and then opens it.
    """
    sns.set_style("darkgrid", {"xtick.major.size":4, "ytick.major.size":4})
    sns.set_palette("deep")
    color_composite_hyp = sns.xkcd_rgb["purplish"] #/  "grape purple"/ "darker purple" / "grape"
    color_perspective_hyp = sns.xkcd_rgb["macaroni and cheese"] #/ "desert" / "sand" / "dusty orange" / "orangish" / "light mustard" / "faded orange"
    color_lexicon_hyp = sns.xkcd_rgb["salmon pink"] #sns.xkcd_rgb["teal"] #/ "turquoise blue / "blue green" / "greenish blue"
    color_lexicon_approximation = sns.xkcd_rgb["salmon pink"]
    linestyles = ['--', ':', '-.', '-']
    hatches = ['-', '.'] #('-', '+', 'x', '\\', '*', 'o', 'O', '.')
    fig, ax = plt.subplots(1)
    ax.plot(np.arange(n_data_points), percentiles_composite_hyp_score[1], label='Composite hyp.', color=color_composite_hyp)
    ax.fill_between(np.arange(n_data_points), percentiles_composite_hyp_score[0], percentiles_composite_hyp_score[2], facecolor=color_composite_hyp, alpha=0.3)
    for speaker_index in range(len(percentiles_perspective_score_per_speaker)):
        ax.plot(np.arange(n_data_points), percentiles_perspective_score_per_speaker[speaker_index][1], label='Perspective hyp., speaker '+str(speaker_index), color=color_perspective_hyp, linestyle=linestyles[speaker_index])
        ax.fill_between(np.arange(n_data_points), percentiles_perspective_score_per_speaker[speaker_index][0], percentiles_perspective_score_per_speaker[speaker_index][2], facecolor=color_perspective_hyp, alpha=0.3, edgecolor='black', hatch=hatches[speaker_index])
    ax.plot(np.arange(n_data_points), percentiles_lexicon_hyp_score[1], label='Lexicon hyp.', color=color_lexicon_hyp)
    ax.fill_between(np.arange(n_data_points), percentiles_lexicon_hyp_score[0], percentiles_lexicon_hyp_score[2], facecolor=color_lexicon_hyp, alpha=0.3)
    # ax.plot(np.arange(n_data_points), percentiles_lexicon_approximation[1], label='Lexicon approx.', color=color_lexicon_approximation)
    # ax.fill_between(np.arange(n_data_points), percentiles_lexicon_approximation[0], percentiles_lexicon_approximation[2], facecolor=color_lexicon_approximation, sal_alpha=0.3)
    plt.ylim((-0.05, 1.05))
    ax.set_title(plot_title)
    ax.set_xlabel("No. observations")
    ax.set_ylabel("Posterior probability")
    ax.legend(loc='center right')
    plt.savefig(plot_file_path+'/Timecrse_Scores_Med_Perc_'+plot_file_title+'.png')
    plt.show()




def plot_timecourse_scores_percentiles_without_error_median_with_speaker_distinction(plot_title, plot_file_path, plot_file_title, n_data_points, percentiles_composite_hyp_score, percentiles_perspective_score_per_speaker, percentiles_lexicon_hyp_score):
    """
    :param plot_title: The title that will be displayed at the top of the graph (i.e. on the plot itself)
    :param plot_file_title: The title that the plot FILE should get (string without file extension)
    :param n_data_points: The number of data_dict points that the learner has received (integer)
    :param percentiles_composite_hyp_score: The percentiles of the composite hypothesis score, calculated over all the runs
    :param percentiles_perspective_score: The percentiles of the perspective hypothesis score, calculated over all the runs
    :param percentiles_lexicon_hyp_score: The percentiles of the lexicon hypothesis score, calculated over all the runs
    :param percentiles_lexicon_approximation: The percentiles of the lexicon APPROXIMATION score, calculated over all the runs
    :return: Creates a plot with the number of observations on the x-axis and the four different performance scores on the y-axis. Saves the plot and then opens it.
    """
    sns.set_style("darkgrid", {"xtick.major.size":4, "ytick.major.size":4})
    sns.set_palette("deep")
    color_composite_hyp = sns.xkcd_rgb["purplish"] #/  "grape purple"/ "darker purple" / "grape"
    color_perspective_hyp = sns.xkcd_rgb["macaroni and cheese"] #/ "desert" / "sand" / "dusty orange" / "orangish" / "light mustard" / "faded orange"
    color_lexicon_hyp = sns.xkcd_rgb["salmon pink"] #sns.xkcd_rgb["teal"] #/ "turquoise blue / "blue green" / "greenish blue"
    color_lexicon_approximation = sns.xkcd_rgb["salmon pink"]
    linestyles = ['--', ':', '-.', '-']
    fig, ax = plt.subplots(1)
    ax.plot(np.arange(n_data_points), percentiles_composite_hyp_score[1], label='Composite hyp.', color=color_composite_hyp)
    for speaker_index in range(len(percentiles_perspective_score_per_speaker)):
        ax.plot(np.arange(n_data_points), percentiles_perspective_score_per_speaker[speaker_index][1], label='Perspective hyp., speaker '+str(speaker_index), color=color_perspective_hyp, linestyle=linestyles[speaker_index])
    ax.plot(np.arange(n_data_points), percentiles_lexicon_hyp_score[1], label='Lexicon hyp.', color=color_lexicon_hyp)
    # ax.plot(np.arange(n_data_points), percentiles_lexicon_approximation[1], label='Lexicon approx.', color=color_lexicon_approximation)
    plt.ylim((-0.05, 1.05))
    ax.set_title(plot_title)
    ax.set_xlabel("No. observations")
    ax.set_ylabel("Posterior probability")
    ax.legend(loc='center right')
    plt.savefig(plot_file_path+'/Timecrse_Scores_Med_'+plot_file_title+'.png')
    plt.show()




def plot_timecourse_scores_percentiles_without_error_mean_with_speaker_distinction(plot_title, plot_file_path, plot_file_title, n_data_points, percentiles_composite_hyp_score, percentiles_perspective_score_per_speaker, percentiles_lexicon_hyp_score):
    """
    :param plot_title: The title that will be displayed at the top of the graph (i.e. on the plot itself)
    :param plot_file_title: The title that the plot FILE should get (string without file extension)
    :param n_data_points: The number of data_dict points that the learner has received (integer)
    :param percentiles_composite_hyp_score: The percentiles of the composite hypothesis score, calculated over all the runs
    :param percentiles_perspective_score: The percentiles of the perspective hypothesis score, calculated over all the runs
    :param percentiles_lexicon_hyp_score: The percentiles of the lexicon hypothesis score, calculated over all the runs
    :param percentiles_lexicon_approximation: The percentiles of the lexicon APPROXIMATION score, calculated over all the runs
    :return: Creates a plot with the number of observations on the x-axis and the four different performance scores on the y-axis. Saves the plot and then opens it.
    """
    sns.set_style("darkgrid", {"xtick.major.size":4, "ytick.major.size":4})
    sns.set_palette("deep")
    color_composite_hyp = sns.xkcd_rgb["purplish"] #/  "grape purple"/ "darker purple" / "grape"
    color_perspective_hyp = sns.xkcd_rgb["macaroni and cheese"] #/ "desert" / "sand" / "dusty orange" / "orangish" / "light mustard" / "faded orange"
    color_lexicon_hyp = sns.xkcd_rgb["salmon pink"] #sns.xkcd_rgb["teal"] #/ "turquoise blue / "blue green" / "greenish blue"
    color_lexicon_approximation = sns.xkcd_rgb["salmon pink"]
    linestyles = ['--', ':', '-.', '-']
    fig, ax = plt.subplots(1)
    ax.plot(np.arange(n_data_points), percentiles_composite_hyp_score[3], label='Composite hyp.', color=color_composite_hyp)
    for speaker_index in range(len(percentiles_perspective_score_per_speaker)):
        ax.plot(np.arange(n_data_points), percentiles_perspective_score_per_speaker[speaker_index][3], label='Perspective hyp., speaker '+str(speaker_index), color=color_perspective_hyp, linestyle=linestyles[speaker_index])
    ax.plot(np.arange(n_data_points), percentiles_lexicon_hyp_score[3], label='Lexicon hyp.', color=color_lexicon_hyp)
    # ax.plot(np.arange(n_data_points), percentiles_lexicon_approximation[3], label='Lexicon approx.', color=color_lexicon_approximation)
    plt.ylim((-0.05, 1.05))
    ax.set_title(plot_title)
    ax.set_xlabel("No. observations")
    ax.set_ylabel("Posterior probability")
    ax.legend(loc='center right')
    plt.savefig(plot_file_path+'/Timecrse_Scores_Mean_'+plot_file_title+'.png')
    plt.show()




def plot_timecourse_scores_percentiles_med_perc_one_hyp_type(plot_title, plot_file_path, plot_file_title, n_data_points, percentiles_composite_hyp_score, percentiles_perspective_hyp_score, percentiles_lexicon_hyp_score, which_hyp_type, x_axis_steps):
    """
    :param plot_title: The title that will be displayed at the top of the graph (i.e. on the plot itself)
    :param plot_file_title: The title that the plot FILE should get (string without file extension)
    :param n_data_points: The number of data_dict points that the learner has received (integer)
    :param percentiles_composite_hyp_score: The percentiles of the composite hypothesis score, calculated over all the runs
    :param percentiles_perspective_score: The percentiles of the perspective hypothesis score, calculated over all the runs
    :param percentiles_lexicon_hyp_score: The percentiles of the lexicon hypothesis score, calculated over all the runs
    :param percentiles_lexicon_approximation: The percentiles of the lexicon APPROXIMATION score, calculated over all the runs
    :return: Creates a plot with the number of observations on the x-axis and the four different performance scores on the y-axis. Saves the plot and then opens it.
    """
    sns.set_style("darkgrid", {"xtick.major.size":4, "ytick.major.size":4})
    sns.set_palette("deep")
    color = sns.xkcd_rgb["purplish"] #/  "grape purple"/ "darker purple" / "grape"
    fig, ax = plt.subplots(1)
    if which_hyp_type == 'composite':
        percentiles = percentiles_composite_hyp_score
        percentiles_label = 'Correct composite hyp.'
    elif which_hyp_type == 'perspective':
        percentiles = percentiles_perspective_hyp_score
        percentiles_label = 'Correct perspective hyp.'
    elif which_hyp_type == 'lexicon':
        percentiles = percentiles_lexicon_hyp_score
        percentiles_label = 'Correct lexicon hyp.'
    ax.plot(np.arange(n_data_points), percentiles[1], label=percentiles_label, color=color)
    ax.fill_between(np.arange(n_data_points), percentiles[0], percentiles[2], facecolor=color, alpha=0.3)
    plt.ylim((-0.05, 1.05))
    ax.set_yticks(np.arange(0.05, 1.05, 0.05))
    ax.set_xticks(np.arange(0, n_data_points, x_axis_steps))
    ax.set_title(plot_title)
    ax.set_xlabel("No. observations")
    ax.set_ylabel("Posterior probability")
    ax.legend(loc='center right')
    plt.savefig(plot_file_path+'/Timecrse_Scores_Comp_Only_'+plot_file_title+'_'+which_hyp_type+'.png')
    plt.show()





def plot_timecourse_scores_percentiles_without_error_mean(plot_title, plot_file_path, plot_file_title, n_data_points, mean_composite_hyp_score, x_axis_steps, label):
    """
    :param plot_title: The title that will be displayed at the top of the graph (i.e. on the plot itself)
    :param plot_file_title: The title that the plot FILE should get (string without file extension)
    :param n_data_points: The number of data_dict points that the learner has received (integer)
    :param percentiles_composite_hyp_score: The percentiles of the composite hypothesis score, calculated over all the runs
    :param percentiles_perspective_score: The percentiles of the perspective hypothesis score, calculated over all the runs
    :param percentiles_lexicon_hyp_score: The percentiles of the lexicon hypothesis score, calculated over all the runs
    :param percentiles_lexicon_approximation: The percentiles of the lexicon APPROXIMATION score, calculated over all the runs
    :return: Creates a plot with the number of observations on the x-axis and the four different performance scores on the y-axis. Saves the plot and then opens it.
    """
    sns.set_style("darkgrid", {"xtick.major.size":4, "ytick.major.size":4})
    sns.set_palette("deep")
    if label == 'composite':
        color = sns.xkcd_rgb["purplish"] #/  "grape purple"/ "darker purple" / "grape"
    elif label == 'perspective':
        color = sns.xkcd_rgb["macaroni and cheese"] #/ "desert" / "sand" / "dusty orange" / "orangish" / "light mustard" / "faded orange"
    elif label == 'lexicon':
        color = sns.xkcd_rgb["salmon pink"] #sns.xkcd_rgb["teal"] #/ "turquoise blue / "blue green" / "greenish blue"
    elif label == 'lex. approximation':
        color = sns.xkcd_rgb["salmon pink"]
    fig, ax = plt.subplots(1)
    ax.plot(np.arange(n_data_points), mean_composite_hyp_score, label=label+' hyp.', color=color)
    plt.ylim((-0.05, 1.05))
    ax.set_yticks(np.arange(0.05, 1.05, 0.05))
    ax.set_xticks(np.arange(0, n_data_points, x_axis_steps))
    ax.set_title(plot_title)
    ax.set_xlabel("No. observations")
    ax.set_ylabel("Posterior probability")
    ax.legend(loc='center right')
    plt.savefig(plot_file_path+'/Timecrse_Scores_Mean_Comp_Only_'+plot_file_title+'.png')
    plt.show()






def plot_cum_belief_percentiles_with_speaker_distinction(plot_title, plot_file_path, plot_file_title, n_data_points, percentiles_composite_hyp_score, percentiles_perspective_score_per_speaker, percentiles_lexicon_hyp_score):
    """
    :param plot_title: The title that will be displayed at the top of the graph (i.e. on the plot itself)
    :param plot_file_title: The title that the plot FILE should get (string without file extension)
    :param n_data_points: The number of data_dict points that the learner has received (integer)
    :param percentiles_composite_hyp_score: The percentiles of the composite hypothesis score, calculated over all the runs
    :param percentiles_perspective_score: The percentiles of the perspective hypothesis score, calculated over all the runs
    :param percentiles_lexicon_hyp_score: The percentiles of the lexicon hypothesis score, calculated over all the runs
    :param percentiles_lexicon_approximation: The percentiles of the lexicon APPROXIMATION score, calculated over all the runs
    :return: Creates a plot with the number of observations on the x-axis and the four different performance scores on the y-axis. Saves the plot and then opens it.
    """
    sns.set_style("darkgrid", {"xtick.major.size":4, "ytick.major.size":4})
    sns.set_palette("deep")
    color_composite_hyp = sns.xkcd_rgb["purplish"] #/  "grape purple"/ "darker purple" / "grape"
    color_perspective_hyp = sns.xkcd_rgb["macaroni and cheese"] #/ "desert" / "sand" / "dusty orange" / "orangish" / "light mustard" / "faded orange"
    color_lexicon_hyp = sns.xkcd_rgb["salmon pink"] #sns.xkcd_rgb["teal"] #/ "turquoise blue / "blue green" / "greenish blue"
    color_lexicon_approximation = sns.xkcd_rgb["salmon pink"]
    linestyles = ['--', ':', '-.', '-']
    hatches = ['-', '.'] #('-', '+', 'x', '\\', '*', 'o', 'O', '.')
    fig, ax = plt.subplots(1)
    ax.plot(np.arange(n_data_points), percentiles_composite_hyp_score[1], label='Composite hyp.', color=color_composite_hyp)
    ax.fill_between(np.arange(n_data_points), percentiles_composite_hyp_score[0], percentiles_composite_hyp_score[2], facecolor=color_composite_hyp, alpha=0.3)
    # for speaker_index in range(len(percentiles_perspective_score_per_speaker)):
    #     ax.plot(np.arange(n_data_points), percentiles_perspective_score_per_speaker[speaker_index][1], label='Perspective hyp., speaker '+str(speaker_index), color=color_perspective_hyp, linestyle=linestyles[speaker_index])
    #     ax.fill_between(np.arange(n_data_points), percentiles_perspective_score_per_speaker[speaker_index][0], percentiles_perspective_score_per_speaker[speaker_index][2], facecolor=color_perspective_hyp, sal_alpha=0.3, edgecolor='black', hatch=hatches[speaker_index])
    # ax.plot(np.arange(n_data_points), percentiles_lexicon_hyp_score[1], label='Lexicon hyp.', color=color_lexicon_hyp)
    # ax.fill_between(np.arange(n_data_points), percentiles_lexicon_hyp_score[0], percentiles_lexicon_hyp_score[2], facecolor=color_lexicon_hyp, sal_alpha=0.3)
    ax.set_title(plot_title)
    ax.set_xlabel("No. observations")
    ax.set_ylabel("Cumulative belief in correct composite hypothesis")
    ax.legend(loc='center right')
    plt.savefig(plot_file_path+'/Cum_Belief_Med_Perc_'+plot_file_title+'.png')
    plt.show()




def plot_cum_belief_percentiles_without_error_median_with_speaker_distinction(plot_title, plot_file_path, plot_file_title, n_data_points, percentiles_composite_hyp_score, percentiles_perspective_score_per_speaker, percentiles_lexicon_hyp_score):
    """
    :param plot_title: The title that will be displayed at the top of the graph (i.e. on the plot itself)
    :param plot_file_title: The title that the plot FILE should get (string without file extension)
    :param n_data_points: The number of data_dict points that the learner has received (integer)
    :param percentiles_composite_hyp_score: The percentiles of the composite hypothesis score, calculated over all the runs
    :param percentiles_perspective_score: The percentiles of the perspective hypothesis score, calculated over all the runs
    :param percentiles_lexicon_hyp_score: The percentiles of the lexicon hypothesis score, calculated over all the runs
    :param percentiles_lexicon_approximation: The percentiles of the lexicon APPROXIMATION score, calculated over all the runs
    :return: Creates a plot with the number of observations on the x-axis and the four different performance scores on the y-axis. Saves the plot and then opens it.
    """
    sns.set_style("darkgrid", {"xtick.major.size":4, "ytick.major.size":4})
    sns.set_palette("deep")
    color_composite_hyp = sns.xkcd_rgb["purplish"] #/  "grape purple"/ "darker purple" / "grape"
    color_perspective_hyp = sns.xkcd_rgb["macaroni and cheese"] #/ "desert" / "sand" / "dusty orange" / "orangish" / "light mustard" / "faded orange"
    color_lexicon_hyp = sns.xkcd_rgb["salmon pink"] #sns.xkcd_rgb["teal"] #/ "turquoise blue / "blue green" / "greenish blue"
    color_lexicon_approximation = sns.xkcd_rgb["salmon pink"]
    linestyles = ['--', ':', '-.', '-']
    fig, ax = plt.subplots(1)
    ax.plot(np.arange(n_data_points), percentiles_composite_hyp_score[1], label='Composite hyp.', color=color_composite_hyp)
    # for speaker_index in range(len(percentiles_perspective_score_per_speaker)):
    #     ax.plot(np.arange(n_data_points), percentiles_perspective_score_per_speaker[speaker_index][1], label='Perspective hyp., speaker '+str(speaker_index), color=color_perspective_hyp, linestyle=linestyles[speaker_index])
    # ax.plot(np.arange(n_data_points), percentiles_lexicon_hyp_score[1], label='Lexicon hyp.', color=color_lexicon_hyp)
    ax.set_title(plot_title)
    ax.set_xlabel("No. observations")
    ax.set_ylabel("Cumulative belief in correct composite hypothesis")
    ax.legend(loc='center right')
    plt.savefig(plot_file_path+'/Cum_Belief_Med_'+plot_file_title+'.png')
    plt.show()




def plot_cum_belief_percentiles_without_error_mean_with_speaker_distinction(plot_title, plot_file_path, plot_file_title, n_data_points, percentiles_composite_hyp_score, percentiles_perspective_score_per_speaker, percentiles_lexicon_hyp_score):
    """
    :param plot_title: The title that will be displayed at the top of the graph (i.e. on the plot itself)
    :param plot_file_title: The title that the plot FILE should get (string without file extension)
    :param n_data_points: The number of data_dict points that the learner has received (integer)
    :param percentiles_composite_hyp_score: The percentiles of the composite hypothesis score, calculated over all the runs
    :param percentiles_perspective_score: The percentiles of the perspective hypothesis score, calculated over all the runs
    :param percentiles_lexicon_hyp_score: The percentiles of the lexicon hypothesis score, calculated over all the runs
    :param percentiles_lexicon_approximation: The percentiles of the lexicon APPROXIMATION score, calculated over all the runs
    :return: Creates a plot with the number of observations on the x-axis and the four different performance scores on the y-axis. Saves the plot and then opens it.
    """
    sns.set_style("darkgrid", {"xtick.major.size":4, "ytick.major.size":4})
    sns.set_palette("deep")
    color_composite_hyp = sns.xkcd_rgb["purplish"] #/  "grape purple"/ "darker purple" / "grape"
    color_perspective_hyp = sns.xkcd_rgb["macaroni and cheese"] #/ "desert" / "sand" / "dusty orange" / "orangish" / "light mustard" / "faded orange"
    color_lexicon_hyp = sns.xkcd_rgb["salmon pink"] #sns.xkcd_rgb["teal"] #/ "turquoise blue / "blue green" / "greenish blue"
    color_lexicon_approximation = sns.xkcd_rgb["salmon pink"]
    linestyles = ['--', ':', '-.', '-']
    fig, ax = plt.subplots(1)
    ax.plot(np.arange(n_data_points), percentiles_composite_hyp_score[3], label='Composite hyp.', color=color_composite_hyp)
    # for speaker_index in range(len(percentiles_perspective_score_per_speaker)):
    #     ax.plot(np.arange(n_data_points), percentiles_perspective_score_per_speaker[speaker_index][3], label='Perspective hyp., speaker '+str(speaker_index), color=color_perspective_hyp, linestyle=linestyles[speaker_index])
    # ax.plot(np.arange(n_data_points), percentiles_lexicon_hyp_score[3], label='Lexicon hyp.', color=color_lexicon_hyp)
    ax.set_title(plot_title)
    ax.set_xlabel("No. observations")
    ax.set_ylabel("Cumulative belief in correct composite hypothesis")
    ax.legend(loc='center right')
    plt.savefig(plot_file_path+'/Cum_Belief_Mean_'+plot_file_title+'.png')
    plt.show()





def boxplot_input_order(plot_file_title, percentiles_three_conditions_dict):
    # sns.set_style("whitegrid", {"xtick.major.size":8, "ytick.major.size":8})
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    # sns.set_context("poster", font_scale=1.6)

    fig, axes = plt.subplots(ncols=3, sharey=True)
    # fig.subplots_adjust(wspace=0)

    blue = sns.color_palette()[0]
    green = sns.color_palette()[2]

    counter = 0
    for ax, name in zip(axes, ['Random', 'Same First', 'Opposite First']):
        bp = ax.boxplot([percentiles_three_conditions_dict[name][item] for item in ['same perspective', 'opposite perspective']], widths = 0.5, patch_artist=True)
        ## change outline color, fill color and linewidth of the boxes
        for i in range(len(bp['boxes'])):
            if i == 0 or i == 2 or i == 4:
                # change outline color
                bp['boxes'][i].set(color=green, alpha=0.3, linewidth=2)
                # change fill color
                bp['boxes'][i].set(facecolor=green, alpha=0.3)
                ## change color and linewidth of the medians
                bp['medians'][i].set(color=green, linewidth=2)
                # ## change the style of fliers and their fill
                # bp['fliers'][i].set(marker='o', color=odd_boxes_colour, sal_alpha=0.5)
            elif i==1 or i==3 or i==5:
                # change outline color
                bp['boxes'][i].set(color=blue, alpha=0.3, linewidth=2)
                # change fill color
                bp['boxes'][i].set(facecolor=blue, alpha=0.3)
                ## change color and linewidth of the medians))
                bp['medians'][i].set(color=blue, linewidth=2)
                # ## change the style of fliers and their fill
                # bp['fliers'][i].set(marker='o', color=even_boxes_colour, sal_alpha=0.5)
        for j in range(len(bp['whiskers'])):
            if j == 0 or j == 1 or j == 4 or j == 5 or j == 8 or j == 9:
                ## change color and linewidth of the whiskers
                bp['whiskers'][j].set(color=green, linewidth=2)
                ## change color and linewidth of the caps
                bp['caps'][j].set(color=green, linewidth=2)
            elif j == 2 or j == 3 or j == 6 or j == 7 or j == 10 or j == 11:
                ## change color and linewidth of the whiskers
                bp['whiskers'][j].set(color=blue, linewidth=2)
                ## change color and linewidth of the caps
                bp['caps'][j].set(color=blue, linewidth=2)
        # Remove the tick-marks from top and right spines
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        # Set xlabels:
        ax.set(xticklabels=['same p.', 'opposite p.'], xlabel=name)
        ax.xaxis.label.set_size(14)
        ax.tick_params(labelsize=11)
        # ax.margins(0.1) # Optional
        if counter == 0:
            # ax.set_title('Learning different perspectives with different orders of input', x=1.5, y=1.05)
            ax.set_ylabel('No. observations required to learn perspective')
            ax.yaxis.label.set_size(14)
        counter += 1
    # fig.subplots_adjust(wspace=0.05)
    # plt.gcf().subplots_adjust(bottom=0.15)
    # Save the figure
    plt.suptitle('Learning different perspectives with different orders of input', fontsize=16)
    plt.savefig(plot_file_title)
    plt.show()







def boxplot_input_order_opposite_only(plot_file_title, percentiles_three_conditions_dict):
    # sns.set_style("whitegrid", {"xtick.major.size":8, "ytick.major.size":8})
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    # sns.set_context("poster", font_scale=1.6)

    fig, axes = plt.subplots(ncols=3, sharey=True)
    # fig.subplots_adjust(wspace=0)

    blue = sns.color_palette()[0]

    counter = 0
    for ax, name in zip(axes, ['Randomly Interleaved', 'Same First', 'Opposite First']):
        bp = ax.boxplot([percentiles_three_conditions_dict[name][item] for item in ['opposite perspective']], widths = 0.5, patch_artist=True)
        ## change outline color, fill color and linewidth of the boxes
        for i in range(len(bp['boxes'])):
            # change outline color
            bp['boxes'][i].set(color=blue, alpha=0.3, linewidth=2)
            # change fill color
            bp['boxes'][i].set(facecolor=blue, alpha=0.3)
            ## change color and linewidth of the medians))
            bp['medians'][i].set(color=blue, linewidth=2)
            # ## change the style of fliers and their fill
            # bp['fliers'][i].set(marker='o', color=even_boxes_colour, sal_alpha=0.5)
        for j in range(len(bp['whiskers'])):
            ## change color and linewidth of the whiskers
            bp['whiskers'][j].set(color=blue, linewidth=2)
            ## change color and linewidth of the caps
            bp['caps'][j].set(color=blue, linewidth=2)
        # Remove the tick-marks from top and right spines
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        # Set xlabels:
        ax.set(xticklabels=['opposite perspective'], xlabel=name)
        ax.xaxis.label.set_size(14)
        ax.tick_params(labelsize=11)
        # ax.margins(0.1) # Optional
        if counter == 0:
            # ax.set_title('Learning opposite perspective with different orders of input', x=1.5, y=1.05)
            ax.set_ylabel('No. observations required to learn perspective')
            ax.yaxis.label.set_size(14)
        counter += 1
    # fig.subplots_adjust(wspace=0.05)
    # plt.gcf().subplots_adjust(bottom=0.15)
    # Save the figure
    plt.suptitle('Learning opposite perspective with different orders of input', fontsize=16)
    plt.savefig(plot_file_title)
    plt.show()









def plot_timecourse_scores_percentiles_for_different_learners(plot_title, plot_file_path, plot_file_title, n_data_points, percentiles_of_learners_matrix, learner_labels_list):
    """
    :param plot_title: The title that will be displayed at the top of the graph (i.e. on the plot itself)
    :param plot_file_title: The title that the plot FILE should get (string without file extension)
    :param n_data_points: The number of data_dict points that the learner has received (integer)
    :param percentiles_of_learners_matrix: Matrix containing the percentiles for all the different learners that we want to plot
    :param learner_labels_list: A list of string labels for the respective learners
    :return: A plot with the scores percentiles of the different learners
    """
    # sns.set_style("white", {"xtick.major.size":8, "ytick.major.size":8})
    sns.set_style("ticks")
    sns.despine()
    sns.set_palette("deep")
    # sns.set_context("poster", font_scale=1.6)
    # sns.set_context("poster", font_scale=1.6, rc={"lines.linewidth": 3.5})
    fig, ax = plt.subplots(1)
    palette = itertools.cycle(sns.color_palette()[:])
    for i in range(len(percentiles_of_learners_matrix)):
        learner_percentiles = percentiles_of_learners_matrix[i]
        learner_label = learner_labels_list[i]
        if learner_label == 'ToM + Unambiguous Lexicon' or learner_label == 'Typical' or learner_label == 'ToM + ca=0.9':
            learner_color = sns.color_palette()[1]
        elif learner_label == 'No ToM + Unambiguous Lexicon' or learner_label == 'No ToM' or learner_label == 'No ToM + ca=0.9':
            learner_color = sns.color_palette()[2]
        elif learner_label == 'ToM + Ambiguous Lexicon' or learner_label == 'ToM + Partly Ambiguous Lexicon' or learner_label == 'Partly Ambiguous Lexicon' or learner_label == 'ToM + ca=0.51':
            learner_color = sns.color_palette()[0]
        elif learner_label == 'ToM + ca=0.45':
            learner_color = sns.color_palette()[0]
        elif learner_label == 'ToM + ca=0.38':
            learner_color = sns.color_palette()[4]
        elif learner_label == 'ToM + ca=0.36':
            learner_color = sns.color_palette()[5]
        elif learner_label == 'ToM + Uninformative Lexicon' or learner_label == 'Uninformative Lexicon' or learner_label == 'ToM + ca=0.33':
            learner_color = sns.color_palette()[3]
        else:
            learner_color = next(palette)
        ax.plot(np.arange(n_data_points), learner_percentiles[1], label=learner_label, color=learner_color)
        ax.fill_between(np.arange(n_data_points), learner_percentiles[0], learner_percentiles[2], facecolor=learner_color, alpha=0.3)
    plt.ylim((-0.05, 1.05))
    sns.despine()
    # ax.set_title(plot_title)
    # ax.set_xlabel("No. of contexts observed, with "+str(params.n_utterances)+" utterance per context")
    ax.set_xlabel("No. observations", fontsize=14)
    ax.set_ylabel("Posterior prob. on correct P+L hypothesis", fontsize=14)
    # ax.legend(loc='center right')
    plt.tight_layout()
    plt.savefig(plot_file_path+'/Diff_Learners_'+plot_file_title+'.pdf')
    plt.show()




def plot_timecourse_scores_percentiles_for_different_learners_cogsci_paper(plot_title, plot_file_path, plot_file_title, n_data_points, percentiles_of_learners_matrix, learner_labels_list):
    """
    :param plot_title: The title that will be displayed at the top of the graph (i.e. on the plot itself)
    :param plot_file_title: The title that the plot FILE should get (string without file extension)
    :param n_data_points: The number of data_dict points that the learner has received (integer)
    :param percentiles_of_learners_matrix: Matrix containing the percentiles for all the different learners that we want to plot
    :param learner_labels_list: A list of string labels for the respective learners
    :return: A plot with the scores percentiles of the different learners
    """
    sns.set_style("whitegrid", {"xtick.major.size":8, "ytick.major.size":8})
    # sns.set_palette("dark")
    sns.set_palette("deep")
    # sns.set_palette("colorblind")
    sns.set_context("poster", font_scale=1.6)
    # sns.set_context("poster", font_scale=1.6, rc={"lines.linewidth": 3.5})
    fig, ax = plt.subplots(1)
    palette = itertools.cycle(sns.color_palette()[3:])
    for i in range(len(percentiles_of_learners_matrix)):
        learner_percentiles = percentiles_of_learners_matrix[i]
        learner_label = learner_labels_list[i]
        if learner_label == 'ToM + Unambiguous Lexicon' or learner_label == 'Typical':
            learner_color = sns.color_palette()[1]
        elif learner_label == 'No ToM + Unambiguous Lexicon' or learner_label == 'No ToM':
            learner_color = sns.color_palette()[2]
        elif learner_label == 'ToM + Ambiguous Lexicon' or learner_label == 'ToM + Partly Ambiguous Lexicon' or learner_label == 'Partly Ambiguous Lexicon':
            learner_color = sns.color_palette()[4]
        elif learner_label == 'ToM + Uninformative Lexicon' or learner_label == 'Uninformative Lexicon':
            learner_color = sns.color_palette()[3]
        else:
            learner_color = next(palette)
        ax.plot(np.arange(n_data_points), learner_percentiles[1], label=learner_label, color=learner_color)
        ax.fill_between(np.arange(n_data_points), learner_percentiles[0], learner_percentiles[2], facecolor=learner_color, alpha=0.3)
    plt.ylim((-0.05, 1.05))
    ax.set_title(plot_title)
    # ax.set_xlabel("No. of contexts observed, with "+str(params.n_utterances)+" utterance per context")
    ax.set_xlabel("No. observations")
    ax.set_ylabel("Posterior prob. on correct P+L hypothesis")
    ax.legend(loc='right')
    plt.tight_layout()
    plt.savefig(plot_file_path+'/Diff_Learners_'+plot_file_title+'.png')
    plt.show()




def plot_timecourse_scores_percentiles_for_different_learners_without_error_median(plot_title, plot_file_path, plot_file_title, n_data_points, percentiles_of_learners_matrix, learner_labels_list):
    """
    :param plot_title: The title that will be displayed at the top of the graph (i.e. on the plot itself)
    :param plot_file_title: The title that the plot FILE should get (string without file extension)
    :param n_data_points: The number of data_dict points that the learner has received (integer)
    :param percentiles_of_learners_matrix: Matrix containing the percentiles for all the different learners that we want to plot
    :param learner_labels_list: A list of string labels for the respective learners
    :return: A plot with the scores percentiles of the different learners
    """
    sns.set_style("darkgrid", {"xtick.major.size":4, "ytick.major.size":4})
    sns.set_palette("deep")
    fig, ax = plt.subplots(1)
    palette = itertools.cycle(sns.color_palette())
    for i in range(len(percentiles_of_learners_matrix)):
        learner_percentiles = percentiles_of_learners_matrix[i]
        learner_label = learner_labels_list[i]
        learner_color = next(palette)
        ax.plot(np.arange(n_data_points), learner_percentiles[1], label=learner_label, color=learner_color)
    plt.ylim((-0.05, 1.05))
    ax.set_title(plot_title)
    ax.set_xlabel("No. of contexts observed, with "+str(params.n_utterances)+" utterance per context")
    ax.set_ylabel("Posterior probability on correct hypothesis")
    ax.legend(loc='center right')
    plt.savefig(plot_file_path+'/Diff_Learners_Med_'+plot_file_title+'.png')
    plt.show()




def plot_timecourse_scores_percentiles_for_different_learners_without_error_mean(plot_title, plot_file_path, plot_file_title, n_data_points, percentiles_of_learners_matrix, learner_labels_list):
    """
    :param plot_title: The title that will be displayed at the top of the graph (i.e. on the plot itself)
    :param plot_file_title: The title that the plot FILE should get (string without file extension)
    :param n_data_points: The number of data_dict points that the learner has received (integer)
    :param percentiles_of_learners_matrix: Matrix containing the percentiles for all the different learners that we want to plot
    :param learner_labels_list: A list of string labels for the respective learners
    :return: A plot with the scores percentiles of the different learners
    """
    sns.set_style("darkgrid", {"xtick.major.size":4, "ytick.major.size":4})
    sns.set_palette("deep")
    fig, ax = plt.subplots(1)
    palette = itertools.cycle(sns.color_palette())
    for i in range(len(percentiles_of_learners_matrix)):
        learner_percentiles = percentiles_of_learners_matrix[i]
        learner_label = learner_labels_list[i]
        learner_color = next(palette)
        ax.plot(np.arange(n_data_points), learner_percentiles[3], label=learner_label, color=learner_color)
        # ax.fill_between(np.arange(n_data_points), learner_percentiles[0], learner_percentiles[2], facecolor=learner_color, sal_alpha=0.3)
    plt.ylim((-0.05, 1.05))
    ax.set_title(plot_title)
    ax.set_xlabel("No. of contexts observed, with "+str(params.n_utterances)+" utterance per context")
    ax.set_ylabel("Posterior probability on correct hypothesis")
    ax.legend(loc='center right')
    plt.savefig(plot_file_path+'/Diff_Learners_Mean_'+plot_file_title+'.png')
    plt.show()




def plot_timecourse_learning_diff_lex_types(plot_file_path, plot_file_title, plot_title, ylabel, inf_level_per_lex_type_sorted, posterior_mass_correct_matrix_sorted, high_cut_off, x_axis_start, x_axis_step, lex_measure, baseline=None, maximum=None, baseline_label=None, maximum_label=None,legend=None):
    sns.set_style("whitegrid")
    # palette = itertools.cycle(sns.cubehelix_palette(n_colors=len(inf_level_per_lex_type_sorted), reverse=True))
    # palette = itertools.cycle(sns.color_palette("RdYlGn", len(inf_level_per_lex_type_sorted)))
    palette = sns.color_palette("Spectral", len(inf_level_per_lex_type_sorted))
    palette.reverse()
    palette = itertools.cycle(palette)
    ## Flip the arrays for plotting and labelling so that legend shows with lowest ca at bottom and highest ca at top:
    inf_level_per_lex_type_sorted = inf_level_per_lex_type_sorted[::-1]
    posterior_mass_correct_matrix_sorted = posterior_mass_correct_matrix_sorted[::-1]
    with sns.axes_style("whitegrid"):
        if legend == True:
            fig, ax = plt.subplots(figsize=(10.5, 6))
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
        if maximum:
            ax.axhline(maximum, color='0.2', linestyle='--', linewidth=2, label=maximum_label)
        for l in range(len(inf_level_per_lex_type_sorted)):
            inf_value = inf_level_per_lex_type_sorted[l]
            # if l < inf_legend_cutoff or l > (len(mean_composite_hyp_posterior_mass_correct_per_lex_type) - inf_legend_cutoff)-1:
            #     ax.plot(mean_composite_hyp_posterior_mass_correct_per_lex_type[l], label='ca = '+str(inf_value), color=next(palette))
            # elif l == inf_legend_cutoff:
            #     ax.plot(mean_composite_hyp_posterior_mass_correct_per_lex_type[l], label='etc.', color=next(palette))
            # else:
            #     ax.plot(mean_composite_hyp_posterior_mass_correct_per_lex_type[l], color=next(palette))
            # ax.plot(posterior_mass_correct_matrix_sorted[l][:high_cut_off+1], label='L-type '+lex_measure+' = '+str(abs(inf_value)), color=next(palette))
            ax.plot(posterior_mass_correct_matrix_sorted[l], label='L-type '+lex_measure+' = '+str(abs(inf_value)), color=next(palette))
        if baseline:
            ax.axhline(baseline, color='0.6', linestyle='--', linewidth=2, label=baseline_label)
        ax.set_xlim(0, high_cut_off)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(np.arange(x_axis_start, (high_cut_off+1), x_axis_step))
        ax.set_yticks(np.arange(0.0, 1.05, 0.1))
        ax.tick_params(labelright=True, labelsize=14)
        if legend == True:
            # Shrink current axis by 28%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.77, box.height])
            # Put a legend to the right of the current axis
            legend = ax.legend(loc='center left', bbox_to_anchor=(1.07, 0.5), frameon=True, fontsize=14)
            legend.get_frame().set_linewidth(1.5)
    plt.xlabel('No. of observations (context + utterance)', fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.suptitle(plot_title, fontsize=18)
    plt.savefig(plot_file_path+'Plot_Tmcrse_Learning_Diff_Lex_Types_'+plot_file_title+'.pdf')
    plt.show()




def plot_timecourse_learning_diff_lex_types_avg_two_measures(plot_file_path, plot_file_title, plot_title, ylabel, posterior_mass_correct_matrix_sorted_msr_1, posterior_mass_correct_matrix_sorted_msr_2, high_cut_off, x_axis_start, x_axis_step, text_size, baseline, maximum):
    sns.set_style("whitegrid")
    sns.set(font_scale=text_size)
    first_quartile_msr_1 = np.percentile(posterior_mass_correct_matrix_sorted_msr_1, 25, axis=0)
    median_msr_1 = np.percentile(posterior_mass_correct_matrix_sorted_msr_1, 50, axis=0)
    third_quartile_msr_1 = np.percentile(posterior_mass_correct_matrix_sorted_msr_1, 75, axis=0)
    first_quartile_msr_2 = np.percentile(posterior_mass_correct_matrix_sorted_msr_2, 25, axis=0)
    median_msr_2 = np.percentile(posterior_mass_correct_matrix_sorted_msr_2, 50, axis=0)
    third_quartile_msr_2 = np.percentile(posterior_mass_correct_matrix_sorted_msr_2, 75, axis=0)
    color_post_prob = sns.color_palette()[1]
    color_inf = sns.color_palette()[3]
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots()
        ax.axhline(baseline, color='0.6', linestyle='--')
        ax.axhline(maximum, color='0.2')
        ax.plot(np.arange(high_cut_off), median_msr_1[0:high_cut_off], color=color_post_prob, label='post. prob. on correct hyp.')
        ax.fill_between(np.arange(high_cut_off), first_quartile_msr_1[0:high_cut_off], third_quartile_msr_1[0:high_cut_off], color=color_post_prob, alpha=0.5)
        ax.plot(np.arange(high_cut_off), median_msr_2[0:high_cut_off], color=color_inf, label='inferred informativeness')
        ax.fill_between(np.arange(high_cut_off), first_quartile_msr_2[0:high_cut_off], third_quartile_msr_2[0:high_cut_off], color=color_inf, alpha=0.5)
        ax.set_xlim(0, high_cut_off)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(np.arange(x_axis_start, (high_cut_off+1), x_axis_steps))
        ax.set_yticks(np.arange(0.0, 1.05, 0.1))
        ax.tick_params(labelright=True)
        # Shrink current axis by 28%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.84, box.height])
        # Put a legend to the right of the current axis
        legend = ax.legend(loc='center left', bbox_to_anchor=(1.07, 0.5), frameon=True, fontsize=12)
        legend.get_frame().set_linewidth(1.5)
    plt.xlabel('No. of observations')
    plt.ylabel(ylabel)
    plt.title(plot_title)
    plt.savefig(plot_file_path+'Plot_Tmcrse_Learning_Diff_Lex_Types_Average_'+plot_file_title+'.pdf')
    plt.show()



def plot_convergence_time_over_theta_range(plot_title, plot_file_path, plot_file_title, theta_range, percentiles_convergence_time_over_theta_composite, percentiles_convergence_time_over_theta_perspective, percentiles_convergence_time_over_theta_lexicon):
    sns.set_style("darkgrid", {"xtick.major.size":4, "ytick.major.size":4})
    sns.set_palette("deep")
    color_composite_hyp = sns.xkcd_rgb["purplish"] #/  "grape purple"/ "darker purple" / "grape"
    color_perspective_hyp = sns.xkcd_rgb["macaroni and cheese"] #/ "desert" / "sand" / "dusty orange" / "orangish" / "light mustard" / "faded orange"
    color_lexicon_hyp = sns.xkcd_rgb["teal"] #/ "turquoise blue / "blue green" / "greenish blue"
    fig, ax = plt.subplots(1)
    ax.plot(theta_range, percentiles_convergence_time_over_theta_composite[1], label='Composite hyp.',color=color_composite_hyp)
    ax.fill_between(theta_range, percentiles_convergence_time_over_theta_composite[0], percentiles_convergence_time_over_theta_composite[2], label='Composite hyp.', facecolor=color_composite_hyp, alpha=0.3)
    ax.plot(theta_range, percentiles_convergence_time_over_theta_perspective[1], label='Perspective hyp.', color=color_perspective_hyp)
    ax.fill_between(theta_range, percentiles_convergence_time_over_theta_perspective[0], percentiles_convergence_time_over_theta_perspective[2], label='Perspective hyp.', facecolor=color_perspective_hyp, alpha=0.3)
    ax.plot(theta_range, percentiles_convergence_time_over_theta_lexicon[1], label='Lexicon hyp.', color=color_lexicon_hyp)
    ax.fill_between(theta_range, percentiles_convergence_time_over_theta_lexicon[0], percentiles_convergence_time_over_theta_lexicon[2], label='Lexicon hyp.', facecolor=color_lexicon_hyp, alpha=0.3)
    ax.set_title(plot_title)
    ax.set_xlabel("theta")
    ax.set_ylabel("No. of observations required (with "+str(params.n_utterances)+" utterance/context)")
    ax.legend(loc='center right')
    plt.savefig(plot_file_path+'/Convrgnc_theta_'+ params.theta_step_string+'_'+plot_file_title+'.png')
    plt.show()



def plot_timecourse_hypotheses_percentiles(plot_title, plot_file_path, plot_file_title, n_data_points, hypotheses_percentiles, correct_hypothesis_index, mirror_hypothesis_index):
    """
    :param plot_title: The title that will be displayed at the top of the graph (i.e. on the plot itself)
    :param plot_file_title: The title that the plot FILE should get (string without file extension)
    :param n_data_points: The number of data_dict points that the learner has received (integer)
    :param hypotheses_percentiles: The percentiles of the posterior probability distribution over the different hypotheses, calculated over runs
    :param correct_hypothesis_index: The index of the correct (or majority) composite hypothesis
    :param mirror_hypothesis_index: The index of the mirror image composite hypothesis (if there exists one)
    :return: Creates a plot with the number of observations on the x-axis and the posterior probability assigned to the different composite hypotheses on the y-axis. Saves the plot and then opens it.
    """
    sns.set_style("darkgrid", {"xtick.major.size":4, "ytick.major.size":4})
    sns.set_palette("deep")
    fig, ax = plt.subplots(1)
    color_correct = sns.xkcd_rgb["greenish"]
    color_mirror = sns.xkcd_rgb["reddish"]
    color_other = sns.xkcd_rgb["flat blue"]
    ax.plot(np.arange(n_data_points), hypotheses_percentiles[1][correct_hypothesis_index], label='oppos. + correct lex.', color=color_correct)
    ax.fill_between(np.arange(n_data_points), hypotheses_percentiles[0][correct_hypothesis_index], hypotheses_percentiles[2][correct_hypothesis_index], facecolor=color_correct, alpha=0.5)
    if mirror_hypothesis_index != 'nan':
        ax.plot(np.arange(n_data_points), hypotheses_percentiles[1][mirror_hypothesis_index], label='same + mirror lex.', color=color_mirror)
        ax.fill_between(np.arange(n_data_points), hypotheses_percentiles[0][mirror_hypothesis_index], hypotheses_percentiles[2][mirror_hypothesis_index], facecolor=color_mirror, alpha=0.5)
    for i in range(len(hypotheses_percentiles[0])):
        if i != correct_hypothesis_index and i != mirror_hypothesis_index:
            ax.plot(np.arange(n_data_points), hypotheses_percentiles[1][i], label='all other hypotheses' if i==0 else '', color=color_other)
            ax.fill_between(np.arange(n_data_points), hypotheses_percentiles[0][i], hypotheses_percentiles[2][i], facecolor=color_other, alpha=0.5)
    plt.ylim((-0.05, 1.05))
    ax.set_yticks(np.arange(0.05, 1.05, 0.05))
    ax.set_xticks(np.arange(0, n_data_points, 50))
    ax.set_title(plot_title)
    ax.set_xlabel("No. of contexts observed, with "+str(params.n_utterances)+" utterance per context")
    ax.set_ylabel("Posterior probability")
    ax.legend(loc='center right')
    plt.savefig(plot_file_path+'/Timecrse_Hyps_'+plot_file_title+'.png')
    plt.show()




def plot_lexicon_heatmap(plot_title, plot_file_path, plot_file_title, x_tick_labels, y_tick_labels, lexicon_matrix, show_values, decimals):
    """
    :param plot_title: The title that will be displayed at the top of the graph (i.e. on the plot itself)
    :param plot_file_title: The title that the plot FILE should get (string without file extension)
    :param lexicon_matrix: A 2D numpy array that contains for each ms-mapping the posterior probability that is assigned to that mapping at the END of each run, averaged over runs
    :return:
    """
    sns.set_style("darkgrid", {"xtick.major.size":4, "ytick.major.size":4})
    sns.set_palette("deep")
    fig, ax = plt.subplots()
    # c = plt.pcolor(lexicon_matrix, cmap=plt.cm.Blues, edgecolors='k', linewidths=4, vmin=0, vmax=1.0)
    c = plt.pcolor(lexicon_matrix, cmap = plt.cm.GnBu, edgecolors='k', linewidths=4, vmin=0, vmax=1.0)
    # plt.gca().invert_yaxis()
    ax.set_xticks(np.arange(lexicon_matrix.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(lexicon_matrix.shape[0]) + 0.5, minor=False)
    ax.yaxis.set_ticks_position('left')
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.set_xticklabels(x_tick_labels, fontsize=30)
    ax.set_yticklabels(y_tick_labels, fontsize=30)
    if show_values == True:
        fmt_string = "%."+str(decimals)+"f"
        def plot_values(pc, fmt=fmt_string, **kw):
            from itertools import izip
            pc.update_scalarmappable()
            ax = pc.get_axes()
            for p, color, value in izip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
                x, y = p.vertices[:-2, :].mean(0)
                if np.all(color[:3] > 0.5):
                    color = (0.0, 0.0, 0.0)
                else:
                    color = (1.0, 1.0, 1.0)
                ax.text(x, y, fmt % value, ha="center", va="center", fontsize=30, color=color, **kw)
        plot_values(c)
    plt.title(plot_title, y=1.08)
    plt.savefig(plot_file_path+'/Lex_Heatmap_'+plot_file_title+'.png')
    plt.show()








def plot_winning_hyps_count(plot_file_path, speaker_perspective, population_lexicons, proportional_winning_hyps_count):
    sns.set_style("darkgrid")
    sns.set_palette("deep")

    lexicon_hyps_times_p_hyps = params.lexicon_hyps
    for i in range((len(params.perspective_hyps)-1)):
        lexicon_hyps_times_p_hyps = np.concatenate((lexicon_hyps_times_p_hyps, params.lexicon_hyps))

    lexicons_and_proportions_dictionary = {}
    for i in range(len(params.lexicon_type_probs)):
        lexicon_type = params.lexicon_types[i]
        lexicon_prob = params.lexicon_type_probs[i]
        lexicons_and_proportions_dictionary.update({lexicon_type : lexicon_prob})
    print "lexicons_and_proportions_dictionary is:"
    print lexicons_and_proportions_dictionary

    print "proportional_winning_hyps_count is:"
    print proportional_winning_hyps_count
    tot_prop_converged = np.sum(proportional_winning_hyps_count)
    print "tot_prop_converged is:"
    print tot_prop_converged

    index = len(params.lexicon_hyps)*len(params.perspective_hyps)
    bar_width = 0.35

    plt.bar(np.arange(index), proportional_winning_hyps_count)
    plt.ylim(0., 1.)
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.xlabel('Hypotheses')
    plt.ylabel('Proportion of runs in which hypothesis won (out of '+str(params.n_runs)+')')
    plt.title('Winning hypotheses for speaker '+str(speaker_perspective)+' in '+str(params.speaker_order_type)+' speaker order condition with '+str(params.perspective_prior_type)+' prior')
    plt.xticks(np.arange(index) + bar_width, ('p0+l1', 'p0+l2', 'p0+l3', 'p0+l4', 'p0+l5', 'p0+l6', 'p0+l7', 'p0+l8', 'p0+l9', 'p1+l1', 'p1+l2', 'p1+l3', 'p1+l4', 'p1+l5', 'p1+l6', 'p1+l7', 'p1+l8', 'p1+l9'), rotation=45)


    for i in range(len(proportional_winning_hyps_count)):
        if proportional_winning_hyps_count[i] > 0:
        # if proportional_winning_hyps_count[i] < 1.5: # This is a dummy statement just to make sure that all lexicons are depicted on the graph
            # this is an inset axes over the main axes
            plt.axes([(0.0801 + (0.043*(i+1))), 0.5, (bar_width/10.), (bar_width/10.)])
            plt.pcolor(lexicon_hyps_times_p_hyps[i], cmap=plt.cm.Blues)
            plt.gca().invert_yaxis()
            plt.xticks([])
            plt.yticks([])
            # plt.xticks(np.arange(n_signals), ['s1', 's2'])
            # plt.gca().xaxis.tick_top()
            # plt.yticks(np.arange(n_meanings), ['m1', 'm2'])

    counter = 0
    for lex_type in lexicons_and_proportions_dictionary:
        lexicon_prob = lexicons_and_proportions_dictionary.get(lex_type)
        if lexicon_prob > 0.:
            for i in range(len(params.lexicon_types)):
                if params.lexicon_types[i] == lex_type:
                    lexicon = population_lexicons[i].lexicon
            plt.axes([(0.12+(0.08*counter)), 0.75, (bar_width/5.), (bar_width/5.)])
            plt.pcolor(lexicon, cmap=plt.cm.Blues)
            plt.gca().invert_yaxis()
            plt.xticks([])
            plt.yticks([])
            plt.title("Correct Lexicon:", bbox={'sal_alpha':0.5, 'pad':10})
            plt.xlabel("prop. speakers = "+str(lexicon_prob), bbox={'sal_alpha':0.5, 'pad':10})
        counter += 1


    plt.text((0.55*index), 0.7, 'tot. prop. converged = '+str(tot_prop_converged),
    bbox={'sal_alpha':0.5, 'pad':10})

    plt.gcf().subplots_adjust(bottom=0.15) # This makes room for the xlabel which would otherwise be cut off because of the rotation of the xticks

    plt.savefig(plot_file_path+'/Winning_hyps_'+str(params.n_runs)+'_R_speaker_'+str(speaker_perspective)+'_speaker_order_'+str(params.speaker_order_type)+'_prior_'+str(params.perspective_prior_type)+'.png')
    plt.show()











def plot_posterior_over_hyps(simulation_type, n_contexts, cut_off_point, plot_file_path, plot_file_title, plot_title, x_label, y_label, hypothesis_space, perspective_hyps, lexicon_hyps, multi_run_log_posterior_matrix, which_lexicon_hyps, which_hyps_on_graph, std):
    sns.set_style("darkgrid")
    sns.set_palette("deep")
    palette = itertools.cycle(sns.color_palette()[3:])


    if simulation_type == 'multi_run':

        unlogged_multi_run_final_log_posterior = np.exp(multi_run_log_posterior_matrix)

        if n_contexts == 0:
            unlogged_multi_run_final_log_posterior = np.array([unlogged_multi_run_final_log_posterior])


        #TODO: Now this counts on having just 1 agent in each generation! If there are more agents in the population we'll have to average the posteriors in a different way
        unlogged_posterior_mean = np.mean(unlogged_multi_run_final_log_posterior, axis=0)
        unlogged_posterior_std = np.std(unlogged_multi_run_final_log_posterior, axis=0)
        unlogged_posterior_median = np.percentile(unlogged_multi_run_final_log_posterior, q=50, axis=0)
        unlogged_posterior_percentile_25 = np.percentile(unlogged_multi_run_final_log_posterior, q=25, axis=0)
        unlogged_posterior_percentile_75 = np.percentile(unlogged_multi_run_final_log_posterior, q=75, axis=0)


    #TODO: Note that the code below expects the population to consist of only one agent per generation!
    elif simulation_type == 'one_long_run':
        unlogged_multi_run_log_posterior_matrix = np.exp(multi_run_log_posterior_matrix[0])
        unlogged_posterior_mean = np.mean(unlogged_multi_run_log_posterior_matrix, axis=0)[0]
        unlogged_posterior_std = np.std(unlogged_multi_run_log_posterior_matrix, axis=0)[0]


    #TODO: Now this counts on having just 1 agent in each generation! (hence the x_index [0] below). If there are more agents in the population we'll have to average the posteriors in a different way
    if which_hyps_on_graph == 'all_hyps':
        if len(unlogged_posterior_mean.shape) == 2:
            avg_posterior = unlogged_posterior_mean[0]
        elif len(unlogged_posterior_mean.shape) == 1:
            avg_posterior = unlogged_posterior_mean
        if len(unlogged_posterior_std.shape) == 2:
            std_posterior = unlogged_posterior_std[0]
        elif len(unlogged_posterior_std.shape) == 1:
            std_posterior = unlogged_posterior_std

    elif which_hyps_on_graph == 'lex_hyps_only':
        if len(unlogged_posterior_mean.shape) == 2:
            avg_posterior_split_on_p_hyps = np.split(unlogged_posterior_mean[0], len(perspective_hyps))
        elif len(unlogged_posterior_mean.shape) == 1:
            avg_posterior_split_on_p_hyps = np.split(unlogged_posterior_mean, len(perspective_hyps))
        avg_posterior = np.sum(avg_posterior_split_on_p_hyps, axis=0)


    if which_hyps_on_graph == 'all_hyps':
        x_index = len(hypothesis_space)
    elif which_hyps_on_graph == 'lex_hyps_only':
        x_index = len(lexicon_hyps)
    bar_width = 0.35


    new_lex_order_automated = get_sorted_lex_hyp_order(lexicon_hyps)

    new_hyp_order_handsorted_all_hyps = np.array([1, 3, 2, 5, 6, 7, 0, 4, 8, 10, 12, 11, 14, 15, 16, 9, 13, 17])

    new_hyp_order_handsorted_lex_hyps_only = np.array([1, 3, 2, 5, 6, 7, 0, 4, 8])

    if which_hyps_on_graph == 'all_hyps':
        new_hyp_order = new_hyp_order_handsorted_all_hyps
    elif which_hyps_on_graph == 'lex_hyps_only':
        new_hyp_order = new_hyp_order_handsorted_lex_hyps_only

    if len(avg_posterior.shape) == 1:
        avg_posterior_sorted = avg_posterior[new_hyp_order]

    elif len(avg_posterior.shape) == 2:
        for i in range(len(avg_posterior)):
            avg_posterior[i] = avg_posterior[i][new_hyp_order]
        avg_posterior_sorted = np.array(avg_posterior).ravel()

    if std == 'yes':
        std_posterior_split_on_p_hyps = np.split(std_posterior, len(perspective_hyps))
        for i in range(len(std_posterior_split_on_p_hyps)):
            std_posterior_split_on_p_hyps[i] = std_posterior_split_on_p_hyps[i][new_hyp_order]
        std_posterior_sorted = np.array(std_posterior_split_on_p_hyps).ravel()

    lexicon_hyps_sorted = lexicon_hyps[new_hyp_order_handsorted_lex_hyps_only]


    if which_hyps_on_graph == 'all_hyps':
        hyp_labels = []
        hyp_space = hypothesis_space
        for i in range(len(hyp_space)):
            p_hyp_index = hyp_space[i][0]
            lex_hyp_index = hyp_space[i][1]
            p_hyp = perspective_hyps[p_hyp_index]
            p_hyp_label = 'p'+str(int(p_hyp))
            lex_hyp_label = 'l'+str(int(lex_hyp_index))
            full_hyp_label = p_hyp_label+'+'+lex_hyp_label
            hyp_labels.append(full_hyp_label)

    elif which_hyps_on_graph == 'lex_hyps_only':
        hyp_labels = []
        for i in range(len(lexicon_hyps)):
            lexicon_label = 'l'+str(i+1)
            hyp_labels.append(lexicon_label)


    # First the bars themselves are plotted:
    if std == 'yes':
        rects = plt.bar(np.arange(x_index), avg_posterior_sorted, yerr=std_posterior_sorted)
    elif std == 'no':
        rects = plt.bar(np.arange(x_index), avg_posterior_sorted)
    # Then the different bar colours are set for the different perspective hypotheses (if all hyps are shown):
    if which_hyps_on_graph == 'all_hyps':
        count = 0
        for i in range(len(perspective_hyps)):
            color = next(palette)
            for j in range(len(lexicon_hyps)):
                rects[count].set_color(color)
                count += 1
    plt.ylim(0., 1.)
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title)
    plt.xticks(np.arange(x_index) + bar_width, hyp_labels, rotation=45)


    width = rects[0].get_width()
    # Then each bar gets the corresponding lexicon plotted above it:
    for i in range(len(avg_posterior_sorted)):
        # this is an inset axes over the main axes
        if which_lexicon_hyps == 'all' and which_hyps_on_graph == 'all_hyps':
            factor = 26 # I found this factor through trial and error, I have no idea why it works.
            lex_width = width/factor
            plt.axes([(0.082 + (0.043*(i+1))), width, lex_width, lex_width])
        elif which_lexicon_hyps == 'all' and which_hyps_on_graph == 'lex_hyps_only':
            factor = 13 # I found this factor through trial and error, I have no idea why it works.
            lex_width = width/factor
            plt.axes([(0.04 + (0.086*(i+1))), width, lex_width, lex_width])
        elif which_lexicon_hyps == 'all_with_full_s_space' and which_hyps_on_graph == 'all_hyps':
            factor = 19.4 # I found this factor through trial and error, I have no idea why it works.
            lex_width = width/factor
            plt.axes([((lex_width*1.7) + (width/14.5*(i+1))), width, lex_width, lex_width])
        elif which_lexicon_hyps == 'all_with_full_s_space' and which_hyps_on_graph == 'lex_hyps_only':
            factor = 9.7 # I found this factor through trial and error, I have no idea why it works.
            lex_width = width/factor
            plt.axes([(lex_width/(factor/1.8) + (width/7.25*(i+1))), width, lex_width, lex_width])
        if which_hyps_on_graph == 'all_hyps':
            hyp = hypothesis_space[i]
            lex_hyp_index = hyp[1]
            lexicon = lexicon_hyps_sorted[lex_hyp_index]
            plt.pcolor(lexicon, cmap=plt.cm.Blues, edgecolors='k', linewidths=1, vmin=0)
        elif which_hyps_on_graph == 'lex_hyps_only':
            plt.pcolor(lexicon_hyps_sorted[i], cmap=plt.cm.Blues, edgecolors='k', linewidths=1, vmin=0)
        plt.gca().invert_yaxis()
        plt.xticks([])
        plt.yticks([])
        # plt.xticks(np.arange(n_signals), ['s1', 's2'])
        # plt.gca().xaxis.tick_top()
        # plt.yticks(np.arange(n_meanings), ['m1', 'm2'])


    # plt.text((0.55*x_index), 0.7, 'tot. prop. converged = '+str(tot_prop_converged),
    # bbox={'sal_alpha':0.5, 'pad':10})

    plt.gcf().subplots_adjust(bottom=0.15) # This makes room for the xlabel which would otherwise be cut off because of the rotation of the xticks

    plt.savefig(plot_file_path+plot_file_title+'_cutoff_'+str(cut_off_point)+'_'+which_hyps_on_graph+'.png')
    plt.show()



def plot_one_iteration_run_hist_hypothesis(plot_file_path, plot_file_title, hypothesis_space, perspective_hyps, lexicon_hyps, lexicon_hyps_sorted, selected_hyp_per_generation_sorted, conf_intervals, which_hyps_on_graph, cut_off_point, text_size):
    sns.set_style("darkgrid")
    sns.set_palette("deep")
    sns.set(font_scale=text_size)

    if which_hyps_on_graph == 'all_hyps':
        hyp_labels = []
        hyp_space = hypothesis_space
        for i in range(len(hyp_space)):
            p_hyp_index = hyp_space[i][0]
            lex_hyp_index = hyp_space[i][1]
            p_hyp = perspective_hyps[p_hyp_index]
            p_hyp_label = 'p'+str(int(p_hyp))
            lex_hyp_label = 'l'+str(int(lex_hyp_index))
            full_hyp_label = p_hyp_label+'+'+lex_hyp_label
            hyp_labels.append(full_hyp_label)

        # n, bins, patches = plt.hist(selected_hyp_per_generation_sorted, bins=len(hypothesis_space))
        bins = np.arange((len(lexicon_hyps) * len(perspective_hyps)))
        plt.bar(bins, selected_hyp_per_generation_sorted, yerr=conf_intervals, error_kw=dict(ecolor='black', lw=1, capsize=5, capthick=1))

    elif which_hyps_on_graph == 'lex_hyps_only':
        hyp_labels = []
        for i in range(len(lexicon_hyps)):
            lex_hyp_index = i
            lex_hyp_label = 'l'+str(int(lex_hyp_index))
            hyp_labels.append(lex_hyp_label)

        # n, bins, patches = plt.hist(selected_hyp_per_generation_sorted, bins=len(lexicon_hyps))
        bins = np.arange(len(lexicon_hyps))

        plt.bar(bins, selected_hyp_per_generation_sorted, yerr=conf_intervals, error_kw=dict(ecolor='black', lw=1, capsize=5, capthick=1))


    elif which_hyps_on_graph == 'lex_hyps_collapsed':
        hyp_labels = ['unambg.', 'part ambg.', 'one-to-all', 'all-to-all']

        # n, bins, patches = plt.hist(selected_hyp_per_generation_sorted, bins=len(lexicon_hyps))
        bins = np.arange(4)
        plt.bar(bins, selected_hyp_per_generation_sorted)


    x_ticks_position_addition = np.array([0.4 for x in range (len(hyp_labels))])
    x_ticks_positions = np.add(bins, x_ticks_position_addition)

    if which_hyps_on_graph == 'all_hyps':
        plt.ylim(0.0, 1.0)
        plt.xticks(x_ticks_positions, hyp_labels, rotation=45)
        plt.xlabel('Hypotheses')
        plt.ylabel('Proportion of generations')
        plt.title('Proportion of generations that selects hypothesis')
    elif which_hyps_on_graph == 'lex_hyps_only':
        plt.ylim(0.0, 1.0)
        plt.xticks(x_ticks_positions, hyp_labels)
        plt.xlabel('Lexicon hypotheses')
        plt.ylabel('Proportion of generations')
        plt.title('Proportion of generations that selects hypothesis')
    elif which_hyps_on_graph == 'lex_hyps_collapsed':
        plt.ylim(0.0, 1.0)
        plt.xticks(x_ticks_positions, hyp_labels)
        plt.xlabel('Lexicon types')
        plt.ylabel('Proportion of generations')
        plt.title('Proportion of generations that selects lexicon type')

    # Then each bar gets the corresponding lexicon plotted above it:

    if which_hyps_on_graph == 'all_hyps':
        for i in range(len(hypothesis_space)):
            # this is an inset axes over the main axes
            plt.axes([0.125+(i*0.043), 0.85, 0.035, 0.035])
            hyp = hypothesis_space[i]
            lex_hyp_index = hyp[1]
            lexicon = lexicon_hyps_sorted[lex_hyp_index]
            plt.pcolor(lexicon, cmap=plt.cm.Blues, edgecolors='k', linewidths=1, vmin=0)
            plt.gca().invert_yaxis()
            plt.xticks([])
            plt.yticks([])

    elif which_hyps_on_graph == 'lex_hyps_only':
        for i in range(len(lexicon_hyps)):
            # this is an inset axes over the main axes
            plt.axes([0.128+(i*0.086), 0.75, 0.065, 0.065])
            hyp = hypothesis_space[i]
            lex_hyp_index = hyp[1]
            lexicon = lexicon_hyps_sorted[lex_hyp_index]
            plt.pcolor(lexicon, cmap=plt.cm.Blues, edgecolors='k', linewidths=1, vmin=0)
            plt.gca().invert_yaxis()
            plt.xticks([])
            plt.yticks([])

    plt.gcf().subplots_adjust(bottom=0.15) # This makes room for the xlabel which would otherwise be cut off because of the rotation of the xticks

    plt.savefig(plot_file_path+'Plot_Prop_Hyps_'+plot_file_title+'_cutoff_'+str(cut_off_point)+'.png')
    plt.show()



def plot_histogram_overlays(directory, filename, bottleneck_range, stationary_distributions_matrix, condition, lexicons, b_legend_cutoff, legend_location, y_upper_lim):
    sns.set_style("darkgrid")
    # # sns.set_palette("cubehelix", len(stationary_distributions_matrix))
    # sns.set_palette("YlOrBr", len(stationary_distributions_matrix))
    # palette = itertools.cycle(sns.color_palette())
    palette = itertools.cycle(sns.cubehelix_palette(n_colors=len(stationary_distributions_matrix)))
    for s in range(len(stationary_distributions_matrix)):
        stationary_distribution = stationary_distributions_matrix[s]
        bottleneck = bottleneck_range[s]
        # plt.plot(stationary_distribution, label='bottleneck = ' + str(bottleneck), color=next(palette))
        #
        if s < b_legend_cutoff or s > (len(stationary_distributions_matrix) - b_legend_cutoff)-1:
            plt.plot(np.arange(len(lexicons)), stationary_distribution, 'o', label='bottleneck = '+str(bottleneck), color=next(palette))
        elif s == b_legend_cutoff:
            plt.plot(np.arange(len(lexicons)), stationary_distribution, 'o', label='etc.', color=next(palette))
        else:
            plt.plot(np.arange(len(lexicons)), stationary_distribution, 'o', color=next(palette))


    plt.ylim(0.0, y_upper_lim)
    plt.xlim(-0.4, ((len(lexicons)-1)+0.4))
    plt.xticks(np.arange(len(lexicons)))
    plt.title('Proportion of generations that select hypothesis')
    plt.xlabel('Lexicon hypotheses')
    plt.ylabel('Proportion of generations')
    plt.legend(loc=legend_location)

    # Then each bar gets the corresponding lexicon plotted above it:

    for i in range(len(lexicons)):
        # this is an inset axes over the main axes
        plt.axes([0.128+(i*0.088), 0.75, 0.065, 0.065])
        lexicon = lexicons[i]
        plt.pcolor(lexicon, cmap=plt.cm.Blues, edgecolors='k', linewidths=1, vmin=0)
        plt.gca().invert_yaxis()
        plt.xticks([])
        plt.yticks([])



    plt.savefig(directory+'/Results/Plots/Iteration/Plot_stat_dist_overlay_'+filename+'_'+condition)
    plt.show()





def plot_iteration_lexicon_distance(plot_title, plot_file_path, plot_file_title, pop_lexicons_distance_percentiles):
    sns.set_style("darkgrid", {"xtick.major.size":4, "ytick.major.size":4})
    sns.set_palette("deep")
    color_lexicon_approximation = sns.xkcd_rgb["salmon pink"]
    fig, ax = plt.subplots(1)
    ax.plot(np.arange(params.n_iterations), pop_lexicons_distance_percentiles[1], label='Overall lexicon difference', color=color_lexicon_approximation)
    ax.fill_between(np.arange(params.n_iterations), pop_lexicons_distance_percentiles[0], pop_lexicons_distance_percentiles[2], facecolor=color_lexicon_approximation, alpha=0.3)
    # plt.ylim((-0.05, 1.05))
    ax.set_title(plot_title)
    ax.set_xlabel("Generations (with "+ params.turnover_type+" turnover)")
    ax.set_ylabel("Overall lexicon distance within population")
    ax.legend(loc='center right')
    plt.savefig(plot_file_path+'/Iter_Lex_Dstnce_Pop'+plot_file_title+'.png')
    plt.show()





