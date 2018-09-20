__author__ = 'Marieke Woensdregt'


import numpy as np



#############################################################################
# STEP 10: The functions below are used to write the results to a readable text file (.txt) and to a pickle file that is reusable by python (.p):



def convert_float_value_to_string(float_value):
    """
    :param array: A 1D numpy array
    :return: The numpy array converted into a string where spaces are replaced by underscores and brackets and dots are removed.
    """
    # if float_value % 1. == 0:
    #     float_value = int(float_value)
    float_string = str(float_value)
    float_string = float_string.replace(".", "")
    return float_string




def convert_array_to_string(array):
    """
    :param array: A 1D numpy array
    :return: The numpy array converted into a string where spaces are replaced by underscores and brackets and dots are removed.
    """
    array = np.around(array, decimals=2)
    array_string = str(array)
    array_string = array_string.replace("[", "")
    array_string = array_string.replace("]", "")
    array_string = array_string.replace(" ", "_")
    array_string = array_string.replace(".", "")
    return array_string

