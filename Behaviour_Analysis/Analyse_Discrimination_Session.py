import os
import Behaviour_Analysis_Functions
import numpy as np
from tqdm import tqdm

def analyse_discrimination_session(base_directory):

    # Load Behaviour Matrix
    behaviour_matrix = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)
    
    # Create Output Directory
    output_directory = os.path.join(base_directory, "Behavioural_Measures")
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    # Average Visual Performance
    visual_trial_outcome_list, visual_hits, visual_misses, visual_false_alarms, visual_correct_rejections, visual_d_prime = Behaviour_Analysis_Functions.analyse_visual_discrimination(behaviour_matrix)

    # Pack All This Into A Dictionary
    performance_dictionary = {
    "visual_trial_outcome_list.npy":visual_trial_outcome_list,
    "visual_hits.npy":visual_hits,
    "visual_misses.npy":visual_misses,
    "visual_false_alarms.npy":visual_false_alarms,
    "visual_correct_rejections":visual_correct_rejections,
    "visual_d_prime":visual_d_prime,
    }

    print("session: ", base_directory, "Visual D Prime: ", visual_d_prime)
    np.save(os.path.join(output_directory, "Performance_Dictionary.npy"), performance_dictionary)

