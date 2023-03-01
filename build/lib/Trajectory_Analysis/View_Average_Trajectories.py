import numpy as np
import matplotlib.pyplot as plt
import tables
from mpl_toolkits.mplot3d import axes3d
from tqdm import tqdm
import os
from sklearn.decomposition import IncrementalPCA, PCA, FastICA, FactorAnalysis
import pickle

from Files import Session_List
from Trajectory_Analysis import Create_Downsampled_Tensors
from Widefield_Utils import widefield_utils, Create_Activity_Tensor, Create_Video_From_Tensor


# Is There a standardisation over learning
# Can we predict errors Early In Trajectory
# Can we Decode Genotype
# Can we Decode Context


# Load Tensors
def load_tensors(nested_session_list, tensor_list):

    """
    Return A Nested List For Each Condition
    Mice - Sessions - Trials - Timepoints - Pixels
    """

    condition_1_tensors = []
    condition_2_tensors = []

    for mouse in tqdm(nested_session_list):
        mouse_condition_1_tensors = []
        mouse_condition_2_tensors = []

        for session in mouse:

            session_tensors = []

            for condition_name in tensor_list:
                condition_name = condition_name.replace('_onsets', '')
                condition_name = condition_name.replace('.npy', '')

                # Get Path Details
                mouse_name, session_name = widefield_utils.get_mouse_name_and_session_name(session)

                # Load Activity Tensor
                activity_tensor = np.load(os.path.join(tensor_save_directory, mouse_name, session_name, condition_name + "_Activity_Tensor.npy"), allow_pickle=True)
                session_tensors.append(activity_tensor)

            mouse_condition_1_tensors.append(session_tensors[0])
            mouse_condition_2_tensors.append(session_tensors[1])

        condition_1_tensors.append(mouse_condition_1_tensors)
        condition_2_tensors.append(mouse_condition_2_tensors)

    return condition_1_tensors, condition_2_tensors



def fit_model_incremental(nested_tensor_list, save_directory, n_components=3):

    # Create Model
    model = IncrementalPCA(n_components=n_components)

    # Fit Iteratively
    print("FIt Model")
    number_of_conditions = len(nested_tensor_list)
    for condition_index in range(number_of_conditions):
        condition = nested_tensor_list[condition_index]
        for mouse in condition:
            for session in mouse:
                print("Session Shape", np.shape(session))

                # Reshape Session
                number_of_trials, trial_length, number_of_pixels = np.shape(session)
                reshaped_session = np.reshape(session, (number_of_trials * trial_length, number_of_pixels))
                print("reshaped_session", np.shape(reshaped_session))

                # Fit Model
                model.partial_fit(reshaped_session)

    # Save Model
    save_filename = os.path.join(save_directory, 'model.sav')
    pickle.dump(model, open(save_filename, 'wb'))

    # View Components
    components = model.components_

    # Load Mask Details
    indicies, image_height, image_width = widefield_utils.load_tight_mask()

    # Load Colourmaps
    colourmap = widefield_utils.get_musall_cmap()

    component_index = 0
    for component in components:
        component_image = widefield_utils.create_image_from_data(component, indicies, image_height, image_width)
        component_magnitude = np.max(np.abs(component_image))
        plt.title(str(component_index))
        plt.imshow(component_image, cmap=colourmap, vmin=-component_magnitude, vmax=component_magnitude)
        plt.savefig(os.path.join(save_directory, "Component_" + str(component_index).zfill(3) + ".png"))
        plt.close()

        component_index += 1


def fit_model_full(combined_data, save_directory, n_components=20):

    # concat data
    combined_data = np.vstack(combined_data)
    print("Combined Data Shape", np.shape(combined_data))

    # Reshape Data
    number_of_trials, trial_length, number_of_pixels = np.shape(combined_data)
    combined_data = np.reshape(combined_data, (number_of_trials * trial_length, number_of_pixels))
    print("Combined Data Shape", np.shape(combined_data))

    # Create Model
    model = FactorAnalysis(n_components=n_components)

    # Fit Iteratively
    print("Fitting Model")

    # Fit Model
    model.fit(combined_data)

    # Save Model
    save_filename = os.path.join(save_directory, 'model.sav')
    pickle.dump(model, open(save_filename, 'wb'))

    # View Components
    components = model.components_

    # Load Mask Details
    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    indicies, image_height, image_width = widefield_utils.downsample_mask_further(indicies, image_height, image_width)

    # Load Colourmaps
    colourmap = widefield_utils.get_musall_cmap()

    component_index = 0
    for component in components:
        component_image = widefield_utils.create_image_from_data(component, indicies, image_height, image_width)
        component_magnitude = np.max(np.abs(component_image))
        plt.title(str(component_index))
        plt.imshow(component_image, cmap=colourmap, vmin=-component_magnitude, vmax=component_magnitude)
        plt.savefig(os.path.join(save_directory, "Component_" + str(component_index).zfill(3) + ".png"))
        plt.close()

        component_index += 1



def view_trajectories(nested_tensor_list, model_directory):

    print("Transforming Data")

    # Load Model
    model_filepath = os.path.join(model_directory, 'model.sav')
    model = pickle.load(open(model_filepath, 'rb'))

    # Plot Trajectories
    colour_list = ['b', 'r']
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(projection='3d')
    number_of_conditions = len(nested_tensor_list)
    for condition_index in range(number_of_conditions):
        condition = nested_tensor_list[condition_index]
        condition_colour = colour_list[condition_index]

        for mouse in condition:
            for session in mouse:
                print("Session Shape", np.shape(session))


                # Get Session Mean
                session_mean = np.mean(session, axis=0)

                # Reshape Session
                #number_of_trials, trial_length, number_of_pixels = np.shape(session)
                #reshaped_session = np.reshape(session, (number_of_trials * trial_length, number_of_pixels))
                #mean_trajectory = np.mean(reshaped_session, axis=0)



                # Transform Data
                print("Mean Trajectory Shape", np.shape(session_mean))
                transformed_data = model.transform(session_mean)
                print("Transformed Data Shape", np.shape(transformed_data))

                # Plot Trajectory
                axis_1.plot(transformed_data[:, 0], transformed_data[:, 1], transformed_data[:, 2], c=condition_colour, alpha=0.1)

    plt.show()


def load_combined_dataset(tensor_names, combined_data_directory):

    combined_dataset = []
    combined_trial_details = []

    for condition_name in tensor_names:
        condition_name = condition_name.replace('_onsets', '')
        condition_name = condition_name.replace('.npy', '')

        condition_data_file = os.path.join(combined_data_directory, condition_name + "_Combined_Downsampled_Data.h5")
        condition_data_storage = tables.open_file(condition_data_file, mode='r')
        condition_data = condition_data_storage.root['Data']
        condition_trial_key = condition_data_storage.root['Trial_Key']

        print("condition data", np.shape(condition_data))

        combined_dataset.append(condition_data)
        combined_trial_details.append(condition_trial_key)


    return combined_dataset, combined_trial_details




def view_session_average_trajectories(combined_data, combined_trial_details, model_directory):

    # Load Model
    model_filepath = os.path.join(model_directory, 'model.sav')
    model = pickle.load(open(model_filepath, 'rb'))

    # Create Figure
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(projection='3d')

    colour_list = ['b', 'r']

    number_of_conditions = len(combined_data)

    for condition_index in range(number_of_conditions):
        condition_data = combined_data[condition_index]
        print("COndition Data", np.shape(condition_data))

        condition_details = combined_trial_details[condition_index]
        session_list = condition_details[:, 1]


        unique_sessions = list(set(session_list))
        print("Unique sessions", unique_sessions)

        for session_index in unique_sessions:
            session_mask = np.where(session_list == session_index, 1, 0)
            session_trial_indicies = np.nonzero(session_mask)[0]
            print("Session trial indicies", session_trial_indicies)

            session_trials = condition_data[session_trial_indicies[0]:session_trial_indicies[-1]]
            print("Session trials", np.shape(session_trials))

            session_mean = np.mean(session_trials, axis=0)

            # Transform Data
            print("Mean Trajectory Shape", np.shape(session_mean))
            transformed_data = model.transform(session_mean)
            print("Transformed Data Shape", np.shape(transformed_data))

            # Plot Trajectory
            axis_1.plot(transformed_data[:, 0], transformed_data[:, 1], transformed_data[:, 2], c=colour_list[condition_index], alpha=0.3)

    plt.show()


### Correct Rejections Post Learning ###
analysis_name = "Unrewarded_Contextual_Modulation"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = widefield_utils.load_analysis_container(analysis_name)
stop_window = 52
trial_length = stop_window - start_window
save_directory = r"/media/matthew/External_Harddrive_3/Widefield_Trajectory_Analysis"
tensor_save_directory = r"/media/matthew/External_Harddrive_3/Widefield_Trajectory_Analysis/Tensors"

# Load Mask
indicies, image_height, image_width = widefield_utils.load_tight_mask()

# Load Session List
nested_session_list = Session_List.control_switching_nested

# Create Tensors
"""
for mouse in nested_session_list:
    for session in mouse:
        for condition in onset_files:
            Create_Activity_Tensor.create_activity_tensor(session, condition, start_window, stop_window, tensor_save_directory, align_within_mice=True, align_across_mice=True, baseline_correct=False)
"""

# Create Combined Dataset
for tensor in onset_files:
    Create_Downsampled_Tensors.create_downsampled_tensors(nested_session_list, tensor, tensor_save_directory, save_directory, trial_length, indicies, image_height, image_width, downsample_size=100)

# Load Combined Dataset
combined_data, combined_trial_details = load_combined_dataset(onset_files, save_directory)

# Set Save Directory
trajectory_base_directory = r"/media/matthew/Expansion/Widefield_Analysis/Trajectory_Analysis"
save_directory = os.path.join(trajectory_base_directory, analysis_name)
widefield_utils.check_directory(save_directory)

# Fit Model
fit_model_full(combined_data, save_directory)

# View Trajectories
view_session_average_trajectories(combined_data, combined_trial_details, save_directory)
#view_trajectories(combined_data, save_directory)
