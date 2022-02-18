import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions



def view_cortical_vector(base_directory, vector, plot_name, save_directory):

    # Load Mask
    indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)

    map_image = np.zeros(image_width * image_height)
    map_image[indicies] = vector
    map_image = np.ndarray.reshape(map_image, (image_height, image_width))

    map_magnitude = np.max(np.abs(vector))


    ax = plt.subplot()
    im = ax.imshow(map_image, cmap='bwr', vmin=-1 * map_magnitude, vmax=map_magnitude)

    session_name = base_directory.split('/')[-2]
    ax.set_title(session_name + " " + plot_name)
    ax.axis('off')
    # Add Colourbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.savefig(save_directory)
    plt.show()


def create_cortical_image_from_vector(indicies, image_height, image_width, vector):

    cortical_image = np.zeros(image_width * image_height)
    cortical_image[indicies] = vector
    cortical_image = np.ndarray.reshape(cortical_image, (image_height, image_width))

    return cortical_image


def view_regresssion_coefficients(base_directory):

    # Visual Stimuli = 0 - 100
    # running = 100
    # bodycam = 100 - 120

    # Load Mask Details
    indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)

    # Load Weights
    regression_directory = os.path.join(base_directory, "Movement_Controls", "Ridge_Regression")
    coefficients = np.load(os.path.join(regression_directory, "Coefficients.npy"))
    intercepts = np.load(os.path.join(regression_directory, "Intercepts.npy"))

    print("Coefficients Shape", np.shape(coefficients))
    condition_1_coefs = coefficients[:, 0:50]
    condition_2_coefs = coefficients[:,50:100]
    running_coefs = coefficients[: 100]
    bodycam_coefs = coefficients[:, 101:]


    figure_1 = plt.figure()

    coefficient_range = np.max(np.abs(coefficients))
    vmin = -1 * coefficient_range
    vmax = coefficient_range

    plt.ion()
    for x in range(50):

        condition_1_axis = figure_1.add_subplot(1,3,1)
        condition_2_axis = figure_1.add_subplot(1,3,2)
        difference_axis = figure_1.add_subplot(1,3,3)

        condition_1_vector = condition_1_coefs[:, x]
        condition_2_vector = condition_2_coefs[:, x]
        difference_vector = np.diff([condition_1_vector, condition_2_vector], axis=0)[0]

        condition_1_image = create_cortical_image_from_vector(indicies, image_height, image_width, condition_1_vector)
        condition_2_image = create_cortical_image_from_vector(indicies, image_height, image_width, condition_2_vector)
        difference_image = create_cortical_image_from_vector(indicies, image_height, image_width, difference_vector)

        condition_1_axis.imshow(condition_1_image, cmap='bwr', vmin=vmin, vmax=vmax)
        condition_2_axis.imshow(condition_2_image, cmap='bwr', vmin=vmin, vmax=vmax)
        difference_axis.imshow(difference_image, cmap='bwr', vmin=vmin, vmax=vmax)

        condition_1_axis.axis('off')
        condition_2_axis.axis('off')
        difference_axis.axis('off')


        plt.title(str(x))
        plt.draw()
        plt.pause(0.1)
        plt.clf()

    # View Other Regressors
    bodycam_components = np.load(os.path.join(regression_directory, "Ridge_Regression_Bodycam_Components.npy"))


    print("Bodycam Coeffs", np.shape(bodycam_coefs))
    number_of_bodycam_coefficients = np.shape(bodycam_coefs)[1]
    plt.ioff()
    for bodycam_coefficient_index in range(number_of_bodycam_coefficients):

        figure_1 = plt.figure()
        coefficient_axis = figure_1.add_subplot(1,2,1)
        component_axis = figure_1.add_subplot(1,2,2)

        coefficient = bodycam_coefs[:, bodycam_coefficient_index]
        coefficient_image = create_cortical_image_from_vector(indicies, image_height, image_width, coefficient)

        component = bodycam_components[bodycam_coefficient_index]
        component_image = np.reshape(component, (480, 640))

        coefficient_axis.imshow(coefficient_image, cmap='jet', vmin=0)
        component_axis.imshow(component_image, cmap='jet', vmin=0)
        plt.show()





controls = ["/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NRXN78.1A/2020_12_09_Switching_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NRXN78.1D/2020_11_29_Switching_Imaging"]

mutants =  ["/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_12_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK12.1F/2021_09_22_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NRXN71.2A/2020_12_17_Switching_Imaging"]

view_regresssion_coefficients(controls[0])