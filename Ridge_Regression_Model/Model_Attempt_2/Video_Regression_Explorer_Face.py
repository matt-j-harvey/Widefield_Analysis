import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os
from skimage.transform import resize
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pyqtgraph
from scipy import stats
import pickle
from tqdm import tqdm
import sys

import Regression_Utils

pyqtgraph.setConfigOptions(imageAxisOrder='row-major')


class coef_viewer(QWidget):

    def __init__(self, r2_map, regression_coefs, parent=None):
        super(coef_viewer, self).__init__(parent)

        # Setup Window
        self.setWindowTitle("Correlation_Coef_Viewer")
        self.setGeometry(0, 0, 1900, 500)
        self.r2_map = r2_map
        self.regression_coefs = regression_coefs

        # Create R2 Map Display View
        self.r2_display_view_widget = QWidget()
        self.r2_display_view_widget_layout = QGridLayout()
        self.r2_display_view = pyqtgraph.ImageView()
        self.r2_display_view.ui.histogram.hide()
        self.r2_display_view.ui.roiBtn.hide()
        self.r2_display_view.ui.menuBtn.hide()
        self.r2_display_view_widget_layout.addWidget(self.r2_display_view, 0, 0)
        self.r2_display_view_widget.setLayout(self.r2_display_view_widget_layout)
        self.r2_display_view.setImage(self.r2_map)

        # Create Regression Coef Display View
        self.coef_display_view_widget = QWidget()
        self.coef_display_view_widget_layout = QGridLayout()
        self.coef_display_view = pyqtgraph.ImageView()
        self.coef_display_view.ui.histogram.hide()
        self.coef_display_view.ui.roiBtn.hide()
        self.coef_display_view.ui.menuBtn.hide()
        self.coef_display_view_widget_layout.addWidget(self.coef_display_view, 0, 0)
        self.coef_display_view_widget.setLayout(self.coef_display_view_widget_layout)

        self.r2_display_view.getView().scene().sigMouseMoved.connect(lambda pos: self.change_pixel(pos, self.r2_display_view))

        # Create Index Map
        indicies, image_height, image_width = Regression_Utils.load_tight_mask_downsized()
        index_map = np.zeros(image_height * image_width)
        index_map[indicies] = list(range(len(indicies)))
        self.index_map = np.reshape(index_map, (image_height, image_width))

        # Set Colourmap
        #colourmap = Regression_Utils.get_musall_cmap()
        colors = [  [0.00, 0.87, 0.90, 1.00],
                    [0.00, 0.00, 1.00, 1.00],
                    [0.00, 0.00, 0.00, 1.00],
                    [1.00, 0.00, 0.00, 1.00],
                    [1.00, 1.00, 0.00, 1.00]]
        colors = np.array(colors)
        print("Colour Shape", np.shape(colors))
        colors = np.multiply(colors, 255)


        cmap = pyqtgraph.ColorMap(pos=np.linspace(0.0, 1.0, 5), color=colors)
        #cmap = pyqtgraph.colormap.getFromMatplotlib('hot')
        self.coef_display_view.setColorMap(cmap)

        self.layout = QGridLayout()
        self.layout.addWidget(self.coef_display_view_widget, 0, 0, 1, 1)
        self.layout.addWidget(self.r2_display_view_widget, 0, 1, 1, 1)
        self.setLayout(self.layout)

    def change_pixel(self, pos, imageview):
        pos = imageview.getImageItem().mapFromScene(pos)
        y = np.clip(int(pos.y()), a_min=0, a_max=100 - 1)
        x = np.clip(int(pos.x()), a_min=0, a_max=100 - 1)
        selected_pixel_index = int(self.index_map[y, x])
        regression_map = self.regression_coefs[selected_pixel_index]
        coef_magnitude = np.max(np.abs(regression_map))
        self.coef_display_view.setImage(regression_map)
        self.coef_display_view.setLevels(min=-0.00002, max=0.00002)




def load_downsampled_mask(base_directory):

    mask = np.load(os.path.join(base_directory, "Generous_Mask.npy"))

    # Transform Mask
    mask = resize(mask, (300, 304), preserve_range=True, order=0, anti_aliasing=True)

    image_height = np.shape(mask)[0]
    image_width = np.shape(mask)[1]

    mask = np.where(mask > 0.1, 1, 0)
    mask = mask.astype(int)
    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    return indicies, image_height, image_width

def load_smallest_mask(base_directory):

    indicies, image_height, image_width = load_downsampled_mask(base_directory)
    template = np.zeros(image_height * image_width)
    template[indicies] = 1
    template = np.reshape(template, (image_height, image_width))
    template = template[0:300, 0:300]
    template = resize(template, (100,100),preserve_range=True, order=0, anti_aliasing=True)
    template = np.reshape(template, 100 * 100)
    downsampled_indicies = np.nonzero(template)
    return downsampled_indicies, 100, 100

def reconstruct_r2_map(r2_values, base_directory):

    # Restructure Coef Matrix
    indicies, image_height, image_width = load_smallest_mask(base_directory)
    r2_map = Regression_Utils.create_image_from_data(r2_values, indicies, image_height, image_width)


    return r2_map


def reconstruct_coefs(regression_coefs, face_pixels):

    # Get Face Extent
    face_y_min = np.min(face_pixels[0])
    face_y_max = np.max(face_pixels[0])
    face_x_min = np.min(face_pixels[1])
    face_x_max = np.max(face_pixels[1])

    face_pixels = np.transpose(face_pixels)

    reconstructed_coefs = []
    for coef in tqdm(regression_coefs):
        template = np.zeros((480, 640))

        count = 0
        for pixel in face_pixels:
            template[pixel[0], pixel[1]] = coef[count]
            count += 1

        template = template[face_y_min:face_y_max, face_x_min:face_x_max]
        reconstructed_coefs.append(template)

        #plt.imshow(template)
        #plt.show()

    reconstructed_coefs = np.array(reconstructed_coefs)
    return reconstructed_coefs


def normalise_components(components):

    abs_components = np.abs(components)
    component_magnitude_vector = np.max(abs_components, axis=0)
    print("Component Magnitude Vector", np.shape(component_magnitude_vector))
    components = np.divide(components, component_magnitude_vector)
    return components


if __name__ == '__main__':

    app = QApplication(sys.argv)

    # Load Files
    base_directory =  r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Control_Data/NRXN78.1D/2020_12_07_Switching_Imaging"
    regression_coefs = np.load(os.path.join(base_directory, "Mousecam_Analysis", "Face_Video_Coefs.npy"))
    r2_values = np.load(os.path.join(base_directory, "Mousecam_Analysis", "Face_Video_R2.npy"))
    components = np.load(os.path.join(base_directory, "Mousecam_Analysis", "Face_SVD_Components.npy"))
    face_pixels = np.load(os.path.join(base_directory, "Mousecam_Analysis", "Whisker_Pixels.npy"))

    print("Regression coef shape", np.shape(regression_coefs))
    print("R2 Values", np.shape(r2_values))
    print("Components Shape", np.shape(components))
    print("Face Pixels", np.shape(face_pixels))

    # Normalise Components
    #components = normalise_components(components)

    # Multiple Components By Their CPD For Each Pixel
    components = np.dot(regression_coefs, components)
    print("Components Shape", np.shape(components))

    # Reconstruct Components Into Face Space
    components = reconstruct_coefs(components, face_pixels)
    print("Components Shape", np.shape(components))
    print("Components max", np.percentile(components, 99))
    print("Components min", np.percentile(components, 1))

    # Reconstrcut R2 Map
    r2_map = reconstruct_r2_map(r2_values, base_directory)

    # View These
    window = coef_viewer(r2_map, components)
    window.showMaximized()

    app.exec_()


