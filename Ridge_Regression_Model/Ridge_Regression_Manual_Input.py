import Crop_Mouse_Face
import Crop_Whisker_Pad
import Zero_Lick_Trace
import Zero_Running_Trace

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import sys

def crop_mouse_face(session_list):

    app = QApplication(sys.argv)
    selection_window = roi_selection_window(session_list)
    selection_window.show()

    app.exec_()


def ridge_regression_manual_input(session_list):

    app = QApplication(sys.argv)

    # Create Face Selection Window
    face_selection_window = Crop_Mouse_Face.face_selection_window(session_list)
    face_selection_window.show()

    # Create Whisker Selection Window
    whisker_selection_window = Crop_Whisker_Pad.whisker_pad_selection_window(session_list)
    whisker_selection_window.show()

    # Create Running Trace Zero Window
    running_trace_window = Zero_Running_Trace.running_threshold_window(session_list)
    running_trace_window.show()

    # Create Lick Trace Zero Window
    lick_trace_window = Zero_Lick_Trace.lick_threshold_window(session_list)
    lick_trace_window.show()

    app.exec_()