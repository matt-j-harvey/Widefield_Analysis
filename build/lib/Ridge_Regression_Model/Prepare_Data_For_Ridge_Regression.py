import Extract_Face_Motion
import Extract_Whisker_Motion
import Get_Running_Onsets
import Get_Lick_Events
import Match_Mousecam_Frames_To_Widefield_Frames

"""
Manual Preprocessing Steps Which Must Be Performed
Zero Running Trace
Zero Lick Trace
Crop Mouse Face
Crop Whisker Pad
Fit Deeplabcut Pupil
Fit Deeplabcut Limbs

# Automatic Preprocessing Steps Which Will Then Be Performed
Match Mousecam Frames To Widefield Frames
Get Lick Events
Get Running Onsets
Decompose Face Motion
Decompose Whisker Motion
"""

def prepare_data_for_ridge_model(base_directory):

    # Match Widefield To Mousecam Frames - Done
    Match_Mousecam_Frames_To_Widefield_Frames.match_mousecam_to_widefield_frames(base_directory)

    # Get Lick Events - Done
    Get_Lick_Events.get_lick_events(base_directory)

    # Get Running Onsets - Done
    Get_Running_Onsets.get_running_onsets(base_directory)

    # Extract Whisker Motion
    Extract_Whisker_Motion.extract_whisks(base_directory)

    # Extract Face Motion
    Extract_Face_Motion.extract_face_motion(base_directory)

    # Extract Limb Movements - Done




