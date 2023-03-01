import cv2
import os


session_list = [r"/media/matthew/29D46574463D2856/Nature_Transition_Analysis_Results/Absence_of_expected_Odour",
                r"/media/matthew/29D46574463D2856/Nature_Transition_Analysis_Results/Significance_Map/Individual_Timepoints"
]

for session in session_list:

    image_folder    = session
    video_name      = session + "_video.wmv"

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    print("Images", images)

    images.sort()
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), frameSize=(width, height), fps=12)  # 0, 12

    count = 0
    for image in images:
        print(count)
        video.write(cv2.imread(os.path.join(image_folder, image)))
        count += 1

    cv2.destroyAllWindows()
    video.release()
    print("Finished")
