import cv2
import os


session_list =[r"/media/matthew/External_Harddrive_1/Opto_Test/KPGC2.2G/2022_10_23_Opto_Test_No_Filter/Stimuli_4"]


for session in session_list:

    image_folder    = session
    video_name      = session + "video.avi"

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    print("Images", images)

    images.sort()
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), frameSize=(width, height), fps=25)  # 0, 12

    count = 0
    for image in images:
        print(count)
        video.write(cv2.imread(os.path.join(image_folder, image)))
        count += 1

    cv2.destroyAllWindows()
    video.release()
    print("Finished")
