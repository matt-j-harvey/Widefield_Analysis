import os
from PIL import Image

"""
image1 = Image.open(r'path where the image is stored\file name.png')
im1 = image1.convert('RGB')
im1.save(r'path where the pdf will be stored\new file name.pdf')
"""
folder = r"/home/matthew/Pictures/TCA_Plots/Vis_2_TCA/"
output_folder = r"/home/matthew/Pictures/TCA_Plots/Vis_2_TCA_PDF/"
images = os.listdir(folder)

for image in images:
    image1 = Image.open(folder + image)
    im1 = image1.convert('RGB')
    im1.save(output_folder + image + ".pdf")