"""Online Parametric Gaussian Process Regression
    Author: Esmaeil Rezaei, 10/02/2022

Example: 2D toy example

Steps:
    1- run the "DataGenerator.Py" to generate the data
    2 - run the "Main.py"
    3- When "Main.py" terminates, you can convert the
       figures to a video using the "Image2Video.py" script.
       Each figure corresponds to a separate stream of data.
"""
import cv2
import os
from natsort import natsorted

image_folder = 'FIGs/'
video_name = 'ReconstructedVideo.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images = natsorted(images)

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

video.release()
