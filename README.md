# Online Parametric Gaussian Process Regression

Author: Esmaeil Rezaei
Date: 10/02/2022

## Overview

This repository provides an example of Online Parametric Gaussian Process Regression using a 2D toy dataset. The code demonstrates how to generate data, run the main processing script, and create a video from the generated figures.

## Example: 2D Toy Dataset

The provided example focuses on a 2D toy dataset to illustrate the Online Parametric Gaussian Process Regression.

## Getting Started

Follow these steps to run the example:

1. Run the "DataGenerator.py" script to generate the required data.
2. Execute the "Main.py" script for the main processing.
3. After "Main.py" has terminated, you can convert the figures to a video by running the "Image2Video.py" script. Each figure in the video corresponds to a separate stream of data.

## Libraries Used

The example code utilizes the following libraries:

- [autograd](https://github.com/HIPS/autograd)
- [scikit-learn](https://scikit-learn.org/stable/)
- [pickle](https://docs.python.org/3/library/pickle.html)
- [matplotlib](https://matplotlib.org/)
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [datetime](https://docs.python.org/3/library/datetime.html)
- [pyDOE](https://github.com/tisimst/pyDOE)
- [cv2 (OpenCV)](https://opencv.org/)
- [natsort](https://github.com/SethMMorton/natsort)

Make sure to have these libraries installed to run the code successfully.
