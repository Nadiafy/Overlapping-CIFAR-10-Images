# Overlapping-CIFAR-10-Images
Overlapping CIFAR-10 Images: Dual Label Prediction

The goal of the project is to correctly classify two CIFAR-10 images, given their overlap.

The model takes as input an image created by averaging two random samples from CIFAR-10 and is tasked with predicting the categories of the two components.

The first image belongs to the first five categories (airplane, automobile, bird, cat, deer), while the second belongs to the remaining five categories (dog, frog, horse, ship, truck). The model must output two labels, each within a range of five possible values.

The evaluation metric for the model is as follows: calculate the classification accuracy for the two component images and then compute the average.

The metric should be evaluated on 10,000 inputs generated from test data. Repeat the calculation 10 times and measure the standard deviation, which must be reported.