import skimage.io
from skimage.measure import label, regionprops
from skimage.color import rgb2hsv
from skimage import morphology
import numpy as np


def image_to_binary(image, hue: float, margin: float):
    if isinstance(image, str):  # Check if input is a filepath
        image = skimage.io.imread(image)
    # Convert the image to the HSV color space
    hsv_image = rgb2hsv(image)
    # Extract the hue channel (green channel)
    hue_channel = hsv_image[:, :, 0]
    # Threshold the hue channel to obtain binary image
    lower_threshold = (hue - margin) / 360
    upper_threshold = (hue + margin) / 360
    if lower_threshold < 0:
        binary_image = np.where(
            (hue_channel >= 1 + lower_threshold) | (hue_channel <= upper_threshold),
            1,
            0,
        )
    elif upper_threshold > 1:
        binary_image = np.where(
            (hue_channel < upper_threshold - 1) | (hue_channel > lower_threshold), 1, 0
        )
    else:
        binary_image = np.where(
            (hue_channel >= lower_threshold) & (hue_channel <= upper_threshold), 1, 0
        )
    return binary_image


def binary_image_to_RGB(image):
    # this block remakes the binary image as rgb image by setting all pixels with 0s to [0,0,0] and all pixels with 1s to [255,255,255]
    rgb_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    rgb_image[image == 0] = [0, 0, 0]
    rgb_image[image == 1] = [255, 255, 255]
    return rgb_image


# hue, margin in degrees
def detect_centroids(filepath, hue: float, margin: float, min_size: int = None):
    if margin >= 180 or margin <= 0:
        raise ValueError("margin must be between 0 and 180 degrees")
    if hue >= 360 or hue < 0:
        raise ValueError("hue must be a value between 0 and 360 degrees")
    if min_size == None:
        min_size = 0

    image = skimage.io.imread(filepath)
    binary_image = image_to_binary(image, hue, margin)
    denoised_image = morphology.opening(binary_image, morphology.square(3))

    # scikit-image tool for collecting connected components
    label_im = label(denoised_image)
    region_data = regionprops(label_im)
    region_data = sorted(
        region_data, key=lambda x: x.area, reverse=True
    )  # sorting regions based on area so that largest area comes first

    centroids = []
    areas = []
    for r in region_data:
        if r.area < min_size:
            break
        centroids.append(r.centroid)
        areas.append(int(r.area))
    return centroids, areas
