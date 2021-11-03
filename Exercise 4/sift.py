import os
import typing

import pandas as pd
import numpy as np
import cv2 as cv
import scipy.ndimage as sn
import matplotlib.pyplot as plt


def resize_image_w_scale_factor(original_img: np.ndarray, re_scale_factor: float = 0.2) -> np.ndarray:
    # Resize to percentage of original size
    width = int(original_img.shape[1] * re_scale_factor)
    height = int(original_img.shape[0] * re_scale_factor)
    resized_image = cv.resize(original_img, (width, height))

    return resized_image


def use_gaussian_filter(input_image: np.ndarray, sigma: float) -> np.ndarray:
    # Calculate filter size based on sigma as done in Matlab's imgaussfilt:
    #  https://ch.mathworks.com/help/images/ref/imgaussfilt.html

    filter_dim = int(2 * np.ceil(2 * sigma) + 1)

    blurred_img = cv.GaussianBlur(src=input_image.copy().astype(np.single),
                                  ksize=(filter_dim, filter_dim), sigmaX=sigma,
                                  borderType=cv.BORDER_REPLICATE)

    return blurred_img


def load_image(image_path: str, re_scale_factor: float = 0.2) -> np.ndarray:
    """
    Open for which we want to detect the features. Open them in grayscale and resize them to 0.2 their original size.

    Usually Resized image shape = (605, 807).

    :param image_path:
    :param re_scale_factor:
    :return: np.ndarray
    """
    # Read image in grayscale
    original_img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    resized_image = resize_image_w_scale_factor(original_img, re_scale_factor)

    return resized_image


def get_gradient_mag_and_dir(image):
    # Generate gradients for the image
    g_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
    g_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)

    # Get the magnitude and direction
    g_mag, g_dir = cv.cartToPolar(g_x, g_y, angleInDegrees=False)

    return g_dir, g_mag


def get_blurred_images(input_img: np.ndarray, num_scales: int, sigma: float = 1.6) -> np.ndarray:
    blurred_img_list = list()

    for scale_i in range(-1, num_scales + 2):
        sigma_i = np.power(2, scale_i / num_scales) * sigma

        # Apply Gaussian kernel with sigma_i = 2^(scale/num_scales) * sigma
        blurred_img_s = use_gaussian_filter(input_image=input_img.copy(), sigma=sigma_i)

        blurred_img_list.append(blurred_img_s)

    blurred_img_array = np.array(blurred_img_list)

    return blurred_img_array


def get_dogs(blurred_imgs: np.ndarray) -> np.ndarray:
    dog_list = list()
    for img_idx_i in range(blurred_imgs.shape[0] - 1):
        dog_i = blurred_imgs[img_idx_i + 1] - blurred_imgs[img_idx_i]
        dog_list.append(dog_i)

    dog_array = np.array(dog_list)

    return dog_array


def generate_dog_pyramid(
        input_img: np.ndarray,
        num_scales: int = 3,
        num_ocatves: int = 5,
        sigma: float = 1.6
) -> list:
    """
    Write code to compute the DoG pyramid by :
        1)  Computing 'num_scales + 3' blurred images per each octave.
        2)  Calculate DoGs using the blurred images. Each octave has 'num_scales + 2' DoGs


    :param input_img: Original image
    :param num_scales: Number of scales (scale factors for the gaussian kernels)
    :param num_ocatves: Number of pyramid levels (num_ocatves - 1 downsizings of the original image)
    :param sigma: Base sigma value
    :return:
    """
    pyramid_input_img = input_img.copy()

    # Get DOG pyramid
    dog_pyramid = list()
    for octave_i in range(num_ocatves):
        if octave_i > 0:
            # Downsize octave in relation to its size
            pyramid_input_img = resize_image_w_scale_factor(pyramid_input_img.copy(), re_scale_factor=0.5)

        # Get the Gaussians for each octave to compute DoG (returns 'num_scales + 3' blurred images)
        blurred_imgs_octave_i = get_blurred_images(input_img=pyramid_input_img.copy(), num_scales=num_scales,
                                                   sigma=sigma)

        # Using the blurred images compute the DoG
        dogs_octave_i = get_dogs(blurred_imgs_octave_i)
        dog_pyramid.append(dogs_octave_i)

    return dog_pyramid


def locate_keypoints(dog_pyramid: list) -> pd.DataFrame:
    """
    Compute the keypoints with non-maximum suppression and discard candidates with the contrast threshold.

    :param dog_pyramid:
    :return: Dictionary with indexes to find each keypoint in the dog_pyramid (octave, scale, height, width)
    """
    key_point_location = {'octave': list(), 'scale': list(), 'height': list(), 'width': list()}
    min_threshold_c = 0.04

    for octave_i in range(len(dog_pyramid)):
        # Set to 0 pixels below a threshold of C=0.04
        dog_octave_i = dog_pyramid[octave_i].copy()
        dog_octave_i[dog_octave_i < min_threshold_c] = 0

        # Search for local maxima voxels in 26 (8 + 9 + 9) nearest neighbors.
        #   We do this via the dilation operation
        local_max_values = sn.morphology.grey_dilation(input=dog_octave_i, size=(3, 9, 9))
        local_maxima_idx = np.where(dog_octave_i[1: 4] == local_max_values[1: 4])

        key_point_location['octave'] += list(octave_i * np.ones_like(local_maxima_idx[0]))
        key_point_location['scale'] += list(local_maxima_idx[0])
        key_point_location['height'] += list(local_maxima_idx[1])
        key_point_location['width'] += list(local_maxima_idx[2])

    return pd.DataFrame(key_point_location).set_index(['octave', 'scale'])


def main(image_dir: str, num_scales: int = 3, num_ocatves: int = 5, sigma: float = 1.6):
    """
    Write code to compute:
    % 1)    image pyramid. Number of images in the pyarmid equals
    %       'num_octaves'.
    % 2)    blurred images for each octave. Each octave contains
    %       'num_scales + 3' blurred images.
    % 3)    'num_scales + 2' difference of Gaussians for each octave.
    % 4)    Compute the keypoints with non-maximum suppression and
    %       discard candidates with the contrast threshold.
    % 5)    Given the blurred images and keypoints, compute the
    %       descriptors. Discard keypoints/descriptors that are too close
    %       to the boundary of the image. Hence, you will most likely
    %       lose some keypoints that you have computed earlier.
    """
    # Get paths to the images in image_dir
    image_paths = tuple([os.path.join(image_dir, image_filename) for image_filename in os.listdir(image_dir)])

    img_1 = load_image(image_path=image_paths[0], re_scale_factor=0.2)
    img_2 = load_image(image_path=image_paths[1], re_scale_factor=0.2)

    dog_pyramid_img_1 = generate_dog_pyramid(img_1, num_scales=num_scales, num_ocatves=num_ocatves, sigma=sigma)
    keypoints_img_1_df = locate_keypoints(dog_pyramid_img_1)

    descriptor_df = get_descriptors(img_1, keypoints_img_1_df, num_scales, sigma)
    print('hello')


def get_descriptors(img: np.ndarray, keypoints_img_df: pd.DataFrame, num_scales: int, sigma: float) -> pd.DataFrame:
    descriptor_dict = {
        'octave': list(), 'scale': list(), 'height': list(), 'width': list(), 'descriptor': list()
    }

    # Iterate the keypoints by grouping them first by the specific octave-scale combination we will use
    for (octave, scale), octave_scale_keypoints_df in keypoints_img_df.groupby(level=['octave', 'scale']):
        # Calculate new sigma to convolve the whole image
        new_s = scale - 1 + num_scales * octave  # TODO: Verify if this is a plus or a minus
        keypoint_sigma = np.power(2, new_s / num_scales) * sigma

        blurred_img_o_s = use_gaussian_filter(input_image=img.copy(), sigma=keypoint_sigma)

        g_dir, g_mag = get_gradient_mag_and_dir(blurred_img_o_s)

        # Iterate over keypoints for the (octave, scale) combination and get the descriptor for each one
        for row_index, row_values in octave_scale_keypoints_df.iterrows():
            # Extract keypoint patch
            width_indexes = (row_values['width'] - 8, row_values['width'] + 8)
            height_indexes = (row_values['height'] - 8, row_values['height'] + 8)

            # Discard keypoints/descriptors that are too close to the boundary of the image.
            if (width_indexes[0] < 0 or height_indexes[0] < 0 or width_indexes[1] > blurred_img_o_s.shape[1] or
                    height_indexes[1] > blurred_img_o_s.shape[0]):
                continue

            keypoint_patch_g_mag = g_mag[height_indexes[0]: height_indexes[1], width_indexes[0]: width_indexes[1]]

            keypoint_patch_g_dir = g_dir[height_indexes[0]: height_indexes[1], width_indexes[0]: width_indexes[1]]

            # Scale the norm of the gradients by their distance to the keypoint center.
            #  Create a patch of gaussian of 16x16 and sigma = 1.5*16 and multiply element-wise (weigh each pixel)
            gaussian_mask_1d = cv.getGaussianKernel(ksize=16, sigma=1.5 * 16)
            gaussian_mask_2d = np.multiply(gaussian_mask_1d.T, gaussian_mask_1d)

            keypoint_patch_g_mag = keypoint_patch_g_mag * gaussian_mask_2d

            # Divide the 16X16 grid into 8 4x4 sub-patches and calculate the histogram for each
            descriptor_patch_i = list()
            for sub_patch_idx_h in range(4):
                for sub_patch_idx_w in range(4):
                    sub_patch_g_mag = keypoint_patch_g_mag[
                                      sub_patch_idx_h * 4: sub_patch_idx_h * 4 + 4,
                                      sub_patch_idx_w * 4: sub_patch_idx_w * 4 + 4]

                    sub_patch_g_dir = keypoint_patch_g_dir[
                                      sub_patch_idx_h * 4: sub_patch_idx_h * 4 + 4,
                                      sub_patch_idx_w * 4: sub_patch_idx_w * 4 + 4]

                    hist_sub_patch_i, _ = np.histogram(sub_patch_g_dir.flatten(), weights=sub_patch_g_mag.flatten(),
                                                       range=(0, 2 * np.pi), bins=8)

                    descriptor_patch_i.append(hist_sub_patch_i)

            descriptor_patch_i = np.array(descriptor_patch_i).flatten()

            # Normalize descriptor so that it is invariant to illumination changes
            descriptor_patch_i = descriptor_patch_i/np.linalg.norm(descriptor_patch_i)

            # Add it to the dictionary
            descriptor_dict['octave'].append(octave)
            descriptor_dict['scale'].append(scale)
            descriptor_dict['height'].append(row_values['height'])
            descriptor_dict['width'].append(row_values['width'])
            descriptor_dict['descriptor'].append(np.array(descriptor_patch_i).flatten())

    return pd.DataFrame(descriptor_dict).set_index(['octave', 'scale', 'height', 'width'])


if __name__ == '__main__':
    main(image_dir='images')
