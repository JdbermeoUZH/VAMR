import os
from typing import Tuple

import pandas as pd
import numpy as np
import cv2 as cv
import scipy.ndimage as sn
from scipy.spatial import distance
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


def generate_pyramid(
        input_img: np.ndarray,
        num_scales: int = 3,
        num_ocatves: int = 5,
        sigma: float = 1.6
) -> Tuple[list, list]:
    """
    Write code to compute the DoG pyramid by :
        1)  Computing 'num_scales + 3' blurred images per each octave.
        2)  Calculate DoGs using the blurred images. Each octave has 'num_scales + 2' DoGs


    :type sigma: object
    :param input_img: Original image
    :param num_scales: Number of scales (scale factors for the gaussian kernels)
    :param num_ocatves: Number of pyramid levels (num_ocatves - 1 downsizings of the original image)
    :param sigma: Base sigma value
    :return:
    """
    pyramid_input_img = input_img.copy()

    # Get DOG pyramid
    blurred_imgs_pyramid = list()
    dog_pyramid = list()
    for octave_i in range(num_ocatves):
        if octave_i > 0:
            # Downsize octave in relation to its size
            pyramid_input_img = resize_image_w_scale_factor(pyramid_input_img.copy(), re_scale_factor=0.5)

        # Get the Gaussians for each octave to compute DoG (returns 'num_scales + 3' blurred images)
        blurred_imgs_octave_i = get_blurred_images(input_img=pyramid_input_img.copy(), num_scales=num_scales,
                                                   sigma=sigma)
        blurred_imgs_pyramid.append(blurred_imgs_octave_i)

        # Using the blurred images compute the DoG
        dogs_octave_i = get_dogs(blurred_imgs_octave_i)
        dog_pyramid.append(dogs_octave_i)

    return blurred_imgs_pyramid, dog_pyramid


def locate_keypoints(dog_pyramid: list, min_threshold_c: float = 0.04) -> pd.DataFrame:
    """
    Compute the keypoints with non-maximum suppression and discard candidates with the contrast threshold.

    :param min_threshold_c:
    :param dog_pyramid:
    :return: Dictionary with indexes to find each keypoint in the dog_pyramid (octave, scale, height, width)
    """
    key_point_location = {'octave': list(), 'scale': list(), 'height': list(), 'width': list()}

    for octave_i in range(len(dog_pyramid)):
        # Set to 0 pixels below a threshold of C=0.04
        dog_octave_i = dog_pyramid[octave_i].copy()

        # Search for local maxima voxels in 26 (8 + 9 + 9) nearest neighbors.
        #   We do this via the dilation operation
        local_max_values = sn.morphology.grey_dilation(input=dog_octave_i, size=(3, 3, 3))
        local_maxima_idx = np.where(
            np.logical_and(dog_octave_i[1: 4] == local_max_values[1: 4], dog_octave_i[1: 4] >= min_threshold_c))

        key_point_location['octave'] += list(octave_i * np.ones_like(local_maxima_idx[0]))
        key_point_location['scale'] += list(local_maxima_idx[0])
        key_point_location['height'] += list(local_maxima_idx[1])
        key_point_location['width'] += list(local_maxima_idx[2])

    return pd.DataFrame(key_point_location).set_index(['octave', 'scale'])


def calculate_sift_descriptors(image, num_ocatves, num_scales, sigma):
    blurred_imgs_pyramid, dog_pyramid_img = generate_pyramid(
        image, num_scales=num_scales, num_ocatves=num_ocatves, sigma=sigma)
    keypoints_img_df = locate_keypoints(dog_pyramid_img)
    descriptor_df = get_descriptors(blurred_imgs_pyramid, keypoints_img_df)

    return descriptor_df


def get_descriptors(blurred_imgs_pyramid: list, keypoints_img_df: pd.DataFrame) \
        -> pd.DataFrame:
    descriptor_dict = {
        'octave': list(), 'scale': list(), 'height': list(), 'width': list(), 'descriptor': list()
    }

    # Iterate the keypoints by grouping them first by the specific octave-scale combination we will use
    for (octave, scale), octave_scale_keypoints_df in keypoints_img_df.groupby(level=['octave', 'scale']):
        # Calculate new sigma to convolve the whole image
        # Access the blurred images from the pyramid
        blurred_img_o_s = blurred_imgs_pyramid[octave][scale]

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


def match_features(img_1_descriptors_df: pd.DataFrame, img_2_descriptors_df: pd.DataFrame) -> pd.DataFrame:
    # Match the features between images
    # Calculate distances
    # https://pypi.org/project/fastdist/
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist
    dist_matrix = distance.cdist(
        XA=np.array(img_1_descriptors_df.descriptor.to_list()).astype(np.float16),
        XB=np.array(img_2_descriptors_df.descriptor.to_list()).astype(np.float16)
    )
    # Get indices of closest described features
    closest_two_features_df = pd.DataFrame(np.argpartition(dist_matrix, 2, axis=1)[:, :2],
                                           columns=('closest_idx', 'second_closest_idx'))
    # Get the value of these distances
    closest_two_features_df['closest_feature_dist'] = \
        np.take_along_axis(dist_matrix, closest_two_features_df.closest_idx.values.reshape((-1, 1)), axis=1)
    closest_two_features_df['second_closest_feature_dist'] = \
        np.take_along_axis(dist_matrix, closest_two_features_df.second_closest_idx.values.reshape((-1, 1)), axis=1)
    # Calculate their ratio and filter those whose closest feature is not at least 0.8 smaller than the second
    closest_two_features_df['dist_ratio'] = closest_two_features_df.closest_feature_dist / \
                                            closest_two_features_df.second_closest_feature_dist
    matched_features_df = closest_two_features_df[closest_two_features_df.dist_ratio < 0.8]
    # Reshape the dataframe to return only the number (indices) of the matched features
    matched_features_df = matched_features_df.reset_index()\
            .loc[:, ['index', 'closest_idx', 'closest_feature_dist', 'dist_ratio']]

    matched_features_df.columns = ['image_1_keypoint_idx', 'image_2_keypoint_idx', 'distance', 'distance_ratio']
    matched_features_df.sort_values(['distance_ratio'], inplace=True)

    return matched_features_df


def convert_to_opencv_keypoints(img_1_sift_df: pd.DataFrame, img_2_sift_df: pd.DataFrame,
                                matched_features_df: pd.DataFrame) -> Tuple[tuple, tuple, list]:
    keypoints_img_1_cv = list()
    keypoints_img_2_cv = list()
    match_list = list()
    for index, row in matched_features_df.reset_index().iterrows():
        # Image 1 Keypoint
        octave, scale, y, x = img_1_sift_df.iloc[int(row['image_1_keypoint_idx'])].name
        keypoints_img_1_cv.append(cv.KeyPoint(x=float(x), y=float(y), size=float(scale), octave=octave))

        # Image 2 Keypoint
        octave, scale, y, x = img_2_sift_df.iloc[int(row['image_2_keypoint_idx'])].name
        keypoints_img_2_cv.append(cv.KeyPoint(x=float(x), y=float(y), size=float(scale), octave=octave))

        # Match object
        match_list.append(cv.DMatch(_imgIdx=0, _queryIdx=index, _trainIdx=index,
                                    _distance=row['distance']))

    keypoints_img_1_cv = tuple(keypoints_img_1_cv)
    keypoints_img_2_cv = tuple(keypoints_img_2_cv)

    return keypoints_img_1_cv, keypoints_img_2_cv, match_list


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

    img_1 = load_image(image_path=image_paths[0], re_scale_factor=0.3)
    img_2 = load_image(image_path=image_paths[1], re_scale_factor=0.3)

    img_1_sift_df = calculate_sift_descriptors(img_1, num_ocatves, num_scales, sigma)
    img_2_sift_df = calculate_sift_descriptors(img_2, num_ocatves, num_scales, sigma)

    matched_features_df = match_features(img_1_sift_df, img_2_sift_df)

    # Plot the matched features using opencv's cv.drawMatches()
    #   Let's re-use the cv.Keypoint class for this
    keypoints_img_1_cv, keypoints_img_2_cv, custom_matches = convert_to_opencv_keypoints(
        img_1_sift_df, img_2_sift_df, matched_features_df)

    # Compare the features found with OpenCVs SIFT implementation and their matching function
    img_1_keypoints = cv.drawKeypoints(img_1, keypoints_img_1_cv, img_1)
    img_2_keypoints = cv.drawKeypoints(img_2, keypoints_img_2_cv, img_2)

    sift = cv.SIFT_create()
    kp_1, des_1 = sift.detectAndCompute(img_1, None)
    kp_2, des_2 = sift.detectAndCompute(img_2, None)

    img_1_keypoints_cv = cv.drawKeypoints(img_1, kp_1, img_1)
    img_2_keypoints_cv = cv.drawKeypoints(img_2, kp_2, img_2)

    # Compare the features
    f, axarr = plt.subplots(2, 2)
    f.tight_layout()
    axarr[0, 0].imshow(img_1_keypoints)
    axarr[0, 0].set_title('Self implemented SIFT')
    axarr[0, 1].imshow(img_1_keypoints_cv)
    axarr[0, 1].set_title("OpenCV's SIFT")

    axarr[1, 0].imshow(img_2_keypoints)
    axarr[1, 0].set_title('image 2', fontsize=8)
    axarr[1, 1].imshow(img_2_keypoints_cv)
    axarr[1, 1].set_title("image 2", fontsize=8)
    plt.show()

    # Compare the matches
    f, axarr = plt.subplots(2, 1)
    f.tight_layout()
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = bf.match(des_1, des_2)
    matches = sorted(matches, key=lambda x: x.distance)
    matched_features_opencv = cv.drawMatches(img_1, kp_1, img_2, kp_2, matches[:50], img_2, flags=2)
    axarr[0].imshow(matched_features_opencv)
    axarr[0].set_title("OpenCV's SIFT")

    # Plot self implemented matches
    matched_features_opencv = cv.drawMatches(img_1, keypoints_img_1_cv, img_2, keypoints_img_2_cv,
                                             custom_matches[:50], img_2, flags=2)
    axarr[1].imshow(matched_features_opencv)
    axarr[1].set_title("Self implemented SIFT")
    plt.show()

    print('hello')


if __name__ == '__main__':
    main(image_dir='images')
