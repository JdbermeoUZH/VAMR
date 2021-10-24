import os
from typing import Tuple, Optional
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def generate_point_grid(number_points_x: int = 9, number_points_y: int = 6, square_size: float = 0.04,
                        show_plot: bool = False) -> np.array:
    x = np.linspace(0, (number_points_x - 1) * square_size, number_points_x)
    y = np.linspace(0, (number_points_y - 1) * square_size, number_points_y)
    xz, yz = np.meshgrid(x, y)
    z = np.zeros_like(xz)

    # 3-dim by n-points array
    point_array = np.concatenate([xz.reshape((1, xz.size)), yz.reshape((1, yz.size)), z.reshape((1, z.size))],
                                 axis=0)

    if show_plot:
        fig = plt.figure(figsize=(10, 18))
        ax = fig.add_subplot(111, projection='3d', azim=-40, elev=40)
        ax.scatter(xz, yz, z, "bx")
        plt.show()

    return point_array


def generate_cube_vertices(cube_square_length: int, cube_up_left_crnr_sqrs: Tuple[int, int],
                           square_size: float = 0.04) -> np.ndarray:
    # Get the upper left corner of the cube in the units of the world plane
    up_left_crnr = (cube_up_left_crnr_sqrs[0] * square_size, cube_up_left_crnr_sqrs[1] * square_size)

    # Create the matrix with the 8 vertices of the cube
    cube_coordinates = list()
    for z in (0, 1):
        for x in (0, 1):
            for y in (0, 1):
                cube_coordinates.append(
                    [x * square_size * cube_square_length + up_left_crnr[0],
                     y * square_size * cube_square_length + up_left_crnr[1],
                     -z * square_size * cube_square_length])

    return np.array(cube_coordinates).T


def get_pose(pose_file_path: str, img_line: int) -> np.array:
    with open(pose_file_path) as f:
        pose_lines = f.readlines()
        pose_line = pose_lines[img_line]
        pose_line = pose_line.strip()

    return np.array(pose_line.split(' '), dtype='float')


def get_rotation_matrix(rotation_vector_: np.array) -> np.array:
    rotation_magnitude = np.linalg.norm(rotation_vector_)
    k_unit_vector = rotation_vector_ / rotation_magnitude
    k_cross_prod_matrix = np.cross(k_unit_vector, np.identity(k_unit_vector.shape[0]) * -1)
    rotation_matrix = np.identity(rotation_vector_.shape[0]) + \
                      np.sin(rotation_magnitude) * k_cross_prod_matrix + \
                      (1 - np.cos(rotation_magnitude)) * np.matmul(k_cross_prod_matrix, k_cross_prod_matrix)

    return rotation_matrix


def get_projection_matrix(pose_array_: np.ndarray, k_matrix: np.ndarray) -> np.ndarray:
    # Create the rotation matrix
    rotation_vector = pose_array_[0:3]
    translation_vector = pose_array_[3:]
    rotation_matrix = get_rotation_matrix(rotation_vector)

    # Create rigid body transformation. Append the translation vector
    rb_transformation = np.concatenate(
        [rotation_matrix, translation_vector.reshape((translation_vector.shape[0], 1))], axis=1)

    # Multiply by the intrinsic parameter matrix K
    projection_matrix = np.matmul(k_matrix, rb_transformation)

    return projection_matrix


def apply_radial_deformation(optic_center_pixels: np.ndarray, projected_points_2D: np.ndarray, d_params: np.ndarray) \
        -> np.ndarray:
    points_wrt_optical_center = projected_points_2D - optic_center_pixels
    radial_dist_vector = np.linalg.norm(points_wrt_optical_center, axis=0)

    # Apply radial distortion model
    projected_points_2D = (1 +
                           d_params[0] * np.power(radial_dist_vector, 2) +
                           d_params[1] * np.power(radial_dist_vector, 4)) * points_wrt_optical_center + \
                          optic_center_pixels

    return projected_points_2D


def project_points(projection_matrix: np.ndarray, points: np.ndarray,
                   k_matrix: Optional[np.ndarray] = None, d_params: Optional[np.ndarray] = None,
                   apply_distortion=False) -> np.ndarray:
    # add a coordinate of 1 to be able to apply the translation of the rigid body transformation
    extended_points = np.concatenate([points, np.ones((1, points.shape[1]))], axis=0)
    projected_points = np.matmul(projection_matrix, extended_points)

    # We still need to divide by the 3rd projected dimension lambda
    projected_points_2D = projected_points[0:2, :] / projected_points[2, :]

    # Apply distortion if we want to apply distortion
    if apply_distortion and k_matrix is not None and d_params is not None:
        optic_center_pixels = np.array([[k_matrix[0, 2]], [k_matrix[1, 2]]])  # u_o, v_0

        # Apply radial distortion model
        projected_points_2D = apply_radial_deformation(
            optic_center_pixels=optic_center_pixels, projected_points_2D=projected_points_2D,
            d_params=d_params)

    return projected_points_2D


def generate_cube_2d_coordinates(cube_square_length: int, cube_up_left_crnr_sqrs: Tuple[int, int],
                                 projection_matrix: np.ndarray, square_size: float,
                                 k_matrix: Optional[np.ndarray] = None, d_params: Optional[np.ndarray] = None,
                                 apply_distortion=False) -> tuple[np.ndarray, np.ndarray]:
    cube_coordinates = generate_cube_vertices(cube_square_length=cube_square_length,
                                              cube_up_left_crnr_sqrs=cube_up_left_crnr_sqrs,
                                              square_size=square_size)

    projected_cube_vertices = project_points(projection_matrix=projection_matrix.copy(), points=cube_coordinates.copy(),
                                             k_matrix=k_matrix, d_params=d_params, apply_distortion=apply_distortion)

    return cube_coordinates, projected_cube_vertices


def plot_img_projected_objects(original_img: np.ndarray, projected_points: Optional[np.ndarray] = None,
                               square_size: Optional[float] = None, cube_coordinates: Optional[np.ndarray] = None,
                               projected_cube_vertices: Optional[np.ndarray] = None,
                               fig_name: Optional[str] = None):
    # Plot cube
    plt.imshow(original_img, cmap='gray')

    if projected_points is not None:
        plt.scatter(projected_points[0], projected_points[1], color='r', s=2)

    if projected_cube_vertices is not None and cube_coordinates is not None and square_size is not None:
        plt.scatter(projected_cube_vertices[0, :1], projected_cube_vertices[1, :1], color='b', s=10)
        connected_vertices = set()  # To make sure we only draw once each line
        for point_i in range(projected_cube_vertices.shape[1] - 1):
            for point_j in range(point_i + 1, projected_cube_vertices.shape[1]):
                dist = np.linalg.norm(cube_coordinates[:, point_i] - cube_coordinates[:, point_j])
                dist = np.round(dist, 3)

                if dist / square_size % 2 == 0 and (point_i, point_j) not in connected_vertices:
                    connected_vertices.add((point_i, point_j))
                    connected_vertices.add((point_j, point_i))

                    plt.plot(
                        [projected_cube_vertices[0, point_i], projected_cube_vertices[0, point_j]],
                        [projected_cube_vertices[1, point_i], projected_cube_vertices[1, point_j]],
                        'b'
                    )
    else:
        raise Exception("Missing either 'projected_cube_vertices', 'cube_coordinates', or 'square_size' ")

    plt.savefig(f'{fig_name}.jpg')
    plt.show()


def undistort_img(distorted_img: np.ndarray, img_shape: Tuple[int, int], d_params: np.ndarray, k_matrix: np.ndarray) \
        -> np.ndarray:
    x = np.linspace(0, img_shape[1] - 1, img_shape[1])
    y = np.linspace(0, img_shape[0] - 1, img_shape[0])
    xz, yz = np.meshgrid(x, y)

    # Array of individual pixel location in 2D
    pixel_idx_array = np.concatenate([xz.reshape((1, xz.size)), yz.reshape((1, yz.size))], axis=0)

    # Apply distortion to the indices of pixels
    optic_center_pixels = np.array([[k_matrix[0, 2]], [k_matrix[1, 2]]])
    pixel_idx_deformed = apply_radial_deformation(
        optic_center_pixels=optic_center_pixels,
        projected_points_2D=pixel_idx_array,
        d_params=d_params)

    pixel_idx_deformed = np.floor(pixel_idx_deformed).astype(int)

    undistorted_image = distorted_img[pixel_idx_deformed[1, :], pixel_idx_deformed[0, :]]\
        .reshape(distorted_img.shape).astype(np.uint8)

    return undistorted_image


def main(
        img_dir: str,
        img_idx: int = 0,
        square_size: float = 0.04,
        cube_square_length: int = 2,
        cube_up_left_crnr_sqrs: Tuple[int, int] = (1, 1),
        distorted_params_path: str = 'data/D.txt',
        intrinsic_param_path: str = 'data/K.txt',
        pose_data_path: str = 'data/poses.txt',
        apply_distortion: bool = False,
        undistort: bool = False,
        fig_name: Optional[str] = None,
):
    # Read undistorted image and convert to Gray-Scale
    original_img = cv.imread(os.path.join(img_dir, f'img_{img_idx:04}.jpg'), cv.IMREAD_GRAYSCALE)

    # Load relevant parameters
    pose_array = get_pose(pose_data_path, img_idx - 1)
    k = np.loadtxt(intrinsic_param_path)
    d = None
    if distorted_params_path is not None:
        d = np.loadtxt(distorted_params_path)

    # Undistort the image if so chose
    if undistort:
        original_img = undistort_img(distorted_img=original_img, img_shape=original_img.shape, d_params=d, k_matrix=k)
        apply_distortion = False

    # Create the grid of red points. Assume z=0, and (x=0,y=0) at the top left of the board
    #  Each square is of size 4cm
    point_grid_array = generate_point_grid(number_points_x=9, number_points_y=6, square_size=square_size,
                                           show_plot=False)

    # Use the the projection mapping to project the dots onto the surface
    projection_matrix = get_projection_matrix(pose_array_=pose_array, k_matrix=k)
    projected_points = project_points(projection_matrix=projection_matrix.copy(), points=point_grid_array.copy(),
                                      k_matrix=k, d_params=d, apply_distortion=apply_distortion)

    # Generate Cube Coordinates
    cube_coordinates, projected_cube_vertices = generate_cube_2d_coordinates(
        cube_square_length=cube_square_length, cube_up_left_crnr_sqrs=cube_up_left_crnr_sqrs,
        projection_matrix=projection_matrix, square_size=square_size,
        k_matrix=k, d_params=d, apply_distortion=apply_distortion)

    # Plot a cube on the image
    plot_img_projected_objects(original_img=original_img,
                               projected_points=projected_points,
                               square_size=square_size, cube_coordinates=cube_coordinates,
                               projected_cube_vertices=projected_cube_vertices,
                               fig_name=fig_name)


if __name__ == '__main__':
    data_dir = os.path.join('data')
    undistorted_img_dir = os.path.join(data_dir, 'images_undistorted')
    distorted_img_dir = os.path.join(data_dir, 'images')

    d_params_path = os.path.join(data_dir, 'D.txt')
    k_params_path = os.path.join(data_dir, 'K.txt')
    poses_path = os.path.join(data_dir, 'poses.txt')

    # EXERCISE 1
    # Plot undistorted image grid and cube
    main(img_dir=undistorted_img_dir, img_idx=1,
         square_size=0.04, cube_square_length=2, cube_up_left_crnr_sqrs=(1, 1),
         distorted_params_path=d_params_path, intrinsic_param_path=k_params_path, pose_data_path=poses_path,
         apply_distortion=False, undistort=False,
         fig_name='Projection on undistorted image')

    # EXERCISE 2.1 (or 3.1)
    # Plot image grid and cube on distorted image
    main(img_dir=distorted_img_dir, img_idx=1,
         square_size=0.04, cube_square_length=2, cube_up_left_crnr_sqrs=(1, 1),
         distorted_params_path=d_params_path, intrinsic_param_path=k_params_path, pose_data_path=poses_path,
         apply_distortion=True, undistort=False,
         fig_name='Projection on distorted image')

    # EXERCISE 2.2 (or 3.1)
    # Plot image grid and cube on distorted image and UNDISTORT IT
    main(img_dir=distorted_img_dir, img_idx=1,
         square_size=0.04, cube_square_length=2, cube_up_left_crnr_sqrs=(1, 1),
         distorted_params_path=d_params_path, intrinsic_param_path=k_params_path, pose_data_path=poses_path,
         apply_distortion=True, undistort=True,
         fig_name='Projection on undistorted image')

    # TODO: Use bilineal interpolation to do the undistortioning

    # TODO: Create animation to based on the sequence of images
