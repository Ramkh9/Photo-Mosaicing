import cv2
import numpy as np
import os
import random
import sys


def print_progress_bar(iteration, total, prefix='', suffix='', length=30, fill='█', print_end='\r'):
    percent = "{:.1f}".format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    # Print New Line on Complete
    if iteration == total:
        print()


# Double the pixel density using nearest-neighbor interpolation
def quadruple_image_density(image):
    new_width = image.shape[1] * 4
    new_height = image.shape[0] * 4
    quad_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("MainImageQuadrupled.jpg", quad_image)

    return quad_image


# Function to resize an image while maintaining aspect ratio
def resize_image(image, new_width, new_height):
    return cv2.resize(image, (new_width, new_height))


def choose_match(images, tile, epsilon, border=None):
    # Sending epsilon == 0 removes the randomness
    if border is not None:
        resized_images = [image[:border[0], :border[1]] for image in images]
    else:
        resized_images = images

    top_matches = sorted(resized_images, key=lambda img: np.linalg.norm(np.average(tile, axis=(0, 1)) - np.mean(img, axis=(0, 1))))[:2]
    if np.random.uniform(0, 1) < epsilon:
        match = top_matches[1]
    else:
        match = top_matches[0]
    return match


# Function to create a photo mosaic
def create_photo_mosaic(main_image_path, secondary_images_folder, output_path, tile_size, epsilon=0.1):

    # Load the main picture
    main_image = cv2.imread(main_image_path)

    # Get the dimensions of the main picture
    mosaic_height, mosaic_width = main_image.shape[:2]

    # Load and preprocess the secondary pictures
    secondary_images = []
    for filename in os.listdir(secondary_images_folder):
        image = cv2.imread(os.path.join(secondary_images_folder, filename))
        image = resize_image(image, tile_size, tile_size)
        secondary_images.append(image)

    # Initialize the output image
    mosaic = np.zeros_like(main_image)

    # For the progress bar
    total_tiles = ((mosaic_height + tile_size) // tile_size) * ((mosaic_width + tile_size) // tile_size)
    current_tile = 0

    # Divide the main picture into tiles and replace them with secondary pictures
    for y in range(0, mosaic_height, tile_size):
        for x in range(0, mosaic_width, tile_size):
            # Update the progress bar
            current_tile += 1
            print_progress_bar(current_tile, total_tiles, prefix='Progress:', suffix='Complete')

            _y = y + tile_size if mosaic_height-y >= tile_size else mosaic_height
            _x = x + tile_size if mosaic_width-x >= tile_size else mosaic_width

            tile = main_image[y:_y, x:_x, :]
            if mosaic_height - y >= tile_size and mosaic_width - x >= tile_size:
                border = None
            else:
                border = tile.shape[:2]
                # border = [border[-1], border[0]]

            # Find the best matching secondary picture based on color similarity
            # avg_color_tile = np.average(tile, axis=(0, 1))

            # best_match = min(secondary_images, key=lambda img: np.linalg.norm(avg_color_tile - np.mean(img, axis=(0, 1))))

            if y > mosaic_height/2 and x < 5*mosaic_width/6:        # this is for a section of the picture that you don't want to have randomness in tile selection
                best_match = choose_match(secondary_images, tile, 0, border)
            else:
                best_match = choose_match(secondary_images, tile, epsilon, border)

            # Convert both the tile and the selected secondary image to HSV color space
            tile_hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
            best_match_hsv = cv2.cvtColor(best_match, cv2.COLOR_BGR2HSV)

            # Adjust the hue, saturation, and value of the selected secondary image to match the tile's properties
            best_match_hsv[:, :, 0] = tile_hsv[:, :, 0]  # Hue

            # Saturation
            # best_match_hsv[:, :, 1] = tile_hsv[:, :, 1]

            # Calculate the average saturation of the tile and the selected secondary image
            avg_saturation_tile = np.average(tile_hsv[:, :, 1])
            avg_saturation_match = np.average(best_match_hsv[:, :, 1])

            # Interpolate the saturation of the selected secondary image to be closer to the tile's saturation
            alpha = 0.5  # Adjust this weight as needed
            interpolated_saturation = int((1 - alpha) * avg_saturation_match + alpha * avg_saturation_tile)

            # Apply Gaussian filtering to the interpolated saturation and value components
            best_match_hsv[:, :, 1] = cv2.GaussianBlur(best_match_hsv[:, :, 1], (5, 5), 0)

            best_match_hsv[:, :, 1] = interpolated_saturation  # Saturation

            # Convert the adjusted image back to BGR color space
            corrected_tile = cv2.cvtColor(best_match_hsv, cv2.COLOR_HSV2BGR)

            # Replace the tile with the corrected version
            mosaic[y:_y, x:_x] = corrected_tile

    # Save the resulting photo mosaic
    cv2.imwrite(output_path, mosaic)


def main():

    quad_image_path = "./MainImageQuadrupled.jpg"   # Quadrupled the pixels in the image
    main_image_path = "./<<>>.jpg"            # Main image path
    secondary_images_folder = "./Secondary_photos"  # Replace with the folder containing secondary images
    output_path = "./Results/result_pic.jpg"        # Replace with the desired output path
    tile_size = 120                                 # Adjust the tile size as needed
    epsilon = 0.3
    if True:                                        # a function to quadruple the image density to make the image have a higher pixel count
        quadruple_image_pixels(main_image_path)     # You need to turn this only on the first, use as it saves the image on disk

    create_photo_mosaic(quad_image_path, secondary_images_folder, output_path, tile_size, epsilon)


if __name__ == "__main__":
    main()




