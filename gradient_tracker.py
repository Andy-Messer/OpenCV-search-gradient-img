##
# file gradient_tracker.py
#
# brief Realization of algorithm for tracking gradient
# section libraries_main Libraries
# - numpy - module
# - cv2 - module
# - json - module
#
# section author Author(s)
# - Created by Andrey Krotov on 31/03/2022
# - Modified by Andrey Krotov on 31/03/2022
#
# Copyright (c) 2022 Andrey Krotov. All rights reserved.

# Imports
import json
import cv2
import numpy as np


# -----------------------------------------------------------------------------#
class GradientTrackerApp:
    """! Realization of 'Gradient tracker' """
    # Filter data
    _default_filter = ''
    _kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    _default_filters = ['sobel', 'laplace', 'scharr']

    # Image data
    _src = []
    _bitmap_img = []

    # Color density factor
    _color_density = int()

    def __init__(self):
        """! Initializes the App. Setting configurations from config.json """
        print("Hello!")
        print("Installing the configuration...")

        json_config_file = open('config.json')
        config = json.load(json_config_file)['cfg_grad']

        # Setup filter.
        if config['filter'] not in self._default_filters:
            print("Bad filter, check config.json!")
            exit(0)
        self._default_filter = config['filter']

        # Setup color density.
        if isinstance(config['color_density'], int):
            if config['color_density'] <= 0 or config['color_density'] > 100:
                print("Wrong color density: negative number or 0 or more than 100, check config.json!")
                exit()
            self._color_density = config['color_density'] / 100
        else:
            print("Wrong color density: wrong type, check config.json!")
            exit()

    def run(self, path=None):
        """! Running the analyzer """
        # If the image has not yet been acquired
        if path is None:
            print("Write the path to the picture:")
            path = input()

            print("Beginning image preprocessing...")
            self._src = cv2.imread(path)

            # Filter selection
            if self._default_filter == 'laplace':
                img_filter = self._laplace_filter(self._src)
            elif self._default_filter == 'sobel':
                img_filter = self._sobel_filter(self._src)
            else:
                img_filter = self._scharr_filter(self._src)

            # Creating bitmap
            self._bitmap_img = self._bitmap_image(img_filter)
        else:
            self._src = cv2.imread(path)
            # Filter selection
            if self._default_filter == 'laplace':
                img_filter = self._laplace_filter(self._src)
            elif self._default_filter == 'sobel':
                img_filter = self._sobel_filter(self._src)
            else:
                img_filter = self._scharr_filter(self._src)

            # Creating bitmap
            self._bitmap_img = self._bitmap_image(img_filter)

        # Calculation of the largest zero submatrix
        area, pos_x, pos_y = self._find_biggest_submatrix(self._bitmap_img)
        boundaries = [pos_x, pos_y]

        print("Do you want to draw frame of it?(y/n)")
        want_to_draw = input()
        ans = ['y', 'n']
        while want_to_draw.lower() not in ans:
            print("Try again(y/n)")
            want_to_draw = input()
        if want_to_draw:
            self._draw_image_with_boundaries(self._src, boundaries)

    def _find_biggest_submatrix(self, matrix):
        """! Calculation of the largest zero submatrix """
        n, m = matrix.shape[:2]

        # Setup default values
        coordinates_x = 0
        coordinates_y = 0
        area = 0

        up = [-1] * m
        left = [0] * m
        right = [0] * m
        st = []

        def find_barrier(stack):
            """! find the maximal boundary in the row """
            while stack and up[stack[-1]] <= up[j]:
                stack.pop()
            return stack

        for i in range(n):
            # Calculate up values
            up = [i if matrix[i][j] == 1 else up[j] for j in range(m)]

            # Calculate left boundary
            st = []
            for j in range(m):
                st = find_barrier(st)
                left[j] = -1 if not st else st[-1]
                st.append(j)

            # Calculate right boundary
            st = []
            for j in range(m - 1, -1, -1):
                st = find_barrier(st)
                right[j] = m if not st else st[-1]
                st.append(j)

            # Calculate new coordinates
            for j in range(m):
                new_area = (i - up[j]) * (right[j] - left[j] - 1)
                if area < new_area:
                    area = new_area
                    coordinates_y = [up[j] + 1, i]
                    coordinates_x = [left[j] + 1, right[j] - 1]

        print("Rectangle coordinates: ")
        print(" - left corner: ", coordinates_x[0], ' ', coordinates_y[0])
        print(" - right corner: ", coordinates_x[1], ' ', coordinates_y[1])
        return area, coordinates_x, coordinates_y

    def _draw_image_with_boundaries(self, img, boundaries):
        """! Draws a frame on two corners """
        # Calculate corners
        left_up_corner = (boundaries[0][0], boundaries[1][0])
        right_down_corner = (boundaries[0][1], boundaries[1][1])

        # Drawing rectangle
        big_dip_oruby_color = [156, 14, 56]
        img = cv2.rectangle(img, left_up_corner, right_down_corner, thickness=2, color=big_dip_oruby_color)

        # Show image
        cv2.imshow("Lapalace_demo", img)

    def _laplace_filter(self, img):
        """! Laplace filter -
        The Laplacian of an image highlights regions
        of rapid intensity change and is an example
        of a second order or a second derivative method of enhancement
        """
        # Make the image gray
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply 2D filtering
        dst = cv2.filter2D(img, cv2.CV_32F, kernel=self._kernel)
        filtered_image = cv2.convertScaleAbs(dst)
        return filtered_image

    def _sobel_filter(self, img, type_of_sobel_filter: str = 'xy'):
        """! Sobel filter -
        The Sobel-Feldman operator is a separable edge detection filter.
        """
        # Make the image gray
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply 2D filtering
        grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1)

        img_grad_x = cv2.convertScaleAbs(grad_x)
        img_grad_y = cv2.convertScaleAbs(grad_y)

        if type_of_sobel_filter == 'x':
            return img_grad_x
        elif type_of_sobel_filter == 'y':
            return img_grad_y
        img_grad_x_y = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
        return img_grad_x_y

    def _scharr_filter(self, img, type_of_scharr_filter: str = 'xy'):
        """! Scharr filter -
        This is a filtering method used to identify and highlight
        gradient edges/features using the 1st derivative.
        """
        # Make the image gray
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply 2D filtering
        grad_x = cv2.Scharr(img, cv2.CV_32F, 1, 0)
        grad_y = cv2.Scharr(img, cv2.CV_32F, 0, 1)

        img_grad_x = cv2.convertScaleAbs(grad_x)
        img_grad_y = cv2.convertScaleAbs(grad_y)

        if type_of_scharr_filter == 'x':
            return img_grad_x
        elif type_of_scharr_filter == 'y':
            return img_grad_y
        img_grad_x_y = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
        return img_grad_x_y

    def _bitmap_image(self, img):
        """! Calculate bitmap image """
        bitmap_image = []
        # Go by line
        for pixel_line in img:
            bitmap_pixel_line = []
            # Go by pixel
            for pixel in pixel_line:
                # Calculate the approximate pixel weight
                current_density = int(pixel)
                # Evaluate pixel
                if current_density < self._color_density * 255:
                    bitmap_pixel_line.append(0)
                else:
                    bitmap_pixel_line.append(1)
            # Add new line to mask
            bitmap_image.append(bitmap_pixel_line.copy())
        return np.array(bitmap_image)

    def _output_the_bitmap_image_to_file(self, img):
        """! dump bitmap to the file """
        output = open('output.txt', 'w')
        for line in img:
            str_line = ''
            for pixel in line:
                str_line += str(pixel)
            str_line += '\n'
            output.write(str_line)
        output.close()

    def __del__(self):
        cv2.waitKey(0)
        cv2.destroyAllWindows()
# -----------------------------------------------------------------------------#


if __name__ == '__main__':
    app = GradientTrackerApp()
    app.run()
