import json

import cv2
import numpy as np
import json as js


# -----------------------------------------------------------------------------#



class GradientTrackerApp:

    _default_filter = ''
    _kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    _src = []
    _number_of_rectangles = 1
    _default_filters = ['sobel', 'laplace', 'scharr']
    _bitmap_img = []

    def __init__(self):
        print("Hello!")
        print("Installing the configuration...")
        json_config_file = open('config.json')
        config = json.load(json_config_file)['cfg_grad']
        if config['filter'] not in self._default_filters:
            print("Bad filter, check config.json!")
            exit(0)
        self._default_filter = config['filter']
        if self._default_filter == 'laplace':
            print('do you want to change from the default laplacian to another?')
        correct_ans = ['y', 'n']
        ans = input()
        while ans not in correct_ans:
            print("try again, check that it is one letter \'y\' or \'n\'")
            ans = input()
        if ans == 'y':
            self._set_laplacian_kernel()
        if isinstance(config['number_of_rectangles'], int):
            if config['number_of_rectangles'] <= 0:
                print("Wrong number of rectangles: negative number or 0, check config.json!")
                exit()
            self._number_of_rectangles = config['number_of_rectangles']
        else:
            print("Wrong number of rectangles: wrong type, check config.json!")
            exit()

    def _set_laplacian_kernel(self):
        laplacian = []
        for i in range(3):
            line = list(map(int, input().split()))
            laplacian.append(line)
        laplacian = np.array(laplacian)
        if laplacian.shape != (3, 3):
            print("Wrong kernel!")
            exit()
        self._kernel = laplacian
        print("correctly installed")

    def run(self, path=None):
        if path is None:
            print("Write the path to the picture:")
            path = input()
            print("Beginning image processing...")
            self._src = cv2.imread(path)
            if self._default_filter == 'laplace':
                img_filter = self._laplace_filter(self._src)
            elif self._default_filter == 'sobel':
                img_filter = self._sobel_filter(self._src)
            else:
                img_filter = self._scharr_filter(self._src)
            self._bitmap_img = self._bitmap_image(img_filter)
        else:
            self._src = cv2.imread(path)
            self._bitmap_img = self._bitmap_image(self._default_filter(self._src))
        self._output_the_bitmap_image_to_file(self._bitmap_img)
        ans, pos_x, pos_y = self._find_biggest_submatrix(self._bitmap_img)
        boundaries = [pos_x, pos_y]
        self._draw_image_with_boundaries(self._bitmap_img, boundaries)

        # for i in range(self._number_of_rectangles):
    #
    #
    # def _del_rectangle(self, matrix, boundaries):
    #     left_up_corner = (boundaries[2], boundaries[0])
    #     right_down_corner = (boundaries[3], boundaries[1])

    def _find_biggest_submatrix(self, matrix):
        n, m = matrix.shape[:2]
        ans_x = ans_y = 0
        ans = 0
        up = [-1] * m
        left = right = [0] * m
        st = []
        for i in range(n):
            for j in range(m):
                if matrix[i][j] == 1:
                    up[j] = i
            st = []
            for j in range(m):
                while st and up[st[-1]] <= up[j]:
                    st.pop()
                left[j] = -1 if not st else st[-1]
                st.append(j)
            st = []
            for j in range(m - 1, -1, -1):
                while st and up[st[-1]] <= up[j]:
                    st.pop()
                right[j] = m if not st else st[-1]
                st.append(j)
            for j in range(m):
                if ans < (i - up[j]) * (right[j] - left[j] - 1):
                    ans = (i - up[j]) * (right[j] - left[j] - 1)
                    ans_y = [up[j] + 1, i]
                    ans_x = [left[j] + 1, right[j] - 1]
        print(ans, ans_x, ans_y)
        return ans, ans_x, ans_y

    def _draw_image_with_boundaries(self, img, boundaries):
        left_up_corner = (boundaries[2], boundaries[0])
        right_down_corner = (boundaries[3], boundaries[1])
        big_dip_oruby_color = np.array([156, 14, 56])
        img = cv2.rectangle(img, left_up_corner, right_down_corner, color=big_dip_oruby_color)

    def _laplace_filter(self, img):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dst = cv2.filter2D(img, cv2.CV_32F, kernel=self._kernel)
        filtered_image = cv2.convertScaleAbs(dst)
        return filtered_image

    def _sobel_filter(self, img, type_of_sobel_filter: str = 'xy'):
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

    def _bitmap_image(self, img, color_density=768 / 10):
        bitmap_image = []
        for pixel_line in img:
            bitmap_pixel_line = []
            for pixel in pixel_line:
                current_density = (int(pixel[0]) + int(pixel[1]) + int(pixel[2]))
                if current_density < color_density:
                    bitmap_pixel_line.append(0)
                else:
                    bitmap_pixel_line.append(1)
            bitmap_image.append(bitmap_pixel_line.copy())
        return np.array(bitmap_image)

    def _output_the_bitmap_image_to_file(self, img):
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
