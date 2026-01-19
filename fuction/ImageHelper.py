import copy
import math
from collections import deque

import cv2
import numpy as np
from numpy import convolve
from scipy.ndimage import label
from scipy.signal import convolve2d
from skimage.draw import line
from scipy.interpolate import interp1d, splprep, splev


class ImageHelper:
    @staticmethod
    def draw_spline_polyline_ex(points, image):
        line_points = np.array(points)
        drawed_image = ImageHelper.interpolate_and_rasterize_ex(line_points, image)
        return drawed_image

    @staticmethod
    def draw_spline_polyline_ex_2(p1, p2, image):
        line_p1 = ImageHelper.find_point_in_line(p1, image, 5)
        line_p4 = ImageHelper.find_point_in_line(p2, image, 5)
        if not line_p1:
            line_p1 = ImageHelper.find_point_in_line(p1, image, 2)
        if not line_p4:
            line_p4 = ImageHelper.find_point_in_line(p2, image, 2)
        if not line_p1:
            line_p1 = ImageHelper.find_point_in_line(p1, image, 1)
        if not line_p4:
            line_p4 = ImageHelper.find_point_in_line(p2, image, 1)

        line_points = np.array([line_p1, p1, p2, line_p4])
        drawed_image, lines_image = ImageHelper.interpolate_and_rasterize(line_points, image)
        # drawed_image_rgb, lines_image_rgb = ImageHelper.interpolate_and_rasterize(line_points, image)
        return drawed_image, lines_image

    @staticmethod
    def draw_spline_polyline(p1, p2, image):
        line_p1 = ImageHelper.find_point_in_line(p1, image, 5)
        line_p4 = ImageHelper.find_point_in_line(p2, image, 5)
        if not line_p1:
            line_p1 = ImageHelper.find_point_in_line(p1, image, 2)
        if not line_p4:
            line_p4 = ImageHelper.find_point_in_line(p2, image, 2)
        if not line_p1:
            line_p1 = ImageHelper.find_point_in_line(p1, image, 1)
        if not line_p4:
            line_p4 = ImageHelper.find_point_in_line(p2, image, 1)
        if not line_p4 or not line_p1:
            return None, None
        line_points = np.array([line_p1, p1, p2, line_p4])
        drawed_image, lines_image = ImageHelper.interpolate_and_rasterize(line_points, image)
        # drawed_image_rgb, lines_image_rgb = ImageHelper.interpolate_and_rasterize(line_points, image)
        return drawed_image, lines_image

    @staticmethod
    def interpolate_and_rasterize_rgb(points, binary_image):
        y = points[:, 0]
        x = points[:, 1]
        binary_image_copy = copy.deepcopy(binary_image)

        # 参数化：使用累积距离作为参数
        t = np.zeros(points.shape[0])
        t[1:] = np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2).cumsum()

        # 创建样条插值函数
        tck_x = interp1d(t, x, kind='cubic')
        tck_y = interp1d(t, y, kind='cubic')

        # 生成插值点
        t_new = np.linspace(t[1], t[2], num=100)
        x_new = tck_x(t_new)
        y_new = tck_y(t_new)

        x_int = np.round(x_new).astype(np.int32)
        y_int = np.round(y_new).astype(np.int32)

        # 转换binary_image到RGB
        rgb_image = np.stack([binary_image_copy] * 3, axis=-1)

        # 初始化RGB格式的line_image图层
        line_image = 255 * np.ones_like(rgb_image)

        # 栅格化曲线并绘制红色线条
        for i in range(len(x_int) - 1):
            rr, cc = line(y_int[i], x_int[i], y_int[i + 1], x_int[i + 1])
            rgb_image[cc, rr] = [255, 0, 0]  # 绘制红色线条
            line_image[cc, rr] = [255, 0, 0]  # 在line_image图层上绘制红色曲线

        return rgb_image, line_image

    @staticmethod
    def interpolate_and_rasterize(points, binary_image):
        """
        样条插值并栅格化曲线到给定的二值图像上。

        参数:
        - points: 控制点的数组，形式为[(y1, x1), (y2, x2), (y3, x3), (y4, x4)]
        - binary_image: 一个二值图像，其中前景像素为0，背景为1

        返回:
        - 修改后的二值图像，其中添加了栅格化的曲线
        """
        y = points[:, 0]
        x = points[:, 1]
        isOverHW = False
        binary_image_copy = copy.deepcopy(binary_image)

        # 参数化：使用累积距离作为参数
        t = np.zeros(points.shape[0])
        t[1:] = np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2).cumsum()

        # 创建样条插值函数
        tck_x = interp1d(t, x, kind='cubic')
        tck_y = interp1d(t, y, kind='cubic')

        # 生成插值点
        t_new = np.linspace(t[1], t[2], num=100)
        x_new = tck_x(t_new)
        y_new = tck_y(t_new)

        x_int = np.round(x_new).astype(np.int32)
        y_int = np.round(y_new).astype(np.int32)

        # 初始化line_image图层
        line_image = np.ones_like(binary_image)

        # # 获取图像的高度和宽度
        # img_height, img_width = binary_image.shape
        #
        # if max(y_int) >= img_height or min(y_int) < 0 or max(x_int) >= img_width or min(x_int) < 0:
        #     isOverHW = True
        # if isOverHW:
        #     drawed_image, line_image = ImageHelper.draw_line_on_binary_image_two_step(binary_image_copy,points[1], points[2])
        #     return drawed_image, line_image
        #
        # # 栅格化曲线
        # for i in range(len(x_int) - 1):
        #     rr, cc = line(y_int[i], x_int[i], y_int[i + 1], x_int[i + 1])
        #     if cc[0] == 471 or rr[0] == 471:
        #         print('find')
        #     binary_image[cc, rr] = 0  # 将前景像素设置为黑色
        #     line_image[cc, rr] = 0  # 在line_image图层上绘制曲线
        #
        # return binary_image, line_image

        # 栅格化曲线
        # for i in range(len(x_int) - 1):
        for i in range(len(x_int) - 1):
            rr, cc = line(y_int[i], x_int[i], y_int[i + 1], x_int[i + 1])
            if rr[0] >1354:
                continue
            binary_image[cc, rr] = 0  # 将前景像素设置为黑色
            line_image[cc, rr] = 0  # 在line_image图层上绘制曲线

        # for i in range(len(x_new) - 1):
        #     rr, cc = line(int(y_new[i]), int(x_new[i]), int(y_new[i + 1]), int(x_new[i + 1]))
        #     rr, cc = rr[(rr >= 0) & (rr < binary_image.shape[0]) & (cc >= 0) & (cc < binary_image.shape[1])], \
        #              cc[(rr >= 0) & (rr < binary_image.shape[0]) & (cc >= 0) & (cc < binary_image.shape[1])]
        #     binary_image[cc, rr] = 0  # 将前景像素设置为0
        #     line_image[cc, rr] = 0
        return binary_image, line_image

    @staticmethod
    def interpolate_and_rasterize_ex(points, binary_image):
        """
        使用样条插值在给定的二值图像上栅格化经过所有控制点的曲线。

        参数:
        - points: 控制点的数组，形式为[[y1, x1], [y2, x2], [y3, x3], [y4, x4]]
        - binary_image: 一个二值图像，其中前景像素为0，背景为1

        返回:
        - 修改后的二值图像，其中添加了栅格化的曲线
        """
        # 确保points是numpy数组
        points = np.array(points)
        y = points[:, 0]
        x = points[:, 1]

        # 参数化：使用累积距离作为参数
        t = np.zeros(len(points))
        t[1:] = np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2).cumsum()

        # 创建样条插值函数
        tck_x = interp1d(t, x, kind='cubic')
        tck_y = interp1d(t, y, kind='cubic')

        # 生成插值点
        t_new = np.linspace(t.min(), t.max(), num=1000)
        x_new = tck_x(t_new)
        y_new = tck_y(t_new)

        for i in range(len(x_new) - 1):
            rr, cc = line(int(y_new[i]), int(x_new[i]), int(y_new[i + 1]), int(x_new[i + 1]))
            rr, cc = rr[(rr >= 0) & (rr < binary_image.shape[0]) & (cc >= 0) & (cc < binary_image.shape[1])], \
                     cc[(rr >= 0) & (rr < binary_image.shape[0]) & (cc >= 0) & (cc < binary_image.shape[1])]
            binary_image[rr, cc] = 0  # 将前景像素设置为0

        return binary_image

    @staticmethod
    def interpolate_and_rasterize_b(points, binary_image):
        """
        使用B样条插值并栅格化曲线到给定的二值图像上。

        参数:
        - points: 控制点的数组，形式为[(y1, x1), (y2, x2), (y3, x3), (y4, x4)]
        - binary_image: 一个二值图像，其中前景像素为0，背景为1

        返回:
        - 修改后的二值图像，其中添加了栅格化的曲线
        """
        # 分离出x和y坐标
        x = points[:, 1]
        y = points[:, 0]

        # 使用splprep创建B样条曲线的参数表示，这里s参数控制平滑度
        tck, u = splprep([x, y], s=2)

        # 使用splev在新参数上评估B样条曲线
        u_new = np.linspace(u.min(), u.max(), 1000)
        x_new, y_new = splev(u_new, tck)

        # 四舍五入坐标值，并转换为整数类型
        x_int = np.round(x_new).astype(np.int32)
        y_int = np.round(y_new).astype(np.int32)

        # 初始化line_image图层
        line_image = np.ones_like(binary_image, dtype=np.uint8)

        # 栅格化曲线
        for i in range(len(x_int) - 1):
            rr, cc = line(y_int[i], x_int[i], y_int[i + 1], x_int[i + 1])
            binary_image[cc, rr] = 0  # 将前景像素设置为黑色
            line_image[cc, rr] = 0  # 在line_image图层上绘制曲线

        return binary_image, line_image

    @staticmethod
    def preprocess_image(image):
        # 进行噪声滤波
        denoised_image = cv2.medianBlur(image, 3)
        # plt.title('medianBlur')
        # plt.imshow(denoised_image, cmap='gray')
        # pylab.show()
        # pylab.pause(0)

        # 进行细化
        thinned_image = cv2.ximgproc.thinning(denoised_image)
        # plt.title('thinning')
        # plt.imshow(thinned_image, cmap='gray')
        # pylab.show()
        # pylab.pause(0)

        _, binary_image2 = cv2.threshold(thinned_image, 0, 255, cv2.THRESH_BINARY_INV)

        return binary_image2

    @staticmethod
    def binarization(path):
        src_img = cv2.imread(path)
        path = path[:path.rfind('\\')]
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        src_img = cv2.GaussianBlur(src_img, (3, 3), 0)
        # cv2.imwrite(path + r'\_gray.jpg', src_img)

        # 使用cv2.threshold进行二值化
        _, AdaptiveThreshold = cv2.threshold(src_img, 180, 255, cv2.THRESH_BINARY)
        AdaptiveThreshold = cv2.bitwise_not(AdaptiveThreshold)
        # AdaptiveThreshold = cv2.adaptiveThreshold(src_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, -2)
        # cv2.imwrite(path + r'\_AdaptiveThreshold.jpg', AdaptiveThreshold)

        # 定义核（结构元素）
        kernel = np.ones((3, 3), np.uint8)
        # 腐蚀操作
        erosion = cv2.erode(AdaptiveThreshold, kernel, iterations=1)
        # 膨胀操作
        AdaptiveThreshold = cv2.dilate(erosion, kernel, iterations=1)
        # AdaptiveThreshold = cv2.morphologyEx(AdaptiveThreshold, cv2.MORPH_OPEN, kernel)

        # plt.imshow(AdaptiveThreshold, cmap='gray')
        # plt.title('AdaptiveThreshold')
        # pylab.show()
        # pylab.pause(0)

        return AdaptiveThreshold

    @staticmethod
    def detect_endpoints_ori(image):
        # 获取图像形状
        height, width = image.shape

        endpoint_map = np.zeros_like(image, dtype=np.uint8)
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                if image[i, j] == 0:
                    if (image[i - 1, j - 1] == 0 and image[i - 1, j] == 255 and image[i - 1, j + 1] == 255
                            and image[i, j - 1] == 255 and image[i, j + 1] == 255
                            and image[i + 1, j - 1] == 255 and image[i + 1, j] == 255 and image[i + 1, j + 1] == 255):
                        endpoint_map[i, j] = 255
                    elif (image[i - 1, j - 1] == 255 and image[i - 1, j] == 0 and image[i - 1, j + 1] == 255
                          and image[i, j + 1] == 255 and image[i, j - 1] == 255
                          and image[i + 1, j + 1] == 255 and image[i + 1, j] == 255 and image[i + 1, j - 1] == 255):
                        endpoint_map[i, j] = 255
                    elif (image[i - 1, j - 1] == 255 and image[i - 1, j] == 255 and image[i - 1, j + 1] == 0
                          and image[i, j + 1] == 255 and image[i, j - 1] == 255
                          and image[i + 1, j + 1] == 255 and image[i + 1, j] == 255 and image[i + 1, j - 1] == 255):
                        endpoint_map[i, j] = 255
                    elif (image[i - 1, j - 1] == 255 and image[i - 1, j] == 255 and image[i - 1, j + 1] == 255
                          and image[i, j + 1] == 0 and image[i, j - 1] == 255
                          and image[i + 1, j + 1] == 255 and image[i + 1, j] == 255 and image[i + 1, j - 1] == 255):
                        endpoint_map[i, j] = 255
                    elif (image[i - 1, j - 1] == 255 and image[i - 1, j] == 255 and image[i - 1, j + 1] == 255
                          and image[i, j + 1] == 255 and image[i, j - 1] == 0
                          and image[i + 1, j + 1] == 255 and image[i + 1, j] == 255 and image[i + 1, j - 1] == 255):
                        endpoint_map[i, j] = 255
                    elif (image[i - 1, j - 1] == 255 and image[i - 1, j] == 255 and image[i - 1, j + 1] == 255
                          and image[i, j + 1] == 255 and image[i, j - 1] == 255
                          and image[i + 1, j + 1] == 0 and image[i + 1, j] == 255 and image[i + 1, j - 1] == 255):
                        endpoint_map[i, j] = 255
                    elif (image[i - 1, j - 1] == 255 and image[i - 1, j] == 255 and image[i - 1, j + 1] == 255
                          and image[i, j + 1] == 255 and image[i, j - 1] == 255
                          and image[i + 1, j + 1] == 255 and image[i + 1, j] == 0 and image[i + 1, j - 1] == 255):
                        endpoint_map[i, j] = 255
                    elif (image[i - 1, j - 1] == 255 and image[i - 1, j] == 255 and image[i - 1, j + 1] == 255
                          and image[i, j + 1] == 255 and image[i, j - 1] == 255
                          and image[i + 1, j + 1] == 255 and image[i + 1, j] == 255 and image[i + 1, j - 1] == 0):
                        endpoint_map[i, j] = 255

        # 找到检测结果中满足条件的像素坐标，即为终点位置
        endpoints = np.where(endpoint_map == 255)

        return list(zip(endpoints[1], endpoints[0]))

    @staticmethod
    def detect_endpoints(image):
        height, width = image.shape

        endpoint_map = np.zeros_like(image, dtype=np.uint8)

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                if image[i, j] == 0:
                    neighbors = image[i - 1:i + 2, j - 1:j + 2]
                    # 判断周围像素值是否符合端点条件
                    if np.sum(neighbors == 255) == 7 and neighbors[1, 1] == 0:
                        endpoint_map[i, j] = 255

        # 找到检测结果中满足条件的像素坐标，即为终点位置
        endpoints = np.where(endpoint_map == 255)
        return list(zip(endpoints[1], endpoints[0]))

    @staticmethod
    def find_direction_allmap(endpoint, image, h=20):
        # 改好则结束
        directions = []
        direction_points = []
        direction = -1
        interval = 1
        for i in range(h):
            direction_point = ImageHelper.find_point_in_line(endpoint, image, interval)
            if direction_point:
                direction_points.append(direction_point)
            interval += 1

        for direction_point in direction_points:
            x = direction_point[0]
            y = direction_point[1]
            if x == endpoint[0] and y < endpoint[1]:
                direction = 0
            if x > endpoint[0] and y < endpoint[1]:
                direction = 1
            if x > endpoint[0] and y == endpoint[1]:
                direction = 2
            if x > endpoint[0] and y > endpoint[1]:
                direction = 3
            if x == endpoint[0] and y > endpoint[1]:
                direction = 4
            if x < endpoint[0] and y > endpoint[1]:
                direction = 5
            if x < endpoint[0] and y == endpoint[1]:
                direction = 6
            if x < endpoint[0] and y < endpoint[1]:
                direction = 7
            if direction not in directions:
                directions.append(direction)
        return directions

    @staticmethod
    def find_direction(endpoint, image, h=10):
        # 改好则结束
        directions = []
        direction_points = []
        direction = -1
        interval = 4
        for i in range(h):
            direction_point = ImageHelper.find_point_in_line(endpoint, image, interval)
            if direction_point:
                direction_points.append(direction_point)
            interval += 1

        for direction_point in direction_points:
            x = direction_point[0]
            y = direction_point[1]
            if x == endpoint[0] and y < endpoint[1]:
                direction = 0
            if x > endpoint[0] and y < endpoint[1]:
                direction = 1
            if x > endpoint[0] and y == endpoint[1]:
                direction = 2
            if x > endpoint[0] and y > endpoint[1]:
                direction = 3
            if x == endpoint[0] and y > endpoint[1]:
                direction = 4
            if x < endpoint[0] and y > endpoint[1]:
                direction = 5
            if x < endpoint[0] and y == endpoint[1]:
                direction = 6
            if x < endpoint[0] and y < endpoint[1]:
                direction = 7
            if direction not in directions:
                directions.append(direction)
        return directions

    @staticmethod
    def find_direction_bezier(endpoint, image):
        # 改好则结束
        directions = []
        direction_points = []
        direction = -1
        interval = 2
        for i in range(10):
            direction_point = ImageHelper.find_point_in_line_bezier(endpoint, image, interval)
            if direction_point:
                direction_points.append(direction_point)
            interval += 1

        for direction_point in direction_points:
            x = direction_point[0]
            y = direction_point[1]
            if x == endpoint[0] and y < endpoint[1]:
                direction = 0
            if x > endpoint[0] and y < endpoint[1]:
                direction = 1
            if x > endpoint[0] and y == endpoint[1]:
                direction = 2
            if x > endpoint[0] and y > endpoint[1]:
                direction = 3
            if x == endpoint[0] and y > endpoint[1]:
                direction = 4
            if x < endpoint[0] and y > endpoint[1]:
                direction = 5
            if x < endpoint[0] and y == endpoint[1]:
                direction = 6
            if x < endpoint[0] and y < endpoint[1]:
                direction = 7
            if direction not in directions:
                directions.append(direction)
        return directions

    @staticmethod
    def connectCtl(endpoint, image, relative_points, paths_bfs, h):
        match_points = ImageHelper.find_match_points(endpoint, image, relative_points, paths_bfs, h)
        if match_points == (-1, -1):
            return -1, -1
        # step3: xy坐标差异
        if len(match_points) != 0:
            delXYs = []
            for point in match_points:
                delX = abs(point[0] - endpoint[0])
                delY = abs(point[1] - endpoint[1])
                delXYs.append(delX + delY)
            min_delXY = min(delXYs)
            match_point = match_points[delXYs.index(min_delXY)]

            match_point = list(match_point)
            index = relative_points.index(match_point)
            # relative_points_new[index][2] = True
        else:
            return -1, -1
        return match_point

    @staticmethod
    def connectCtl_allmap(endpoint, image, relative_points, h, step):
        match_points = ImageHelper.find_match_points_allmap(endpoint, image, relative_points, h, step)
        if match_points == (-1, -1):
            return -1, -1
        # step3: xy坐标差异
        if len(match_points) != 0:
            delXYs = []
            for point in match_points:
                delX = abs(point[0] - endpoint[0])
                delY = abs(point[1] - endpoint[1])
                delXYs.append(delX + delY)
            min_delXY = min(delXYs)
            match_point = match_points[delXYs.index(min_delXY)]

            match_point = list(match_point)
            index = relative_points.index(match_point)
            # relative_points_new[index][2] = True
        else:
            return -1, -1
        return match_point

    @staticmethod
    def connectCtl_original(endpoint, image, relative_points, paths_bfs, h=60):
        if endpoint[2]:
            return -1, -1
        # step1: 待匹配点搜索
        point_set = []
        if endpoint[0] == 44:
            print(1)
        for i in range(-h, h):
            for j in range(-h, h):
                if i == 0 and j == 0:
                    continue
                dx = endpoint[0] + i
                dy = endpoint[1] + j
                if [dx, dy] in relative_points:
                    # index = relative_points.index([dx, dy])
                    # if not relative_points_new[index][2]:
                    point_set.append((dx, dy))
        if len(point_set) == 0:
            return -1, -1
        # step2: 计算方向 || 八个方向 || 调整方向
        endpoint_x = endpoint[0]
        endpoint_Y = endpoint[1]
        if endpoint_x == 56 and endpoint_Y == 80:
            print(1)
        endpoint_directions = ImageHelper.find_direction(endpoint, image)
        match_points = []
        direction = -1

        for point in point_set:
            point_directions = ImageHelper.find_direction(point, image)
            x = point[0]
            y = point[1]

            if x == 56 and y == 92:
                print(1)
            if x == 44 and y == 88:
                print(1)
            for point_direction in point_directions:
                for endpoint_direction in endpoint_directions:
                    # 抛物线方向
                    if (point_direction == 1 and endpoint_direction == 3) and y < endpoint_Y and abs(
                            x - endpoint_x) <= 10:
                        match_points.append(point)
                    if (point_direction == 3 and endpoint_direction == 1) and y > endpoint_Y and abs(
                            x - endpoint_x) <= 10:
                        match_points.append(point)
                    if (point_direction == 3 and endpoint_direction == 5) and x > endpoint_x and abs(
                            y - endpoint_Y) <= 10:
                        match_points.append(point)
                    if (point_direction == 5 and endpoint_direction == 3) and x < endpoint_x and abs(
                            y - endpoint_Y) <= 10:
                        match_points.append(point)
                    if (point_direction == 5 and endpoint_direction == 7) and y > endpoint_Y:
                        match_points.append(point)
                    if (point_direction == 7 and endpoint_direction == 5) and y < endpoint_Y:
                        match_points.append(point)
                    if (point_direction == 7 and endpoint_direction == 1) and x < endpoint_x:
                        match_points.append(point)
                    if (point_direction == 1 and endpoint_direction == 7) and x > endpoint_x:
                        match_points.append(point)
                    # # 相反方向
                    if (point_direction == 1 and endpoint_direction == 5) and x > endpoint_x and y < endpoint_Y:
                        match_points.append(point)
                    if (point_direction == 5 and endpoint_direction == 1) and x < endpoint_x and y > endpoint_Y:
                        match_points.append(point)
                    if (point_direction == 3 and endpoint_direction == 7) and x > endpoint_x and y > endpoint_Y:
                        match_points.append(point)
                    if (point_direction == 7 and endpoint_direction == 3) and x < endpoint_x and y < endpoint_Y:
                        match_points.append(point)
                    if (point_direction == 2 and endpoint_direction == 6) and x > endpoint_x and abs(
                            y - endpoint_Y) <= 10:
                        match_points.append(point)
                    if (point_direction == 6 and endpoint_direction == 2) and x < endpoint_x and abs(
                            y - endpoint_Y) <= 10:
                        match_points.append(point)
                    if (point_direction == 0 and endpoint_direction == 4) and y < endpoint_Y and abs(
                            x - endpoint_x) <= 10:
                        match_points.append(point)
                    if (point_direction == 4 and endpoint_direction == 0) and y > endpoint_Y and abs(
                            x - endpoint_x) <= 10:
                        match_points.append(point)
        # 连通像素筛选【不可匹配短线的两个端点(不能在同一个连通path上)】
        endpoint_temp = (endpoint[0], endpoint[1])
        match_points_copy = copy.deepcopy(match_points)
        for path in paths_bfs:
            for point in match_points_copy:
                # point_temp = (point[1], point[0])
                if endpoint_temp in path and point in path:
                    match_points.remove(point)
        # step3: xy坐标差异
        if len(match_points) != 0:
            # step3: y坐标差异
            # delYs = []
            # for point in match_points:
            #     delY = abs(point[1] - endpoint[1])
            #     delYs.append(delY)
            # min_delY = min(delYs)
            # match_point = match_points[delYs.index(min_delY)]
            # step3: x坐标差异
            # delXs = []
            # for point in match_points:
            #     delX = abs(point[0] - endpoint[0])
            #     delXs.append(delX)
            # min_delX = min(delXs)
            # match_point = match_points[delXs.index(min_delX)]

            delXYs = []
            for point in match_points:
                delX = abs(point[0] - endpoint[0])
                delY = abs(point[1] - endpoint[1])
                delXYs.append(delX + delY)
            min_delXY = min(delXYs)
            match_point = match_points[delXYs.index(min_delXY)]

            match_point = list(match_point)
            index = relative_points.index(match_point)
            # relative_points_new[index][2] = True
        else:
            return -1, -1
        return match_point

    @staticmethod
    def find_match_points_allmap(endpoint, image, relative_points, h, step):
        if endpoint[2]:
            return -1, -1
        # step1: 待匹配点搜索
        point_set = []

        for i in range(-h, h):
            for j in range(-h, h):
                if i == 0 and j == 0:
                    continue
                dx = endpoint[0] + i
                dy = endpoint[1] + j
                if [dx, dy] in relative_points:
                    # index = relative_points.index([dx, dy])
                    # if not relative_points_new[index][2]:
                    point_set.append((dx, dy))
        if len(point_set) == 0:
            return -1, -1
        # step2: 计算方向 || 八个方向 || 调整方向
        endpoint_x = endpoint[0]
        endpoint_Y = endpoint[1]

        endpoint_directions = ImageHelper.find_direction_allmap(endpoint, image)
        match_points = []
        direction = -1

        for point in point_set:
            point_directions = ImageHelper.find_direction_allmap(point, image)
            x = point[0]
            y = point[1]

            # 计算角度
            p1 = endpoint
            p2 = point
            line = (p1, p2)
            # angel_cost = ImageHelper.get_angel_cost(line, image)
            # print('angel_cost:' + str(angel_cost))

            scale1 = 9999
            scale2 = 9999
            for point_direction in point_directions:
                for endpoint_direction in endpoint_directions:
                    # 抛物线方向
                    if (point_direction == 1 and endpoint_direction == 3) and y < endpoint_Y and abs(
                            x - endpoint_x) <= step:
                        match_points.append(point)
                    if (point_direction == 3 and endpoint_direction == 1) and y > endpoint_Y and abs(
                            x - endpoint_x) <= step:
                        match_points.append(point)
                    if (point_direction == 3 and endpoint_direction == 5) and x > endpoint_x and abs(
                            y - endpoint_Y) <= step:
                        match_points.append(point)
                    if (point_direction == 5 and endpoint_direction == 3) and x < endpoint_x and abs(
                            y - endpoint_Y) <= step:
                        match_points.append(point)
                    if (point_direction == 5 and endpoint_direction == 7) and y > endpoint_Y and abs(
                            x - endpoint_x) <= step:
                        match_points.append(point)
                    if (point_direction == 7 and endpoint_direction == 5) and y < endpoint_Y and abs(
                            x - endpoint_x) <= step:
                        match_points.append(point)
                    if (point_direction == 7 and endpoint_direction == 1) and x < endpoint_x and abs(
                            y - endpoint_Y) <= step:
                        match_points.append(point)
                    if (point_direction == 1 and endpoint_direction == 7) and x > endpoint_x and abs(
                            y - endpoint_Y) <= step:
                        match_points.append(point)
                    # # 相反方向 15 37方向加限制条件(最开始用的y/x的比例，现在还是用角度限制最合理)
                    if (
                            point_direction == 1 and endpoint_direction == 5) and x >= endpoint_x and y <= endpoint_Y :
                        match_points.append(point)
                    if (
                            point_direction == 5 and endpoint_direction == 1) and x <= endpoint_x and y >= endpoint_Y :
                        match_points.append(point)
                    if (
                            point_direction == 3 and endpoint_direction == 7) and x >= endpoint_x and y >= endpoint_Y :
                        match_points.append(point)
                    if (
                            point_direction == 7 and endpoint_direction == 3) and x <= endpoint_x and y <= endpoint_Y:
                        match_points.append(point)
                    if (point_direction == 2 and endpoint_direction == 6) and x >= endpoint_x and abs(
                            y - endpoint_Y) <= step:
                        match_points.append(point)
                    if (point_direction == 6 and endpoint_direction == 2) and x <= endpoint_x and abs(
                            y - endpoint_Y) <= step:
                        match_points.append(point)
                    if (point_direction == 0 and endpoint_direction == 4) and y <= endpoint_Y and abs(
                            x - endpoint_x) <= step:
                        match_points.append(point)
                    if (point_direction == 4 and endpoint_direction == 0) and y >= endpoint_Y and abs(
                            x - endpoint_x) <= step:
                        match_points.append(point)
        # 连通像素筛选【不可匹配短线的两个端点(不能在同一个连通path上)】
        endpoint_temp = (endpoint[0], endpoint[1])
        match_points_copy = copy.deepcopy(match_points)
        # for path in paths_bfs:
        #     for point in match_points_copy:
        #         # point_temp = (point[1], point[0])
        #         if endpoint_temp in path and point in path:
        #             match_points.remove(point)
        return match_points


    @staticmethod
    def find_match_points_conop(endpoint, image, relative_points, h, step):
        if endpoint[2]:
            return -1, -1
        # step1: 待匹配点搜索
        point_set = []

        for i in range(-h, h):
            for j in range(-h, h):
                if i == 0 and j == 0:
                    continue
                dx = endpoint[0] + i
                dy = endpoint[1] + j
                if [dx, dy] in relative_points:
                    # index = relative_points.index([dx, dy])
                    # if not relative_points_new[index][2]:
                    point_set.append((dx, dy))
        if len(point_set) == 0:
            return -1, -1
        # step2: 计算方向 || 八个方向 || 调整方向
        endpoint_x = endpoint[0]
        endpoint_Y = endpoint[1]
        if endpoint_x == 131 and endpoint_Y == 201:
            print(1)
        endpoint_directions = ImageHelper.find_direction_allmap(endpoint, image)
        match_points = []
        direction = -1

        for point in point_set:
            point_directions = ImageHelper.find_direction_allmap(point, image)
            x = point[0]
            y = point[1]

            # 计算角度
            p1 = endpoint
            p2 = point
            line = (p1, p2)
            angel_cost = ImageHelper.get_angel_cost(line, image)
            # print('angel_cost:' + str(angel_cost))

            if x == 143 and y == 192:
                print(1)

            scale1 = 3
            scale2 = 2.15
            for point_direction in point_directions:
                for endpoint_direction in endpoint_directions:
                    # 抛物线方向
                    if (point_direction == 1 and endpoint_direction == 3) and y < endpoint_Y and abs(
                            x - endpoint_x) <= step and angel_cost < scale1:
                        match_points.append(point)
                    if (point_direction == 3 and endpoint_direction == 1) and y > endpoint_Y and abs(
                            x - endpoint_x) <= step and angel_cost < scale1:
                        match_points.append(point)
                    if (point_direction == 3 and endpoint_direction == 5) and x > endpoint_x and abs(
                            y - endpoint_Y) <= step and angel_cost < scale1:
                        match_points.append(point)
                    if (point_direction == 5 and endpoint_direction == 3) and x < endpoint_x and abs(
                            y - endpoint_Y) <= step and angel_cost < scale1:
                        match_points.append(point)
                    if (point_direction == 5 and endpoint_direction == 7) and y > endpoint_Y and abs(
                            x - endpoint_x) <= step and angel_cost < scale1:
                        match_points.append(point)
                    if (point_direction == 7 and endpoint_direction == 5) and y < endpoint_Y and abs(
                            x - endpoint_x) <= step and angel_cost < scale1:
                        match_points.append(point)
                    if (point_direction == 7 and endpoint_direction == 1) and x < endpoint_x and abs(
                            y - endpoint_Y) <= step and angel_cost < scale1:
                        match_points.append(point)
                    if (point_direction == 1 and endpoint_direction == 7) and x > endpoint_x and abs(
                            y - endpoint_Y) <= step and angel_cost < scale1:
                        match_points.append(point)
                    # # 相反方向 15 37方向加限制条件(最开始用的y/x的比例，现在还是用角度限制最合理)
                    if (point_direction == 1 and endpoint_direction == 5) and x > endpoint_x and y < endpoint_Y and angel_cost < scale2:
                        match_points.append(point)
                    if (point_direction == 5 and endpoint_direction == 1) and x < endpoint_x and y > endpoint_Y and angel_cost < scale2:
                        match_points.append(point)
                    if (point_direction == 3 and endpoint_direction == 7) and x > endpoint_x and y > endpoint_Y and angel_cost < scale2:
                        match_points.append(point)
                    if (point_direction == 7 and endpoint_direction == 3) and x < endpoint_x and y < endpoint_Y and angel_cost < scale2:
                        match_points.append(point)
                    if (point_direction == 2 and endpoint_direction == 6) and x > endpoint_x and abs(
                            y - endpoint_Y) <= step and angel_cost < scale2:
                        match_points.append(point)
                    if (point_direction == 6 and endpoint_direction == 2) and x < endpoint_x and abs(
                            y - endpoint_Y) <= step and angel_cost < scale2:
                        match_points.append(point)
                    if (point_direction == 0 and endpoint_direction == 4) and y < endpoint_Y and abs(
                            x - endpoint_x) <= step and angel_cost < scale2:
                        match_points.append(point)
                    if (point_direction == 4 and endpoint_direction == 0) and y > endpoint_Y and abs(
                            x - endpoint_x) <= step and angel_cost < scale2:
                        match_points.append(point)
        return match_points


    @staticmethod
    def find_match_points_loose(endpoint, image, relative_points, paths_bfs, h):
        if endpoint[2]:
            return -1, -1
        # step1: 待匹配点搜索
        point_set = []
        for i in range(-h, h):
            for j in range(-h, h):
                if i == 0 and j == 0:
                    continue
                dx = endpoint[0] + i
                dy = endpoint[1] + j
                if [dx, dy] in relative_points:
                    point_set.append((dx, dy))
        if len(point_set) == 0:
            return -1, -1
        # step2: 计算方向 || 八个方向 || 调整方向
        endpoint_x = endpoint[0]
        endpoint_Y = endpoint[1]

        endpoint_directions = ImageHelper.find_direction(endpoint, image, 10)
        match_points = []
        direction = -1

        for point in point_set:
            point_directions = ImageHelper.find_direction(point, image, 10)
            x = point[0]
            y = point[1]

            for point_direction in point_directions:
                for endpoint_direction in endpoint_directions:
                    # 抛物线方向
                    if (point_direction == 1 and endpoint_direction == 3) and y < endpoint_Y and abs(
                            x - endpoint_x) <= 10:
                        match_points.append(point)
                    if (point_direction == 3 and endpoint_direction == 1) and y > endpoint_Y and abs(
                            x - endpoint_x) <= 10:
                        match_points.append(point)
                    if (point_direction == 3 and endpoint_direction == 5) and x > endpoint_x and abs(
                            y - endpoint_Y) <= 10:
                        match_points.append(point)
                    if (point_direction == 5 and endpoint_direction == 3) and x < endpoint_x and abs(
                            y - endpoint_Y) <= 10:
                        match_points.append(point)
                    if (point_direction == 5 and endpoint_direction == 7) and y > endpoint_Y and abs(
                            x - endpoint_x) <= 10:
                        match_points.append(point)
                    if (point_direction == 7 and endpoint_direction == 5) and y < endpoint_Y and abs(
                            x - endpoint_x) <= 10:
                        match_points.append(point)
                    if (point_direction == 7 and endpoint_direction == 1) and x < endpoint_x and abs(
                            y - endpoint_Y) <= 10:
                        match_points.append(point)
                    if (point_direction == 1 and endpoint_direction == 7) and x > endpoint_x and abs(
                            y - endpoint_Y) <= 10:
                        match_points.append(point)
                    # # 相反方向 15 37方向加限制条件(最开始用的y/x的比例，现在还是用角度限制最合理)
                    if (
                            point_direction == 1 and endpoint_direction == 5) and x > endpoint_x and y < endpoint_Y:
                        match_points.append(point)
                    if (
                            point_direction == 5 and endpoint_direction == 1) and x < endpoint_x and y > endpoint_Y:
                        match_points.append(point)
                    if (
                            point_direction == 3 and endpoint_direction == 7) and x > endpoint_x and y > endpoint_Y:
                        match_points.append(point)
                    if (
                            point_direction == 7 and endpoint_direction == 3) and x < endpoint_x and y < endpoint_Y:
                        match_points.append(point)
                    if (point_direction == 2 and endpoint_direction == 6) and x > endpoint_x and abs(
                            y - endpoint_Y) <= 20:
                        match_points.append(point)
                    if (point_direction == 6 and endpoint_direction == 2) and x < endpoint_x and abs(
                            y - endpoint_Y) <= 20:
                        match_points.append(point)
                    if (point_direction == 0 and endpoint_direction == 4) and y < endpoint_Y and abs(
                            x - endpoint_x) <= 20:
                        match_points.append(point)
                    if (point_direction == 4 and endpoint_direction == 0) and y > endpoint_Y and abs(
                            x - endpoint_x) <= 20:
                        match_points.append(point)
        # 连通像素筛选【不可匹配短线的两个端点(不能在同一个连通path上)】
        endpoint_temp = (endpoint[0], endpoint[1])
        match_points_copy = copy.deepcopy(match_points)
        for path in paths_bfs:
            for point in match_points_copy:
                # point_temp = (point[1], point[0])
                if endpoint_temp in path and point in path:
                    match_points.remove(point)
        if len(match_points) == 0:
            return -1, -1
        return match_points


    @staticmethod
    def find_match_points_for_bezier(endpoint, image, point_set, paths_bfs):
        # step: 计算方向 || 八个方向 || 调整方向
        endpoint_x = endpoint[0]
        endpoint_Y = endpoint[1]
        if endpoint_x == 131 and endpoint_Y == 201:
            print(1)
        endpoint_directions = ImageHelper.find_direction_bezier(endpoint, image)
        match_points = []

        for point in point_set:
            point_directions = ImageHelper.find_direction_bezier(point, image)
            x = point[0]
            y = point[1]

            if x == 143 and y == 192:
                print(1)

            for point_direction in point_directions:
                for endpoint_direction in endpoint_directions:
                    # 抛物线方向
                    if (point_direction == 1 and endpoint_direction == 3) and y < endpoint_Y:
                        match_points.append(point)
                    if (point_direction == 3 and endpoint_direction == 1) and y > endpoint_Y:
                        match_points.append(point)
                    if (point_direction == 3 and endpoint_direction == 5) and x > endpoint_x:
                        match_points.append(point)
                    if (point_direction == 5 and endpoint_direction == 3) and x < endpoint_x:
                        match_points.append(point)
                    if (point_direction == 5 and endpoint_direction == 7) and y > endpoint_Y:
                        match_points.append(point)
                    if (point_direction == 7 and endpoint_direction == 5) and y < endpoint_Y:
                        match_points.append(point)
                    if (point_direction == 7 and endpoint_direction == 1) and x < endpoint_x:
                        match_points.append(point)
                    if (point_direction == 1 and endpoint_direction == 7) and x > endpoint_x:
                        match_points.append(point)
                    # # 相反方向 15 37方向加限制条件(最开始用的y/x的比例，现在还是用角度限制最合理)
                    if (point_direction == 1 and endpoint_direction == 5) and x > endpoint_x and y < endpoint_Y and abs(x - endpoint_x) < 160:
                        match_points.append(point)
                    if (point_direction == 5 and endpoint_direction == 1) and x < endpoint_x and y > endpoint_Y and abs(x - endpoint_x) < 160:
                        match_points.append(point)
                    if (point_direction == 3 and endpoint_direction == 7) and x > endpoint_x and y > endpoint_Y:
                        match_points.append(point)
                    if (point_direction == 7 and endpoint_direction == 3) and x < endpoint_x and y < endpoint_Y:
                        match_points.append(point)
                    if (point_direction == 2 and endpoint_direction == 6) and x > endpoint_x:
                        match_points.append(point)
                    if (point_direction == 6 and endpoint_direction == 2) and x < endpoint_x:
                        match_points.append(point)
                    if (point_direction == 0 and endpoint_direction == 4) and y < endpoint_Y:
                        match_points.append(point)
                    if (point_direction == 4 and endpoint_direction == 0) and y > endpoint_Y:
                        match_points.append(point)
        # 连通像素筛选【不可匹配短线的两个端点(不能在同一个连通path上)】
        endpoint_temp = (endpoint[0], endpoint[1])
        match_points_copy = copy.deepcopy(match_points)
        for path in paths_bfs:
            for point in match_points_copy:
                if endpoint_temp in path and point in path:
                    match_points.remove(point)
        # xy坐标差异
        if len(match_points) != 0:
            delXYs = []
            for point in match_points:
                delX = abs(point[0] - endpoint[0])
                delY = abs(point[1] - endpoint[1])
                delXYs.append(delX + delY)
            min_delXY = min(delXYs)
            match_point = match_points[delXYs.index(min_delXY)]
            match_point = list(match_point)
            index = point_set.index(match_point)
            point_set[index][2] = True
        else:
            return -1, -1
        return match_point

    @staticmethod
    def find_path_with_step_limit(binary_image, start, step_limit):
        rows, cols = len(binary_image), len(binary_image[0])

        def is_valid(x, y):
            return 0 <= x < cols and 0 <= y < rows and binary_image[y, x] == 0

        def dfs(x, y, path, steps):
            if steps >= step_limit:
                path.append((x, y))
                return True

            if is_valid(x, y) and (x, y) not in visited and steps < step_limit:
                visited.add((x, y))
                path.append((x, y))

                # 从当前点的八个方向进行深度优先搜索
                directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
                for dx, dy in directions:
                    if dfs(x + dx, y + dy, path, steps + 1):
                        return True

                path.pop()

            return False

        visited = set()
        path = []

        if dfs(start[0], start[1], path, 0):
            return path
        else:
            return None

    @staticmethod
    def find_path_with_step_limit_bezier(binary_image, start, step_limit):
        rows, cols = len(binary_image), len(binary_image[0])

        def is_valid(x, y):
            return 0 <= y < cols and 0 <= x < rows and binary_image[x, y] == 0

        def dfs(x, y, path, steps):
            if steps >= step_limit:
                path.append((x, y))
                return True

            if is_valid(x, y) and (x, y) not in visited and steps < step_limit:
                visited.add((x, y))
                path.append((x, y))

                # 从当前点的八个方向进行深度优先搜索
                directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
                for dx, dy in directions:
                    if dfs(x + dx, y + dy, path, steps + 1):
                        return True

                path.pop()

            return False

        visited = set()
        path = []

        if dfs(start[0], start[1], path, 0):
            return path
        else:
            return None

    @staticmethod
    def find_point_in_line_bezier(point, image, iteration):
        path = ImageHelper.find_path_with_step_limit_bezier(image, point, iteration)
        if path is None:
            # print('未找到目标点')
            return
        return path[-2]

    @staticmethod
    def find_point_in_line(point, image, iteration):
        path = ImageHelper.find_path_with_step_limit(image, point, iteration)
        if path is None:
            # print('未找到目标点')
            return
        return path[-2]

    @staticmethod
    def draw_line_on_binary_image_two_step(image, point1, point2):
        x1, y1 = map(int, point1)
        x2, y2 = map(int, point2)
        image_temp = copy.deepcopy(image)
        height, width = image.shape
        line_image = np.ones((height, width), dtype=np.uint8)

        # 使用 Bresenham 算法计算线段上的像素点
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            image_temp[y1, x1] = 0
            line_image[y1, x1] = 0

            if x1 == x2 and y1 == y2:
                break

            e2 = 2 * err

            if e2 > -dy:
                err -= dy
                x1 += sx

                if e2 < dx and (y1 + sy >= 0) and (y1 + sy < image_temp.shape[0]) and (image_temp[y1 + sy, x1] == 0):
                    # 在这里加入垂直方向的步骤，仅设置像素，不改变 image_temp 的值
                    image_temp[y1, x1] = 0
                    line_image[y1, x1] = 0
            else:
                # 在这里加入水平方向的步骤，仅设置像素，不改变 image_temp 的值
                if (x1 + sx >= 0) and (x1 + sx < image_temp.shape[1]) and (image_temp[y1, x1 + sx] == 0):
                    image_temp[y1, x1 + sx] = 0
                    line_image[y1, x1 + sx] = 0

            if e2 < dx:
                err += dx
                y1 += sy
            else:
                # 在这里加入垂直方向的步骤，仅设置像素，不改变 image_temp 的值
                if (y1 + sy >= 0) and (y1 + sy < image_temp.shape[0]) and (image_temp[y1 + sy, x1] == 0):
                    image_temp[y1 + sy, x1] = 0
                    line_image[y1 + sy, x1] = 0
        return image_temp, line_image

    @staticmethod
    def bfs_search_noise_error(image):
        height, width = image.shape[:2]
        visited = np.zeros_like(image, dtype=np.uint8)
        paths = []

        def is_valid(x, y):
            return 0 <= x < width and 0 <= y < height and not visited[y, x] and image[y, x] == 0

        def bfs(start_x, start_y):
            path = []
            queue = deque([(start_x, start_y)])

            while queue:
                x, y = queue.popleft()
                path.append((x, y))
                visited[y, x] = 1

                # 八邻域
                neighbors = [(x + dx, y + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if (dx, dy) != (0, 0)]

                for nx, ny in neighbors:
                    if is_valid(nx, ny):
                        queue.append((nx, ny))
                        visited[ny, nx] = 1

            return path

        # 使用连通区域标记
        labeled_image, num_features = label(image == 0)

        for label_num in range(1, num_features + 1):
            positions = np.argwhere(labeled_image == label_num)
            if positions.size > 0:
                start_x, start_y = positions[0]
                path = bfs(start_x, start_y)
                if path:
                    paths.append(path)

        return paths

    @staticmethod
    def bfs_search_noise(image):
        height, width = image.shape[:2]
        visited = np.zeros_like(image, dtype=np.uint8)
        paths = []

        def is_valid(x, y):
            return 0 <= x < width and 0 <= y < height and not visited[y, x] and image[y, x] == 0

        def bfs(start_x, start_y):
            path = []
            queue = deque([(start_x, start_y)])

            while queue:
                x, y = queue.popleft()
                path.append((x, y))
                visited[y, x] = 1

                # 八邻域
                neighbors = [(x + dx, y + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if (dx, dy) != (0, 0)]

                for nx, ny in neighbors:
                    if is_valid(nx, ny):
                        queue.append((nx, ny))
                        visited[ny, nx] = 1

            return path

        for y in range(height):
            for x in range(width):
                if not visited[y, x] and image[y, x] == 0:
                    path = bfs(x, y)
                    if path:
                        paths.append(path)
        return paths

    @staticmethod
    def colorize_pixel(image, x, y, color):
        # 设置指定像素位置的颜色值
        image.itemset((y, x, 0), color[0])  # Blue channel
        image.itemset((y, x, 1), color[1])  # Green channel
        image.itemset((y, x, 2), color[2])  # Red channel

    @staticmethod
    def get_angel_cost(line, image):
        total_cost = 0
        p1 = line[0]
        p2 = line[1]

        p3 = ImageHelper.find_point_in_line(p1, image, 10)
        p4 = ImageHelper.find_point_in_line(p2, image, 10)
        # print('before', p3, p4)
        if not p3:
            p3 = ImageHelper.find_point_in_line(p1, image, 5)
        if not p4:
            p4 = ImageHelper.find_point_in_line(p2, image, 5)
        if not p3:
            p3 = ImageHelper.find_point_in_line(p1, image, 2)
        if not p4:
            p4 = ImageHelper.find_point_in_line(p2, image, 2)
        # print('after', p3, p4)

        angle1 = ImageHelper.calculate_angle(p2[0], p2[1], p1[0], p1[1], p3[0], p3[1])
        angle2 = ImageHelper.calculate_angle(p1[0], p1[1], p2[0], p2[1], p4[0], p4[1])
        # print(angle1, angle2)
        if angle2 == 0 or angle1 == 0:
            angle2 += 1
            angle1 += 1
        total_cost += 180.0 / angle1 + 180.0 / angle2
        return total_cost

    @staticmethod
    def calculate_angle(x1, y1, x2, y2, x3, y3):
        vector_BA = (x1 - x2, y1 - y2)
        vector_BC = (x3 - x2, y3 - y2)

        dot_product = vector_BA[0] * vector_BC[0] + vector_BA[1] * vector_BC[1]
        magnitude_AB = math.sqrt(vector_BA[0] ** 2 + vector_BA[1] ** 2)
        magnitude_BC = math.sqrt(vector_BC[0] ** 2 + vector_BC[1] ** 2)

        # if math.isclose(magnitude_AB * magnitude_BC, 0.0, abs_tol=1e-10):
        #     # 处理除零错误，避免无效的余弦值
        #     return 0.0

        # 使用 min 和 max 来确保 cos_theta 的值在 [-1, 1] 范围内
        cos_theta = min(1.0, max(-1.0, dot_product / (magnitude_AB * magnitude_BC)))

        # if magnitude_AB * magnitude_BC == 0:
        #     # 处理除零错误，避免无效的余弦值
        #     return 0.0

        # cos_theta = dot_product / (magnitude_AB * magnitude_BC)
        angle_in_radians = math.acos(cos_theta)
        angle_in_degrees = math.degrees(angle_in_radians)

        return angle_in_degrees
