import copy
import json
import math
import os
import re
from collections import deque

import cv2
import numpy as np
import pylab
from matplotlib import pyplot as plt

from function.CoNop import CoNop
from function.ImageHelper import ImageHelper

class STMExtract:
    @staticmethod
    def connnect_ctl(img, distance=70, step=10, num=3):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, pruning_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

        before_drawed_image = copy.deepcopy(pruning_img)
        endpoints = ImageHelper.detect_endpoints(pruning_img)
        endpoints = [list(t) for t in endpoints]
        print(endpoints)

        # rgb_image_show = cv2.cvtColor(before_drawed_image, cv2.COLOR_GRAY2RGB)
        # for point in endpoints:
        #     cv2.circle(rgb_image_show, point, 1, (255, 0, 0), -1)
        # plt.imshow(rgb_image_show, cmap='gray')
        # plt.title('connnect image')
        # # plt.axis('off')
        # pylab.show()
        # pylab.pause(0)

        # paths_bfs = ImageHelper.bfs_search_noise(pruning_img)
        count = 0
        while (count < 1):
            match_points_pairs = []
            relative_points_new = copy.deepcopy(endpoints)
            for point in relative_points_new:
                point.append(False)

            for point in relative_points_new:
                if point[0] == 1087 and point[1] == 386:
                    print(1)
                if not point[2]:
                    match_point = ImageHelper.connectCtl_allmap(point, pruning_img, endpoints, distance, step)
                    if match_point != (-1, -1):
                        drawed_image, line_image = ImageHelper.draw_line_on_binary_image_two_step(before_drawed_image,
                                                                                                  point[:2], match_point)
                        # drawed_image, line_image = ImageHelper.draw_spline_polyline(point[:2], match_point, before_drawed_image)
                        result_image = cv2.bitwise_or(line_image, before_drawed_image)

                        # 进行按位与运算,判断是否与原图(注意是与没有连接的原图)相交
                        count_ones = np.prod(result_image.shape) - np.count_nonzero(result_image)  # 统计像素值为0的数量
                        if count_ones <= num:
                            match_points_pairs.append((point[:2], match_point))

                        before_drawed_image = drawed_image
                        point[2] = True
            before_drawed_image = copy.deepcopy(pruning_img)

            # points补充
            # pairs = [((),()),((),()),((),()),((),())]
            # for pair in pairs:
            #     match_points_pairs.append(pair)

            # 找到共用同一个点的线
            result = STMExtract.find_shared_point_pairs(match_points_pairs)
            print(result)
            for pair in result:
                for item in pair:
                    if item in match_points_pairs:
                        match_points_pairs.remove(item)

            for pair in result:
                # share_point = find_shared_point(pair)
                target_index = -1
                num = 0
                new_cost = 99999
                for index_p, item in enumerate(pair):
                    cost = STMExtract.calculater_angle(item, before_drawed_image)
                    if cost < new_cost:
                        target_index = index_p
                        new_cost = cost
                    num += 1
                match_points_pairs.append(pair[target_index])

            unique_pairs = set()
            unique_points = []
            for pair in match_points_pairs:
                sorted_pair = tuple(sorted(map(tuple, pair)))
                if sorted_pair not in unique_pairs:
                    unique_pairs.add(sorted_pair)
                    unique_points.append(pair)

            # 根据匹配的线对画图
            for line in unique_points:
                # drawed_image, line_image = ImageHelper.draw_line_on_binary_image_two_step( before_drawed_image, line[0], line[1])
                drawed_image, line_image = ImageHelper.draw_spline_polyline(line[0], line[1], before_drawed_image)
                if drawed_image is not None:
                    before_drawed_image = drawed_image

            print(match_points_pairs)

            # # 从总的endpoints中删除match_points_pairs
            # endpoints_copy = copy.deepcopy(endpoints)
            # for match_point_c in match_points_pairs:
            #     if match_point_c[0] in endpoints_copy:
            #         endpoints.remove(match_point_c[0])
            #     if match_point_c[1] in endpoints_copy:
            #         endpoints.remove(match_point_c[1])
            #
            # # pruning_img = before_drawed_image
            #
            # # 一定范围内寻找点
            # conop_points = []
            # endpoints_find = copy.deepcopy(endpoints)
            # for point in endpoints:
            #     point.append(False)
            # for point in endpoints:
            #     if not point[2]:
            #         match_points = ImageHelper.find_match_points_conop(point, pruning_img, endpoints_find, distance, 30)
            #         if match_points != (-1, -1) and len(match_points) != 0:
            #             point[2] = True
            #             conop_points.append(tuple(point[:2]))
            # print(conop_points)

            # rgb_image_show = cv2.cvtColor(before_drawed_image, cv2.COLOR_GRAY2RGB)
            # for point in conop_points:
            #     cv2.circle(rgb_image_show, point, 1, (255, 0, 0), -1)
            # plt.imshow(rgb_image_show, cmap='gray')
            # # plt.axis('off')
            # plt.title('conop points image')
            # # plt.show()
            # pylab.show()
            # pylab.pause(0)

            count += 1

            # if len(conop_points) != 0:
            #     coNop = CoNop(conop_points, before_drawed_image)
            #     best = coNop.simulated_annealing()
            # else:
            #     best = before_drawed_image
            # 得到一个对比图
            result_image = STMExtract.compare_and_show(pruning_img, before_drawed_image)
            # result_image = STMExtract.compare_and_show(pruning_img, best)
            return result_image

    @staticmethod
    def find_shared_point_pairs(point_pairs):
        shared_point_pairs = []

        for i in range(len(point_pairs)):
            for j in range(i + 1, len(point_pairs)):
                pair1 = point_pairs[i]
                pair2 = point_pairs[j]

                if any(point in pair2 for point in pair1):
                    shared_point_pairs.append((pair1, pair2))

        return shared_point_pairs

    @staticmethod
    def calculater_angle(line, image):
        p1 = line[0]
        p2 = line[1]

        p3 = STMExtract.find_point_in_line(p1, image, 3)
        p4 = STMExtract.find_point_in_line(p2, image, 3)
        # print('before', p3, p4)
        if not p3:
            p3 = STMExtract.find_point_in_line(p1, image, 5)
        if not p4:
            p4 = STMExtract.find_point_in_line(p2, image, 5)
        if not p3:
            p3 = STMExtract.find_point_in_line(p1, image, 2)
        if not p4:
            p4 = STMExtract.find_point_in_line(p2, image, 2)
        # print('after', p3, p4)

        angle1 = STMExtract.calculate_angle(p2[0], p2[1], p1[0], p1[1], p3[0], p3[1])
        angle2 = STMExtract.calculate_angle(p1[0], p1[1], p2[0], p2[1], p4[0], p4[1])
        # print(angle1, angle2)
        if angle2 == 0 or angle1 == 0:
            angle2 += 1
            angle1 += 1
        total_cost = 180.0 / angle1 + 180.0 / angle2
        return total_cost

    @staticmethod
    def find_point_in_line(point, image, iteration):
        path = STMExtract.find_path_with_step_limit(image, point, iteration)
        if path is None:
            # print('未找到目标点')
            return
        return path[-2]

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

    @staticmethod
    def read_and_binarize_image(image):
        _, binary_image = cv2.threshold(image, 170, 255, cv2.THRESH_BINARY)
        src_img = np.where(binary_image == 0, 0, binary_image)
        src_img = np.where(src_img == 255, 1, src_img)
        return src_img

    @staticmethod
    def compare_and_show(img1, img2):
        image1 = STMExtract.read_and_binarize_image(img1)
        image2 = STMExtract.read_and_binarize_image(img2)
        differences = np.logical_xor(image1, image2)
        # differences = np.where(differences == 1, 255, differences)
        # differences = np.where(differences == 0, 0, differences)

        height, width = img1.shape[:2]
        differences_open = np.zeros((height, width), dtype=np.uint8)
        differences_open[differences] = 255

        # 定义一个3x3的结构元素
        # kernel = np.ones((5, 5), np.uint8)
        # differences_open = cv2.dilate(differences_open, kernel, iterations=1)

        output_image = np.ones((image1.shape[0], image1.shape[1], 3), dtype=np.uint8) * 255
        output_image[(image1 == image2) & (image1 == 0)] = [0, 0, 0]
        output_image[np.where(differences_open == 255)] = [0, 0, 255]
        return output_image