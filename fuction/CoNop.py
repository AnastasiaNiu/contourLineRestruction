import copy
import math
import random
from collections import Counter

import cv2
import matplotlib.pyplot
import numpy as np
import pylab
from matplotlib import pyplot as plt

from shapely.geometry import LineString, Point


from function.ImageHelper import ImageHelper


class CoNop:
    def __init__(self, points, image):
        self.__points = points
        self.__image = image

    def count_point_occurrences(self, lines):
        # 将所有点的坐标展开成一个列表
        all_points = [point for line in lines for point in line]

        # 使用Counter计算每个点的出现次数
        point_counts = Counter(all_points)

        # 统计出现次数为2的点的数量
        twice_used_points = sum(count == 2 for count in point_counts.values())

        # 返回结果
        return twice_used_points

    def calculate_angle(self, x1, y1, x2, y2, x3, y3):
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

    def find_path_with_step_limit(self, binary_image, start, step_limit):
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

    def find_point_in_line(self, point, image, iteration):
        path = self.find_path_with_step_limit(image, point, iteration)
        if path is None:
            # print('未找到目标点')
            return
        return path[-2]

    def bitwiseAnd(self, image1, image2):
        height, width = image1.shape
        result_img = copy.deepcopy(image1)
        # plt.imshow(result_img, cmap='gray')  # 注意颜色通道的转换
        # plt.title('result_img')
        # pylab.show()
        # pylab.pause(0)

        result_img[result_img == 0] = 1
        # plt.imshow(result_img, cmap='gray')  # 注意颜色通道的转换
        # plt.title('result_img2')
        # pylab.show()
        # pylab.pause(0)

        for y in range(height):
            for x in range(width):
                if image2[y, x] == 0 and image1[y, x] == 0:
                    result_img[y, x] = 0
                    # plt.imshow(result_img, cmap='gray')  # 注意颜色通道的转换
                    # plt.title('Cropped Image')
                    # pylab.show()
                    # pylab.pause(0)
        return result_img

    # 计算代价
    def draw_line_on_binary_image_two_step(self, image, point1, point2):
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

    # 计算代价
    def draw_line_on_binary_image(self, image, point1, point2):
        x1, y1 = map(int, point1)
        x2, y2 = map(int, point2)
        image_temp = copy.deepcopy(image)
        height, width = image.shape
        line_image = np.ones((height, width), dtype=np.uint8)
        # plt.imshow(line_image, cmap='gray')  # 注意颜色通道的转换
        # plt.title('line_image Image')
        # pylab.show()
        # pylab.pause(0)

        # 使用 Bresenham 算法计算线段上的像素点
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            image_temp[y1, x1] = 0
            line_image[y1, x1] = 0
            # lines_image[y1, x1] = 1
            if x1 == x2 and y1 == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx

            if e2 < dx:
                err += dx
                y1 += sy

        return image_temp, line_image

    def calculate_total_cost(self, lines, image):
        total_cost = 0
        cross_weight = 0.7
        angle_weight = 0.3
        before_drawed_image = copy.deepcopy(image)
        # print('cost', lines)
        # 连线相交或连线与原图线相交

        # print(lines)
        for line in lines:
            # fig, axes = plt.subplots(1, 4, figsize=(15, 5))

            # drawed_image, line_image = self.draw_line_on_binary_image_two_step(before_drawed_image, line[0], line[1])
            # drawed_image, line_image = Helpers.ImageHelper.draw_spline_polyline_ex_2(line[0], line[1], before_drawed_image)
            drawed_image, line_image = ImageHelper.draw_spline_polyline(line[0], line[1], before_drawed_image)
            # drawed_image, line_image = ImageHelper.draw_spline_polyline(before_drawed_image, line[0], line[1])

            # axes[0].imshow(drawed_image, cmap='gray')
            # axes[0].set_title('drawed_image')
            # axes[1].imshow(before_drawed_image, cmap='gray')  # 注意颜色通道的转换
            # axes[1].set_title('before_drawed_image')
            # axes[2].imshow(line_image, cmap='gray')  # 注意颜色通道的转换
            # axes[2].set_title('line_image')

            # result_image = self.bitwiseAnd(line_image, before_drawed_image)  # 进行按位与运算
            result_image = cv2.bitwise_or(line_image, before_drawed_image)  # 进行按位与运算

            # 代价1：相交
            count_ones = np.prod(result_image.shape) - np.count_nonzero(result_image)  # 统计像素值为0的数量
            # print(count_ones)
            # axes[3].imshow(result_image, cmap='gray')  # 注意颜色通道的转换
            # axes[3].set_title('result_image')
            # pylab.show()
            # pylab.pause(0)
            if 7 >= count_ones >= 1:
                total_cost -= 9999
            if 11 >= count_ones > 7:
                total_cost -= 9999
            if count_ones > 11:
                total_cost -= 9999

            # if 7 >= count_ones >= 3:
            #     total_cost -= float('inf')
            # if 11 >= count_ones > 7:
            #     total_cost -= float('inf')
            # if count_ones > 11:
            #     total_cost -= float('inf')

            # 代价2：角度
            p1 = line[0]
            p2 = line[1]

            p3 = self.find_point_in_line(p1, image, 10)
            p4 = self.find_point_in_line(p2, image, 10)
            # print('before', p3, p4)
            if not p3:
                p3 = self.find_point_in_line(p1, image, 5)
            if not p4:
                p4 = self.find_point_in_line(p2, image, 5)
            if not p3:
                p3 = self.find_point_in_line(p1, image, 2)
            if not p4:
                p4 = self.find_point_in_line(p2, image, 2)
            # print('after', p3, p4)

            angle1 = self.calculate_angle(p2[0], p2[1], p1[0], p1[1], p3[0], p3[1])
            angle2 = self.calculate_angle(p1[0], p1[1], p2[0], p2[1], p4[0], p4[1])
            # print(angle1, angle2)
            if angle2 == 0 or angle1 == 0:
                angle2 += 1
                angle1 += 1
            total_cost -= 180.0 / angle1 + 180.0 / angle2
            before_drawed_image = drawed_image

        # 代价3：一个point不能使用两次
        # temp = self.count_point_occurrences(lines)
        # if temp == 0:
        #     total_cost -= 10
        # else:
        #     total_cost += temp

        return total_cost, drawed_image

    def distance(self, point1, point2):
        """计算两点之间的欧氏距离"""
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def find_nearest_points(self, target_point, all_points, k=6):  # 10
        """找到距离目标点最近的 k 个点"""
        distances = [(point, self.distance(target_point, point)) for point in all_points]

        # 将距离为 0 的点（即目标点自身）排除在外
        distances = [(point, distance) for point, distance in distances if distance > 0]

        if not distances:
            return []  # 没有找到最近的点，返回空列表

        sorted_distances = sorted(distances, key=lambda x: x[1])
        nearest_points = [point[0] for point in sorted_distances[:k]]
        return nearest_points

    # 进行随机扰动
    def __get_neighbor(self):
        lines = []
        processed_points = set()
        for point in self.__points:
            # 检查点是否已经处理过
            if point in processed_points:
                continue

            nearest_points = self.find_nearest_points(point, self.__points)
            # 从最近点中remove已处理点
            for point1 in processed_points:
                if point1 in nearest_points:
                    nearest_points.remove(point1)

            if len(nearest_points) == 0:
                continue
                # return lines

            point1 = random.choice(nearest_points)
            # 检查新线是否已经存在
            new_line = [point, point1, False]
            reversed_line = (point1, point)
            if new_line not in lines and reversed_line not in lines:
                lines.append(new_line)
                # 将点标记为已处理
                processed_points.add(point1)
                processed_points.add(point)
        return lines

    def new_lines(self, lines):
        if len(self.__points) <= 2:
            return

        # 随机选择两个元组（两条线）
        selected_lines = random.sample(lines, 1)

        while selected_lines[0][2]:
            selected_lines = random.sample(lines, 1)

        # 从每条选中的线中提取点
        (x1, y1), (x2, y2), _ = selected_lines[0]
        random_point = random.sample([(x1, y1), (x2, y2)], 1)
        (x3, y3) = random_point[0]

        if (x3, y3) == (x2, y2):
            (x2, y2), (x1, y1) = (x1, y1), (x2, y2)

        nearest_points = self.find_nearest_points((x1, y1), self.__points)
        if (x2, y2) in nearest_points:
            nearest_points.remove((x2, y2))

        point1 = random.choice(nearest_points)
        new_line = [(x1, y1), point1, False]

        lines.remove(selected_lines[0])
        lines.append(new_line)

    def delete_repate_line_improve(self, lines):
        unique_lines = {}  # 用于存储唯一线条的字典
        for line in lines:
            p1, p2, _ = line
            # 保证无论方向如何，线条的标识是一致的
            line_key = tuple(sorted([p1, p2]))
            distance = self.distance(p1, p2)
            # 如果这条线是新的，或者更长，则保存它
            if line_key not in unique_lines or unique_lines[line_key][2] < distance:
                unique_lines[line_key] = line[:2] + [distance]

        # 从 unique_lines 中提取最终的线条列表
        final_lines = [line[:2] for line in unique_lines.values()]

        # 更新 self.__points
        # 注意：这里假设 self.__points 已经包含了所有需要的点
        # 如果需要调整点集合，可以在这里进行

        return final_lines  # 返回去重并保留更优连线后的线条列表


    def delete_cross_line(self, lines):
        points = {}

        for i in range(0, len(lines)):
            line = lines[i]

            (x1, y1), (x2, y2), _ = line

            if (x1, y1) in points or (x2, y2) in points:
                if (x1, y1) in points:
                    pre_line = lines[points[(x1, y1)]]
                else:
                    pre_line = lines[points[(x2, y2)]]

                dis1 = self.distance(pre_line[0], pre_line[1])
                dis2 = self.distance(line[0], line[1])

                rx = x1
                ry = y1
                if (x2, y2) in points:
                    rx = x2
                    ry = y2

                if dis1 < dis2:
                    lines[points[(rx, ry)]][2] = True
                    lines[i][2] = False
                else:
                    lines[points[(rx, ry)]][2] = False
                    lines[i][2] = True
            else:
                lines[i][2] = True
                points[(x1, y1)] = i
                points[(x2, y2)] = i

        for line in lines:
            if line[2]:
                if line[0] in self.__points:
                    self.__points.remove(line[0])
                if line[1] in self.__points:
                    self.__points.remove(line[1])

        count = 0
        # for i in range(len(lines)):
        #     line = lines[i]
        #     if not line[2]:
        #         lines[i][0] = self.__points[count]
        #         count += 1
        #         lines[i][1] = self.__points[count]
        #         count += 1
        for i in range(len(lines)):
            line = lines[i]
            if not line[2]:
                if count + 1 < len(self.__points):  # 确保有足够的点可以分配
                    lines[i][0] = self.__points[count]
                    count += 1
                    lines[i][1] = self.__points[count]
                    count += 1
                else:
                    print("Error: Not enough points available to replace line endpoints")
                    break  # 如果点不够用，提前退出循环

    def delete_repate_line(self, lines):
        points = {}

        for i in range(0, len(lines)):
            line = lines[i]

            (x1, y1), (x2, y2), _ = line

            if (x1, y1) in points or (x2, y2) in points:
                if (x1, y1) in points:
                    pre_line = lines[points[(x1, y1)]]
                else:
                    pre_line = lines[points[(x2, y2)]]

                dis1 = self.distance(pre_line[0], pre_line[1])
                dis2 = self.distance(line[0], line[1])

                rx = x1
                ry = y1
                if (x2, y2) in points:
                    rx = x2
                    ry = y2

                if dis1 < dis2:
                    lines[points[(rx, ry)]][2] = True
                    lines[i][2] = False
                else:
                    lines[points[(rx, ry)]][2] = False
                    lines[i][2] = True
            else:
                lines[i][2] = True
                points[(x1, y1)] = i
                points[(x2, y2)] = i

        for line in lines:
            if line[2]:
                if line[0] in self.__points:
                    self.__points.remove(line[0])
                if line[1] in self.__points:
                    self.__points.remove(line[1])

        count = 0
        # for i in range(len(lines)):
        #     line = lines[i]
        #     if not line[2]:
        #         lines[i][0] = self.__points[count]
        #         count += 1
        #         lines[i][1] = self.__points[count]
        #         count += 1
        for i in range(len(lines)):
            line = lines[i]
            if not line[2]:
                if count + 1 < len(self.__points):  # 确保有足够的点可以分配
                    lines[i][0] = self.__points[count]
                    count += 1
                    lines[i][1] = self.__points[count]
                    count += 1
                else:
                    print("Error: Not enough points available to replace line endpoints")
                    break  # 如果点不够用，提前退出循环

    # 模拟退火
    def simulated_annealing(self, temperature=100, cooling_rate=0.5, stop_temp=20):
        lines = self.__get_neighbor()
        print(lines)
        best_cost, _ = self.calculate_total_cost(lines, self.__image)  # 最佳代价
        current_temp = temperature
        best_costs = [best_cost]
        best = copy.deepcopy(_)
        temp_solution = copy.deepcopy(self.__image)
        i = 0
        # 关闭交互模式，保存图形，关闭图形窗口
        plt.imshow(best, cmap='gray')  # 注意颜色通道的转换
        # plt.title('best Temp Image')
        plt.ioff()
        plt.axis('off')
        plt.savefig('E:\\round\\res_' + str(i) + '.png',bbox_inches='tight')
        plt.close()

        while current_temp > stop_temp:
            count = 0
            i += 1
            while True:
                # 随机扰动(在前一轮基础上，随机改变两个点进行连接)
                prelines = copy.deepcopy(lines)

                self.new_lines(lines)
                # print(lines)

                # lines = self.__get_neighbor(self.__points, self.__image)

                # 计算新代价(代价1：等高线相交 || 代价2：连接的线的方位角偏差)
                new_cost, res = self.calculate_total_cost(lines, temp_solution)

                if new_cost > best_cost:
                    best = copy.deepcopy(res)
                    # temp_solution = res
                    best_cost = new_cost
                    count = 0

                    plt.imshow(best, cmap='gray')  # 注意颜色通道的转换
                    # plt.title('best Temp Image')
                    plt.axis('off')
                    plt.ioff()
                    plt.savefig('E:\\round\\res_' + str(i) + '.png', bbox_inches='tight')
                    plt.close()
                    i += 1
                else:
                    # temp_solution = res
                    lines = prelines
                    count += 1

                # 在一定序列内稳定
                if count == 400:
                    break
                # print(new_cost)

            # 删除重复点的线
            print('delete_repeat', best_cost)
            self.delete_repate_line(lines)

            # 删除交叉的线
            # self.delete_cross_line(lines)

            # print(lines)
            new_cost, res = self.calculate_total_cost(lines, temp_solution)
            plt.imshow(res, cmap='gray')  # 注意颜色通道的转换
            # plt.title('best Temp Image')
            plt.ioff()
            plt.axis('off')
            plt.savefig('E:\\round\\res_' + str(i) + '.png', bbox_inches='tight')
            plt.close()

            # if new_cost > best_cost:
            best = copy.deepcopy(res)
            best_cost = new_cost

            current_temp *= cooling_rate
            best_costs.append(best_cost)
            # matplotlib.pyplot.plot(best_costs)
            # matplotlib.pyplot.show()
            # matplotlib.pyplot.cla()
            print("current temp:" + str(current_temp) + ",current best cost:" + str(best_cost))

        # plt.imshow(best, cmap='gray')  # 注意颜色通道的转换
        # plt.title('best Image')
        # pylab.show()
        # matplotlib.pyplot.plot(best)
        # matplotlib.pyplot.title("Simulated annealing")
        # matplotlib.pyplot.savefig(r'G:/AAAcontourLines/模拟退火结果.jpg')
        return best
