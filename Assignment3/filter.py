#!/usr/bin/env python
from scipy import ndimage as ndi

import rospy
import ros_numpy as ros_np
import numpy as np
from sensor_msgs.msg import PointCloud2


def grids(point_cloud, step_size):
    x = point_cloud['x'].copy()
    y = point_cloud['y'].copy()

    x_ri = np.min(x)
    x_le = np.max(x)
    y_ri = np.min(y)
    y_le = np.max(y)

    x_length = int((x_le - x_ri) / step_size) + 1
    y_length = int((y_le - y_ri) / step_size) + 1

    sum_2d = np.zeros((x_length, y_length))

    for x_i in range(x_length):
        x_coord = x_ri + step_size * x_i
        for y_i in range(y_length):
            y_coord = y_ri + step_size * y_i

            sum_2d[x_i, y_i] = np.sum((x_coord <= x) & (x < x_coord + step_size) & (y_coord <= y) & (y < y_coord + step_size))

    return sum_2d


class dd_filter:
    def __init__(self):
        self.pc_sub = rospy.Subscriber('/cloud_map', PointCloud2, self.callback)
        self.pc_pub = rospy.Publisher('/filtered_pc', PointCloud2, queue_size=50)
        self.old_length = 0
        self.msg_counter = 0

    def callback(self, data):
        pc_raw = ros_np.point_cloud2.pointcloud2_to_array(data)
        pc_filtered_arr = apply_filter(pc_raw)

        pc_rec_arr = np.array(pc_filtered_arr,
                              dtype=[('x', np.float32),
                                     ('y', np.float32),
                                     ('z', np.float32),
                                     ('rgb', np.float32)]
                              )

        pc_filtered = ros_np.point_cloud2.array_to_pointcloud2(pc_rec_arr, stamp=data.header.stamp, frame_id='map')

        self.pc_pub.publish(pc_filtered)


def apply_filter(raw_pc):
    filtered_pc = raw_pc[0]

    shigma = 0.125
    step_size = 0.1
    grid_2d = grids(raw_pc, step_size)
    filt_img = ndi.gaussian_filter(grid_2d, sigma=sigma)
    mean = 390
    ind_zero = np.where(filt_img <= mean)
    filt_img[ind_zero] = 0
    ind_nonzero = np.nonzero(filt_img)
    filt_img[ind_nonzero] = 1


    x = raw_pc['x'].copy()
    y = raw_pc['y'].copy()

    x_min = np.min(x)
    y_min = np.min(y)

    for x_i in range(np.shape(filt_img)[0]):
        x_coord = x_min + step_size * x_i
        for y_i in range(np.shape(filt_img)[1]):
            if filt_img[x_i, y_i] != 1.0:
                continue
        y_coord = y_min + step_size * y_i
        filtered_pc = np.append(filtered_pc, raw_pc[(x_coord <= x) & (x < x_coord + step_size) & (y_coord <= y) & (y < y_coord + step_size)])

    return filtered_pc, raw_pc




def Main():
    rospy.init_node('dd_filter', anonymous=True)
    pcf = dd_filter()
    print('start:')


if __name__ == '__main__':
    Main()