#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from interop.msg import Object, ObjectType, Orientation, Color, Shape
from interop.srv import AddObject, SetObjectCompressedImage, UpdateObject

import tf2_ros
import tf.transformations
import image_geometry

import cv2
from cv_bridge import CvBridge, CvBridgeError

from cv.msg import target_crop
from fast_detection import DANN
import tensorflow as tf
import numpy as np
import time
import os


def detect_targets():

    path_ckpt = '/home/elsa/Robotics/drone/catkin_ws/src/cv/fast_detection_ckpts/fast_detection.ckpt-98'
    rospy.init_node('detect_targets')
    dann = DANN()
    dann.build_predict()
    saver = tf.train.Saver()
    saver.restore(dann.sess, path_ckpt)
    print('Done')
    # image publisher
    crop_pub = rospy.Publisher('/image_crops', target_crop, queue_size=3)
    camera_info = rospy.wait_for_message("/camera/camera_info", CameraInfo)
    camera_model = image_geometry.PinholeCameraModel()
    camera_model.fromCameraInfo(camera_info)

    # Instantiate CvBridge
    bridge = CvBridge()
    epsilon = 1e-6

    def image_callback(msg):
        # get timestamp
        time_sec = msg.header.stamp.secs
        time_nsec = msg.header.stamp.nsecs
        print(time_sec, time_nsec)

        try:
            # Convert your ROS Image message to OpenCV2
            img = np.fromstring(msg.data, np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (img.shape[1] / 4, img.shape[0] / 4))

            output = dann.predict(img / 255)
            output = cv2.resize(output[0, :, :, 0],
                                (img.shape[1], img.shape[0]), cv2.INTER_NEAREST)
            map = np.uint8(np.round(output))
            map, cnt, heir = cv2.findContours(map, cv2.RETR_LIST,
                                              cv2.CHAIN_APPROX_SIMPLE)

            # for each detected contour of the output map
            for c in cnt:
                # print("contour found")
                # compute the center as the mean coordinate from the list of points that make up the contour
                center = np.int16(np.mean(c, axis=0).flatten())

                # find a bounding rectangle around the contour
                x, y, w, h = cv2.boundingRect(c)

                if w > 100 or w < 22 or h > 100 or h < 22:
                    # not a real target
                    # print("wrong size")
                    continue

                # allow some leaway to make sure to get the whole target,
                # +- 11 pixels based on receptive field of detector
                lower_x = -min(0, -(x - 11))
                higher_x = min(x + w + 11, img.shape[1] - 1)
                lower_y = -min(0, -(y - 11))
                higher_y = min(y + h + 11, img.shape[0] - 1)

                # crop the resized image, encode to jpg and publish
                crop = img[lower_y:higher_y, lower_x:higher_x, :]
                crop_str = cv2.imencode('.jpg', crop)[1].tostring()
                compImg = CompressedImage(msg.header, 'jpg', crop_str)

                # publish just to test
                print("target found!")
                crop_pub.publish(msg.header, compImg, center[0], center[1])

        except CvBridgeError, e:
            print(e)

    # subscribe to topic /camera_wide/image_color
    image_sub = rospy.Subscriber('/camera/image_color/compressed',
                                 CompressedImage, image_callback)

    # Once the listener is created, it starts receiving tf2 transformations over
    # the wire, and buffers them for up to 10 seconds.
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    rospy.spin()


if __name__ == "__main__":
    detect_targets()
