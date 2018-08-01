#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
import numpy as np
from cv.msg import target_crop

import tf2_ros
import tf.transformations
import image_geometry


def target_locator():

    rospy.init_node('target_locator')
    loc_pub = rospy.Publisher('/target_locations', target_crop, queue_size=3)
    camera_info = rospy.wait_for_message("/camera/camera_info", CameraInfo)
    camera_model = image_geometry.PinholeCameraModel()
    camera_model.fromCameraInfo(camera_info)

    # Once the listener is created, it starts receiving tf2 transformations over
    # the wire, and buffers them for up to 10 seconds.
    tfBuffer = tf2_ros.Buffer(cache_time=rospy.Duration(30))
    listener = tf2_ros.TransformListener(tfBuffer)
    epsilon = 1e-6

    def quaternion_vector_multiply(q, v):
        """
        multiplies quaternion q with vector v
        :param q:
        :param v:
        :return:
        """
        qv = np.zeros(4)
        qv[:3] = tf.transformations.unit_vector(v)

        return tf.transformations.quaternion_multiply(
            tf.transformations.quaternion_multiply(q, qv),
            tf.transformations.quaternion_conjugate(q))[:3]

    def target_callback(msg):

        # get timestamp
        time_sec = msg.header.stamp.secs
        time_nsec = msg.header.stamp.nsecs
        print(time_sec, time_nsec)

        try:
            # lookup transform at the time of capture
            tf_stamped = tfBuffer.lookup_transform('ground', 'camera',
                                                   msg.header.stamp)

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException), e:
            print("unable to get transform")
            rospy.logwarn_throttle(5.0,
                                   "Waiting for ground to camera transform")
            print(str(e))
            return

        camera_position = np.array([
            tf_stamped.transform.translation.x,
            tf_stamped.transform.translation.y,
            tf_stamped.transform.translation.z,
        ])

        camera_orientation = np.array([
            tf_stamped.transform.rotation.x,
            tf_stamped.transform.rotation.y,
            tf_stamped.transform.rotation.z,
            tf_stamped.transform.rotation.w,
        ])

        print("got transform")
        # convert the center pixel of the target to a unit vector wrt the image center
        ray = camera_model.projectPixelTo3dRay([msg.x, msg.y])

        # compute endpoints of the ray, project the ray onto the camera's frame
        ray_source = camera_position
        ray_direction = quaternion_vector_multiply(camera_orientation, ray)

        print(ray_source, ray_direction)
        # We need to ignore all rays pointing up or coming from under ground.
        if ray_direction[2] > -epsilon or ray_source[2] < epsilon:
            print("No valid intersection found with the ground plane")
            raise ValueError(
                "No valid intersection found with the ground plane")

        # parameterize the line defined by the ray s.t. |z-component|=1
        t = ray_source[2] / ray_direction[2]
        # intersection with ground occurs when z-component = 0
        intersection = ray_source - ray_direction * t

        # publish ground coordinates
        print("publishing")
        loc_pub.publish(msg.header, msg.image, intersection[0], intersection[1])

    target_sub = rospy.Subscriber("/image_crops", target_crop, target_callback)

    rospy.spin()


if __name__ == "__main__":

    target_locator()
