#!/usr/bin/env python
import rospy
import tf.transformations
from pyproj import Proj, transform
from transformers.points import ENUCoord
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import Point
from interop.msg import Object, ObjectType, Orientation, Color, Shape
from interop.srv import AddObject, AddObjectRequest
from interop.srv import SetObjectCompressedImage, SetObjectCompressedImageRequest
from interop.srv import UpdateObject, UpdateObjectRequest
import numpy as np
from cv.msg import target_crop


def target_clustering():

    # keep a finite record of targets
    max_candidates = 10
    max_per_candidate = 5

    print("waiting for service")
    rospy.wait_for_service("interop/objects/add")
    rospy.wait_for_service("interop/objects/image/compressed/set")

    print("services ready")
    add_object = rospy.ServiceProxy('interop/objects/add', AddObject)
    set_img = rospy.ServiceProxy('interop/objects/image/compressed/set',
                                 SetObjectCompressedImage)

    candidate_arr = {}
    sent_arr = []

    # clustering parameters
    r = 10

    def target_callback(msg):

        # receive a msg with a header, compressed image and ground coordinates
        new_coord = np.array([msg.x, msg.y])
        print(new_coord)
        isadded = False

        keys = list(candidate_arr.keys())

        if len(keys) >= max_candidates:
            # find the dictionary entry with the most images
            max_count = 0
            max_key = 0
            for i in keys:
                count = candidate_arr[i]['count']
                if count > max_count:
                    max_key = i
                    max_count = count

            mean = max_key
            dict_entry = candidate_arr[mean]

            # find the image closest to the mean
            mean = np.array(mean)
            diff = np.array(dict_entry['pos']) - mean[np.newaxis, :]
            dist = map(np.linalg.norm, diff)
            msg_idx = np.argmin(dist)

            compImg = dict_entry['image'][msg_idx]

            # create an Object msg
            lat, lon = to_latlon(mean[0], mean[1])
            object = Object(
                type=ObjectType("standard"),
                latitude=lat,
                longitude=lon,
                orientation=Orientation("n"),
                shape=Shape("circle"),
                background_color=Color("purple"),
                alphanumeric_color=Color("blue"),
                alphanumeric=("Z"),
                autonomous=True)

            add_obj_req = AddObjectRequest()
            add_obj_req.object = object

            # use the service to add the object
            try:
                print("adding object")
                response = add_object(add_obj_req)
                print(response.success)
                if response.success:
                    # add the object's image
                    set_img_req = SetObjectCompressedImageRequest(
                        id=response.id, image=compImg)
                    response = set_img(set_img_req)
                    print(response.success)
                    if response.success:
                        candidate_arr.pop((mean[0], mean[1]))

            except rospy.ServiceException:
                return

        for coord in keys:
            if np.linalg.norm(coord - new_coord) <= r:

                dict_entry = candidate_arr[coord]
                dict_entry['header'].append(msg.header)
                dict_entry['image'].append(msg.image)
                dict_entry['pos'].append(new_coord)
                dict_entry['count'] += 1

                # compute the new mean for that cluster
                mean = np.mean(dict_entry['pos'], axis=0)
                candidate_arr[(mean[0], mean[1])] = candidate_arr.pop(coord)

                if dict_entry['count'] >= max_per_candidate:
                    # send the entry
                    # create an Object msg, position has type Point
                    lat, lon = to_latlon(mean[0], mean[1])
                    object = Object(
                        type=ObjectType("standard"),
                        latitude=lat,
                        longitude=lon,
                        orientation=Orientation("n"),
                        shape=Shape("circle"),
                        background_color=Color("purple"),
                        alphanumeric_color=Color("blue"),
                        alphanumeric=("Z"),
                        autonomous=True)

                    # find the image closest to the mean
                    diff = np.array(dict_entry['pos']) - mean[np.newaxis, :]
                    dist = map(np.linalg.norm, diff)
                    msg_idx = np.argmin(dist)

                    compImg = dict_entry['image'][msg_idx]

                    # use the service to add the object
                    print("adding object")
                    add_obj_req = AddObjectRequest(object)
                    response = add_object(add_obj_req)
                    print(response.success)
                    if response.success:
                        # add the object's image
                        set_img_req = SetObjectCompressedImageRequest(
                            response.id, compImg)
                        response = set_img(set_img_req)
                        print(response.success)
                        if response.success:
                            candidate_arr.pop((mean[0], mean[1]))

                isadded = True
                break

        if not isadded:
            candidate_arr[(new_coord[0], new_coord[1])] = {
                'header': [msg.header],
                'image': [msg.image],
                'pos': [new_coord],
                'count': 1
            }

    loc_sub = rospy.Subscriber("/target_locations", target_crop,
                               target_callback)
    rospy.spin()


def to_latlon(x, y):
    inv_rot = tf.transformations.quaternion_inverse(ref_point.q)
    tmp = tf.transformations.quaternion_multiply(inv_rot, [x, y, 0.0, 0.0])
    tr = tf.transformations.quaternion_multiply(
        tmp, tf.transformations.quaternion_conjugate(inv_rot))[:3]

    translated = [tr[0] + ref_point.x, tr[1] + ref_point.y, tr[2] + ref_point.z]

    lla = Proj(proj="latlon", ellps="WGS84", datum="WGS84")
    ecef = Proj(proj="geocent", ellps="WGS84", datum="WGS84")

    lon, lat, _ = transform(
        ecef, lla, translated[0], translated[1], translated[2], radians=False)

    return lat, lon


if __name__ == "__main__":
    rospy.init_node('cluster_targets')
    ref_point = ENUCoord.ref_point_from_map_transform()
    target_clustering()
