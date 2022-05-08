import numpy as np
import rospy

from drone_localization_ros.msg import PosesWithCovariance
from geometry_msgs.msg import PoseWithCovariance



def get_poses_msg(node, dets_msg):
    poses_msg = PosesWithCovariance()
    for det in dets_msg.detections:
        x_mid = det.xmin + (det.xmax - det.xmin)/2.0
        y_mid = det.ymin + (det.ymax - det.ymin)/2.0
        center_vector = np.array(node.camera.projectPixelTo3dRay((x_mid, y_mid)))  # node.camera.rectifyPoint is not needed
        right_vector = np.array(node.camera.projectPixelTo3dRay((det.xmax, y_mid)))

        estimated_distance = node._drone_width / 2.0 * np.linalg.norm(center_vector) \
                / np.linalg.norm(right_vector - center_vector)

        if estimated_distance > 0 and estimated_distance is not np.NaN:
            print("estimated_distance: " + str(estimated_distance))  # TODO
            position_vector = estimated_distance * center_vector
            pose_w_cov_msg = PoseWithCovariance()
            pose_w_cov_msg.pose.position.x = position_vector[0]
            pose_w_cov_msg.pose.position.y = position_vector[1]
            pose_w_cov_msg.pose.position.z = position_vector[2]
            pose_w_cov_msg.pose.orientation.x = 0
            pose_w_cov_msg.pose.orientation.y = 0
            pose_w_cov_msg.pose.orientation.z = 0
            pose_w_cov_msg.pose.orientation.w = 1

            pose_w_cov_msg.covariance = generate_covariance(node, position_vector)
                        
            poses_msg.poses.append(pose_w_cov_msg)
        else:
            rospy.logerr("Invalid distance estimate {}! Skipping detection.".format(estimated_distance))

    return poses_msg 


def rotation_matrix_from_vectors(source, dest):
    a, b = (source / np.linalg.norm(source)).reshape(3), (dest / np.linalg.norm(dest)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    
    return rotation_matrix


def generate_covariance(node, position_vector):
    # calculate covariance
    cov_matrix = np.identity(3)
    cov_matrix[0][0] = cov_matrix[1][1] = node._covariance_xy
    cov_matrix[2][2] = position_vector[2] * np.sqrt(position_vector[2]) * node._covariance_z
    if cov_matrix[2][2] < 0.33 * node._covariance_z:
        cov_matrix[2][2] = 0.33 * node._covariance_z

    # rotation
    rotation_vector = rotation_matrix_from_vectors(np.array([0.0, 0.0, 1.0]), position_vector)
    rotated_cov = rotation_vector * cov_matrix * np.transpose(rotation_vector)
        
    # fill covariance matrix
    cov = [0.0] * 36
    for r in range(6):
        for c in range(6):
            if r < 3 and c < 3:
                cov[r * 6 + c] = rotated_cov[r][c]
            elif r == c:
                cov[r * 6 + c] = 999

    return cov
