#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas
import cv2 

from deep_sort.detection import Detection
from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from deep_sort import generate_detections
from deep_sort import preprocessing as prep

import rospy
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
import tf

class TrackerPeople():

    def __init__(self):
        rospy.loginfo("Start PeopleTracker Init process...")
        # get an instance of RosPack with the default search paths

        self.rate = rospy.Rate(5)

        self.bridge_object = CvBridge()
        rospy.loginfo("Start camera suscriber...")
        self.topic = "/camera/rgb/image_raw"
        self.point_cloud_topic = "/camera/depth/points"
        self._check_cam_ready()
        self.image_sub = rospy.Subscriber(self.topic,Image,self.camera_callback)
        if self.point_cloud_topic is not None:
            rospy.Subscriber(self.point_cloud_topic, PointCloud2, self.pc_callback)
        else:
            rospy.loginfo(
            'No point cloud information available. Objects will not be placed in the scene.')

        rospy.loginfo("Finished PeopleTracker Init process...Ready")
        self._tf_listener = tf.TransformListener()
        self._tfpub = tf.TransformBroadcaster()

        self._current_pc = None


    def _check_cam_ready(self):
      self.cam_image = None
      while self.cam_image is None and not rospy.is_shutdown():
         try:
               self.cam_image = rospy.wait_for_message(self.topic, Image, timeout=1.0)
               rospy.logdebug("Current "+self.topic+" READY=>" + str(self.cam_image))

         except:
               rospy.logerr("Current "+self.topic+" not ready yet, retrying.")


    def camera_callback(self,data):
        self.cam_image = data
    
    def pc_callback(self, pc):
        self._current_pc = pc
    
    def loop(self):

        while not rospy.is_shutdown():
            self.tracker(self.cam_image)
            self.rate.sleep()
        

    def tracker(self,data):

        # Get a reference to webcam #0 (the default one)
        try:
            # We select bgr8 because its the OpneCV encoding by default
            cv_image = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)

        model = torch.hub.load('/home/hera/catkin_hera/src/3rdParty/vision_system/tracker/src/yolov5', 'custom', path='/home/hera/catkin_hera/src/3rdParty/vision_system/tracker/src/yolov5/weights/yolov5s.pt', source='local')
        model.eval()

        # Inference
        results = model(cv_image, size=640)

        results.xyxy[0] 

        deteccao = results.pandas().xyxy[0]

        detect = []
        scores = []

        for i in range(len(deteccao)):
            x_center = round((deteccao['xmin'][i] + deteccao['xmax'][i])/2)
            y_center = round((deteccao['ymin'][i] + deteccao['ymax'][i])/2)
            pc_list = list(pc2.read_points(self._current_pc, skip_nans=True, field_names=('x', 'y', 'z'), uvs=[(x_center, y_center)]))

        #if len(pc_list) > 0:
            #pc = pc_list[0]
            #self._tfpub.sendTransform((pc[0], pc[1], pc[2]), (0, 0, 0, 1), rospy.Time.now(), "person", "camera_rgb_optical_frame")

        # minX, minY, width, height
        for i in range(len(deteccao)):
            if deteccao['class'][i] == 0:
                detect.append(np.array([deteccao['xmin'][i], deteccao['ymin'][i], deteccao['xmax'][i] - deteccao['xmin'][i], deteccao['ymax'][i] - deteccao['ymin'][i]]))
                scores.append(np.array(deteccao['confidence'][i]))
    
    
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.5, 100)
        tracker = Tracker(metric, max_iou_distance = 0.7, max_age = 70, n_init = 3)
        model_filename = "/home/hera/catkin_hera/src/3rdParty/vision_system/tracker/src/deep_sort/model_data/mars-small128.pb" #Change it to your directory
        encoder = generate_detections.create_box_encoder(model_filename)

        features = encoder(cv_image, detect) 
        detections_new = [Detection(bbox, score, feature) for bbox,score, feature in zip(detect,scores, features)] # cria um objeto detection para cada deteccao
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections_new]) # cria um array com as coordenadas de cada deteccao
        scores_new = np.array([d.confidence for d in detections_new]) # cria um array com as confiancas de cada deteccao
        indices = prep.non_max_suppression(boxes, 1.0 , scores_new) # aplica o non-maxima suppression para eliminar deteccoes duplicadas
        detections_new = [detections_new[i] for i in indices] # cria um array com as deteccoes que sobraram
        # Call the tracker
        tracker.predict() # faz a predicao do tracker
        tracker.update(detections_new) # atualiza o estado do objeto

        for track in tracker.tracks:
            if track.is_confirmed() and track.time_since_update > 1: # se o objeto foi confirmado e nao foi atualizado a mais de 1 frame
                continue
            bbox = track.to_tlbr()
            id = track.track_id
            cv2.rectangle(cv_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(cv_image, str(id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

        cv2.imshow("YOLO + SORT", cv_image)
        cv2.waitKey(1)



if __name__ == '__main__':
    rospy.init_node('tracker_people', log_level=rospy.INFO)
    TrackerPeople()

    tracking = TrackerPeople()

    tracking.loop()
    cv2.destroyAllWindows()



