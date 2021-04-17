#================================================================
#
#   File name   : object_tracker.py
#   Author      : PyLessons
#   Created date: 2020-09-17
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : code to track detected object from video or webcam
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import sysconfig
import sys
##np.set_printoptions(threshold=sys.maxsize)
import tensorflow as tf
from yolov3.utils import Load_Yolo_model, image_preprocess, postprocess_boxes, nms, draw_bbox, read_class_names
from yolov3.configs import *
import time
import csv

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

#video_path   = "D:/PELUSO/ITSligo/lectures_MENG/4-Symulation and Testing/assignments/assignment 2 - group/TensorFlow-2.x-YOLOv3-master/IMAGES/test.mp4"##"./IMAGES/test.mp4"##"../Modelling  Simulation Group 3 Test Video.mp4"
video_path = "D:/PELUSO/ITSligo/lectures_MENG/4-Symulation and Testing/assignments/assignment 2 - group/simulation_V5.mp4"
output_path = "D:/PELUSO/ITSligo/lectures_MENG/4-Symulation and Testing/assignments/assignment 2 - group/Simulation_V5_detection.mp4"
YOLO_COCO_CLASSES = "D:/PELUSO/ITSligo/lectures_MENG/4-Symulation and Testing/assignments/assignment 2 - group/TensorFlow-2.x-YOLOv3-master/model_data/coco/coco.names"


def Object_tracking(Yolo, video_path, output_path, input_size=416, show=True, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3, iou_threshold=0.45, rectangle_colors='', Track_only = []):
    output_file = "D:/PELUSO/ITSligo/lectures_MENG/4-Symulation and Testing/assignments/assignment 2 - group/simV5_anotation.CSV"
    csv_file = open(output_file, mode='a')  #new
    results_csv = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    results_csv.writerow(['Frame_index','Score','Confidence','Pixel_Area','X1','Y1','X2','Y2','ClassID'])
    
    # Definition of the parameters
    max_cosine_distance = 0.9
    nn_budget = None
    
    #initialize deep sort object
    model_filename = "D:/PELUSO/ITSligo/lectures_MENG/4-Symulation and Testing/assignments/assignment 2 - group/TensorFlow-2.x-YOLOv3-master/model_data/mars-small128.pb"##'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    times, times_2 = [], []

    if video_path:
        vid = cv2.VideoCapture(video_path) # detect on video
    else:
        vid = cv2.VideoCapture(0) # detect from webcam

    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'mp4v')##(*'XVID')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height)) # output_path must be .mp4
    print("FPS:::::"+str(fps))

    NUM_CLASS = read_class_names(CLASSES)
    key_list = list(NUM_CLASS.keys()) 
    val_list = list(NUM_CLASS.values())
    frame_idx = 0
    while True:
        _, frame = vid.read()

        try:
            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        except:
            break
        
        image_data = image_preprocess(np.copy(original_frame), [input_size, input_size])
        #image_data = tf.expand_dims(image_data, 0)
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        t1 = time.time()
        if YOLO_FRAMEWORK == "tf":
            pred_bbox = Yolo.predict(image_data)
        elif YOLO_FRAMEWORK == "trt":
            batched_input = tf.constant(image_data)
            result = Yolo(batched_input)
            pred_bbox = []
            for key, value in result.items():
                value = value.numpy()
                pred_bbox.append(value)
        
        #t1 = time.time()
        #pred_bbox = Yolo.predict(image_data)
        t2 = time.time()
        
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_frame, input_size, score_threshold)
        
        bboxes = nms(bboxes, iou_threshold, method='nms', sigma = 0.4)
        print(np.argmax(pred_bbox[:, 5:], axis=-1))

        # extract bboxes to boxes (x, y, width, height), scores and names
        boxes, scores, names = [], [], []
        for bbox in bboxes:
            if len(Track_only) !=0 and NUM_CLASS[int(bbox[5])] in Track_only or len(Track_only) == 0:
                boxes.append([bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(int)-bbox[0].astype(int), bbox[3].astype(int)-bbox[1].astype(int)])
                scores.append(bbox[4])
                names.append(NUM_CLASS[int(bbox[5])])

        # Obtain all the detections for the given frame.
        boxes = np.array(boxes) 
        names = np.array(names)
        scores = np.array(scores)##---esse eh o fidaputi
        features = np.array(encoder(original_frame, boxes))
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes, scores, names, features)]

        # Pass detections to the deepsort object and obtain the track information.
        tracker.predict()
        tracker.update(detections)

        # Obtain info from the tracks
        tracked_bboxes = []
        tracked_scores = []
        for i,track in enumerate(tracker.tracks):
            if not track.is_confirmed() or track.time_since_update > 5:
                continue 
            bbox = track.to_tlbr() # Get the corrected/predicted bounding box
            class_name = track.get_class() #Get the class name of particular object
            tracking_id = track.track_id # Get the ID for the particular track
            index = key_list[val_list.index(class_name)] # Get predicted object index by object name
            tracked_bboxes.append(bbox.tolist() + [tracking_id, index]) # Structure data, that we could use it with our draw_bbox function
            try:
                tracked_scores.append(track.Confidence)      ##new
            except:
                print("skip")


        # draw detection on frame draw_bbox(image, bboxes, CLASSES=YOLO_COCO_CLASSES, show_label=True, show_confidence = True, Text_colors=(255,255,0), rectangle_colors='', tracking=False)
        frame_idx +=1
        print("------frame_idx-------"+str(frame_idx))
        image = draw_bbox(original_frame,results_csv, tracked_bboxes,frame_idx, tracked_scores,show_label=True, show_confidence = True, CLASSES=CLASSES, tracking=True)     ##new

        t3 = time.time()
        times.append(t2-t1)
        times_2.append(t3-t1)
        
        times = times[-20:]
        times_2 = times_2[-20:]

        ms = sum(times)/len(times)*1000
        fps = 1000 / ms
        fps2 = 1000 / (sum(times_2)/len(times_2)*1000)
        
        ##image = cv2.putText(image, "Time: {:.1f} FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        # draw original yolo detection
        #image = draw_bbox(image, bboxes, CLASSES=CLASSES, show_label=False, rectangle_colors=rectangle_colors, tracking=True)

        print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps, fps2))
        if output_path != '': out.write(image)
        if show:
            cv2.imshow('output', image)
            
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
            
    results_csv.close()
    cv2.destroyAllWindows()


yolo = Load_Yolo_model()
Object_tracking(yolo, video_path, output_path, input_size=YOLO_INPUT_SIZE, show=True, score_threshold=0.45, iou_threshold=0.35, rectangle_colors=(255,0,0), Track_only = ["person", "bicycle", "car"])
