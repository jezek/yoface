#/usr/bin/python

# *******************************************************************
#
# Author : jEzEk, 2019
# Email  : jezek@chicki.sk
# Github : https://github.com/jezek
#
# Face find using yolofaces's YOLOv3 and openfaces's FaceNet
#
# Description : yof_find.py
# The main code of the Face find using yolofaces's YOLOv3 and openfaces's ResNet
#
# *******************************************************************

# Usage example:  python yof_find.py \
#                 --face samples/faces/Family\ Father/front.jpg \
#                 samples/*
#
#                 python yof_find.py \
#                 --face samples/faces/Family\ Father/front.jpg \
#                 --output-dir outputs/
#                 samples/video.mp4
#
#                 python yof_find.py \
#                 --face samples/faces/Family\ Father/front.jpg \
#                 --output-dir outputs/
#                 --src 1 


import argparse
import dlib
import cv2
import numpy as np
import openface
import os
import sys
import yoloface.utils

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, default='', nargs='*',
                    help='Path to input file(s). Can be a directory, image file or video file. If no input given, a camera input will be used.')
parser.add_argument('--face', type=str, default=None,
                    help='Path to image file with one face to find')
parser.add_argument('--output-dir', type=str, default=None,
                    help='Path to the output directory. If empty no output images will be generated.')
parser.add_argument('--camera-src', type=int, default=0,
                    help='Source of the camera')
parser.add_argument('--verbose', default=False,
                    help='Print more info')
parser.add_argument('--find-distance', type=float, default=0.6,
                    help='Similiar face representation squared distance treshold')
args = parser.parse_args()


yolov3_face_detect = yoloface.utils.yolo_dnn_face_detection_model_v3("./yoloface/cfg/yolov3-face.cfg", "./yoloface/model-weights/yolov3-wider_16000.weights") 
#TODO rename
align = openface.AlignDlib("./openface/models/dlib/shape_predictor_68_face_landmarks.dat")
ofnet = cv2.dnn.readNetFromTorch("./openface/models/openface/nn4.small2.v1.t7")
def facenet_face_recognize(image, rect, output_aligned = ""):
    left, top, width, height = rect
    rect = dlib.rectangle(left, top, left + width, top + height)
    aligned = align.align(96, image, rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if aligned is None:
        print("[!] ==> Can't align face {} in image {}".format(rect, output_aligned))
        return None

    if output_aligned:
        cv2.imwrite(output_aligned, aligned.astype(np.uint8))

    # Create a 4D blob from a image.
    blob = cv2.dnn.blobFromImage(aligned, 1.0 / 255, (96, 96),
                                [0, 0, 0], 1, crop=False)
    # set blob as input
    ofnet.setInput(blob)
    # get representation vector
    return tuple(ofnet.forward(yoloface.utils.get_outputs_names(ofnet))[0][0].tolist())


def find_known_faces(input, known, root_input=False):
    def string_input():
        if input=='':
            if not root_input:
                return "Empty non root input"
            print("camera has to be done")
            #TODO camera
            return

        ainput = os.path.realpath(input)
        if os.path.isdir(ainput):
            return "TODO directory crawl"
        elif os.path.isfile(ainput):
            image = cv2.imread(ainput)
            if image.size == 0:
                return "Input \"{}\" is not a valid image".format(ainput)

            boxes = yolov3_face_detect(image)
            print('[i] Input image \"{}\": # detected faces: {}'.format(ainput, len(boxes)))

            for bn, box in enumerate(boxes):
                box = box[:4]
                rep = facenet_face_recognize(image, box)                
                if not rep is None:
                    for knn, known_name in enumerate(known):
                        for krn, known_rep in enumerate(known[known_name]):
                            d = np.array(rep) - np.array(known_rep)
                            sqd = np.dot(d, d)
                            if sqd < args.find_distance or args.verbose:
                                print("{};{};{};{}".format(ainput, known_name, box, sqd))
                
            
        else:
            return "Input \"{}\" is not a directory, or a file".format(ainput)
        
        return

    def list_input():
        for item in input:
            res = find_known_faces(item, known, root_input)
            if res:
                print("[!] Input item \"{}\" returned: {}".format(item, res))
        return

    input_types = {
        str:string_input,
        list:list_input,
    }
    if not type(input) in input_types:
        return "Unknown input type: {}".format(type(input))

    return input_types[type(input)]()

def _main():

    # known_faces{face_name_string:{representation_vector_as_tuple:{full_path_filename_string:(box_tuple,)}}}
    known_faces = {}

    if args.face is None:
        return "No face to find"
    else:
        if args.verbose:
            print("[i] Face to find: ", args.face)


        if not os.path.isfile(args.face):
            return "Find face image path \"{}\" is not a file".format(args.face)

        img = cv2.imread(args.face)
        if img.size == 0:
            return "Could not load image from {}".format(args.face)

        boxes = yolov3_face_detect(img)
        if args.verbose:
            print('[i] Find face image \"{}\": # detected faces: {}'.format(args.face, len(boxes)))

        if len(boxes) != 1:
            print("[w] Find face image \"{}\" doesn't contain only one face".format(args.face, len(boxes)))
        else:
            box = boxes[0][:4]
            rep = facenet_face_recognize(img, box) 
            if rep is None:
                print("[!] Find face image \"{}\": detected face {} can't align for recognition".format(args.face, box))
            else:
                if args.verbose:
                    print('[i] Find face image \"{}\": detected face {} representation vector:\n{}'.format(args.face, box, rep))

                face_name = os.path.splitext(os.path.basename(args.face))[0]
                if face_name == '':
                    face_name = args.face 

                if face_name in known_faces:
                    if rep in known_faces[face_name]:
                        if os.path.realpath(args.face) in known_faces[face_name][rep]:
                            known_faces[face_name][rep][os.path.realpath(args.face)]+=(box,)
                        else:
                            known_faces[face_name][rep][os.path.realpath(args.face)]=(box,)
                    else:
                        known_faces[face_name][rep]={os.path.realpath(args.face):(box,)}
                else:
                    known_faces[face_name]={}
                    known_faces[face_name][rep]={os.path.realpath(args.face):(box,)}

    #print("known_faces: {}".format(known_faces))
    if len(known_faces) == 0:
        return "No known faces to find"

    return find_known_faces(args.input, known_faces, root_input=True)


if __name__ == '__main__':
    sys.exit(_main())
