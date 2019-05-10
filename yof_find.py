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
parser.add_argument('--verbose', default=True,
                    help='Print more info')
args = parser.parse_args()


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
    return ofnet.forward(yoloface.utils.get_outputs_names(ofnet))[0][0]

def _main():
    yolov3_face_detect = yoloface.utils.yolo_dnn_face_detection_model_v3("./yoloface/cfg/yolov3-face.cfg", "./yoloface/model-weights/yolov3-wider_16000.weights") 

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
            rep = facenet_face_recognize(img, boxes[0][:4]) 
            if args.verbose:
                print('[i] Find face image \"{}\": detected face {} representation vector:\n{}'.format(args.face, boxes[0], rep))




if __name__ == '__main__':
    sys.exit(_main())
