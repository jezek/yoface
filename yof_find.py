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
import hashlib
import numpy as np
import openface
import os
import pickle
import sys
import yoloface.utils

# detection models
# detection_model['model_name'] = function(image)
detection_models = {
    'yoloface.yolov3.wider': yoloface.utils.yolo_dnn_face_detection_model_v3("./yoloface/cfg/yolov3-face.cfg", "./yoloface/model-weights/yolov3-wider_16000.weights"),
}

def openface_align_and_recognice(align_model, recognition_model):
    ''' returns function(image, rect, output_aligned='') '''
    aligner = openface.AlignDlib(align_model)
    facenet = cv2.dnn.readNetFromTorch(recognition_model)

    def facenet_face_recognize(image, rect, output_aligned = ""):
        left, top, width, height = rect
        rect = dlib.rectangle(left, top, left + width, top + height)
        aligned = aligner.align(96, image, rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if aligned is None:
            return None

        if output_aligned:
            cv2.imwrite(output_aligned, aligned.astype(np.uint8))

        # Create a 4D blob from a image.
        blob = cv2.dnn.blobFromImage(aligned, 1.0 / 255, (96, 96), [0, 0, 0], 1, crop=False)
        # set blob as input
        facenet.setInput(blob)
        # get representation vector
        return tuple(facenet.forward(yoloface.utils.get_outputs_names(facenet))[0][0].tolist())

    # return function
    return facenet_face_recognize

# recognition_models['model_name'] = function(image, rectangle)
recognition_models = {
    'openface.facenet.tiny': openface_align_and_recognice("./openface/models/dlib/shape_predictor_68_face_landmarks.dat", "./openface/models/openface/nn4.small2.v1.t7"),
}


# arguments
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--verbose', '-v', default=False, action='store_true',
                    help='Print more info')
parser.add_argument('--face', type=str, default=None,
                    help='Path to image file with one face to find')
parser.add_argument('--camera-src', type=int, default=0,
                    help='Source of the camera')
parser.add_argument('--find-distance', type=float, default=0.6,
                    help='Similiar face representation squared distance treshold')
parser.add_argument('--cache-dir', type=str, default=None,
                    help='Path to input face cache folder. If provided, cache will be created or used and than updated. Default uses no cache.')
parser.add_argument('--detection-model', type=str, default='yoloface.yolov3.wider',
                    choices=detection_models.keys(),
                    help='Face detection model name')
parser.add_argument('--recognition-model', type=str, default='openface.facenet.tiny',
                    choices=recognition_models.keys(),
                    help='Face recognition model name')
parser.add_argument('--output-dir', type=str, default=None,
                    help='Path to the output directory. If empty no output images will be generated.')

parser.add_argument('input', type=str, default='', nargs='*',
                    help='Path to input file(s). Can be a directory, image file or video file. If no input given, a camera input will be used.')
args = parser.parse_args()

if args.verbose:
    print("[i] verbose: {}".format(args.verbose))

def imagehashhex(image_array):
    return hashlib.sha512(image_array.data).hexdigest()

class ImageCache(object):
    def __init__(self, cache_dir):
        if os.path.exists(cache_dir):
            if not os.path.isdir(cache_dir):
                raise ValueError("Path to cache directory \"{}\" is not a directory".format(cache_dir))
        else:
            os.mkdir(cache_dir)
            if args.verbose:
                print("[i] Created cache dir: {}".format(cache_dir))

        self.dir = os.path.realpath(cache_dir)
        self._imagesHashCache = {}

    def get(self, image_array, function, model):
        fh = imagehashhex(image_array)

        if fh in self._imagesHashCache and function in self._imagesHashCache[fh] and model in self._imagesHashCache[fh][function]:
            return self._imagesHashCache[fh][function][model]

        pickle_file = os.path.join(self.dir, fh+'.pkl')
        if os.path.isfile(pickle_file):
            with open(pickle_file, 'rb') as input:
                storage = pickle.load(input)
                self._imagesHashCache[fh] = storage
                if function in storage and model in storage[function]:
                    return storage[function][model]
                

        return None
    

    def set(self, image_array, function, model, result):
        fh = imagehashhex(image_array)

        storage = {}
        if fh in self._imagesHashCache:
            storage = self._imagesHashCache[fh]

        if not function in storage:
            storage[function] = {}

        storage[function][model] = result

        self._imagesHashCache[fh] = storage
        pickle_file = os.path.join(self.dir, fh+'.pkl')
        with open(pickle_file, 'wb') as output:
            pickle.dump(storage, output, pickle.HIGHEST_PROTOCOL)



# cache initialization
cache = None
if not args.cache_dir is None:
    try:
        cache = ImageCache(args.cache_dir)
        if args.verbose:
            print("[i] Using cache in: {}".format(cache.dir))
    except Exception as e:
        print("[!] Creating cache failed: {}".format(e))
        cache = None
if cache is None and args.verbose:
    print("[i] Not using cache")

# uncached and cached face detect functions
face_detect = detection_models.get(args.detection_model)
def face_detect_with_cache(image):
    if cache is None:
        return face_detect(image)

    cached = cache.get(image, 'face_detect', args.detection_model)
    if cached is None:
        res = face_detect(image.copy())
        cache.set(image, 'face_detect', args.detection_model, res)
        return res 

    return cached

# uncached and cached face recognition functions
face_recognize = recognition_models.get(args.recognition_model)
def face_recognize_with_cache(image, rectangle): 
    if cache is None:
        return face_recognize(image)

    cached = cache.get(image, 'face_recognize', args.recognition_model)
    if cached is None:
        res = face_recognize(image.copy())
        cache.set(image, 'face_recognize', args.recognition_model, res)
        return res 

    return cached


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

            boxes = face_detect_with_cache(image)
            print('[i] Input image \"{}\": # detected faces: {}'.format(ainput, len(boxes)))

            for bn, box in enumerate(boxes):
                box = box[:4]
                rep = face_recognize(image, box)                
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
            print("[i] Face to find: {}".format(args.face))


        if not os.path.isfile(args.face):
            return "Find face image path \"{}\" is not a file".format(args.face)

        img = cv2.imread(args.face)
        if img.size == 0:
            return "Could not load image from {}".format(args.face)

        boxes = face_detect_with_cache(img)
        if args.verbose:
            print('[i] Find face image \"{}\": # detected faces: {}'.format(args.face, len(boxes)))

        if len(boxes) != 1:
            print("[w] Find face image \"{}\" doesn't contain only one face".format(args.face, len(boxes)))
        else:
            box = boxes[0][:4]
            rep = face_recognize(img, box) 
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
