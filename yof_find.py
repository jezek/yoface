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
import filetype
import hashlib
import numpy as np
import openface
import os
import pickle
import shutil
import struct
import sys
import yoloface.utils

script_path = os.path.dirname(os.path.realpath(sys.argv[0]))

# detection models
# detection_model['model_name'] = function(image)
detection_models = {
    'yoloface.yolov3.wider': yoloface.utils.yolo_dnn_face_detection_model_v3(os.path.join(script_path, "yoloface/cfg/yolov3-face.cfg"), os.path.join(script_path, "yoloface/model-weights/yolov3-wider_16000.weights")),
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
    'openface.facenet.tiny': openface_align_and_recognice(os.path.join(script_path, "openface/models/dlib/shape_predictor_68_face_landmarks.dat"), os.path.join(script_path, "openface/models/openface/nn4.small2.v1.t7")),
}

# arguments
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--verbose', '-v', default=False, action='store_true',
                    help='Print more info')
parser.add_argument('--face', '-f', type=str, default=None,
                    help='Path to directory or image file. If image, face name will be derived from file name and image has to contain only one face. For directory, face name will be the directory name and images under that folder will be used.')
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
parser.add_argument('--output-dir', '-o', type=str, default=None,
                    help='Path to the output directory. If empty no output images will be generated.')
parser.add_argument('--recursive', '-r', default=False, action='store_true',
                    help='Recursive input directory crawl')

parser.add_argument('input', type=str, default='', nargs='*',
                    help='Path to input file(s). Can be a directory, image file or video file. If no input given, a camera input will be used.')
args = parser.parse_args()


if args.verbose:
    sys.stderr.write("[i] verbose: {}\n".format(args.verbose))
    sys.stderr.write("[i] face find distance treshold: {}\n".format(args.find_distance))

def imagehashhex(data):
    return hashlib.sha512(data).hexdigest()

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

    def get(self, data, function, model):
        fh = imagehashhex(data)

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
    

    def set(self, data, function, model, result):
        fh = imagehashhex(data)

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
if args.cache_dir is not None:
    try:
        cache = ImageCache(args.cache_dir)
        if args.verbose:
            sys.stderr.write("[i] Using cache in: {}\n".format(cache.dir))
    except Exception as e:
        sys.stderr.write("[!] Creating cache failed: {}\n".format(e))
        cache = None
if cache is None and args.verbose:
    sys.stderr.write("[i] Not using cache\n")

# uncached and cached face detect functions
face_detect = detection_models.get(args.detection_model)
def face_detect_with_cache(image):
    if cache is None:
        return face_detect(image)

    cached = cache.get(image.data, 'face_detect', args.detection_model)
    if cached is None:
        res = face_detect(image.copy())
        if args.verbose:
            sys.stderr.write("[i] Caching image {} detected faces\n".format(imagehashhex(image.data)))
        cache.set(image.data, 'face_detect', args.detection_model, res)
        return res 

    return cached

# uncached and cached face recognition functions
face_recognize = recognition_models.get(args.recognition_model)
def face_recognize_with_cache(image, rectangle): 
    if cache is None:
        return face_recognize(image, rectangle)

    
    rdata = struct.pack("{}l".format(len(rectangle)), *rectangle)
    cached = cache.get(image.data, 'face_recognize', args.recognition_model)
    if cached is None:
        cached = {}

    if not rdata in cached:
        res = face_recognize(image.copy(), rectangle)
        cached[rdata] = res
        cache.set(image.data, 'face_recognize', args.recognition_model, cached)
        if args.verbose:
            sys.stderr.write("[i] Cached image {} face representation for box {}\n".format(imagehashhex(image.data), rectangle))
        return res 

    return cached[rdata]


def find_known_faces_in_image(image_realpath, known):
    image = cv2.imread(image_realpath)
    if image is None or image.size == 0:
        return "Input \"{}\" is not a valid image".format(image_realpath)

    sys.stderr.write("[i] Input image \"{}\"\n".format(image_realpath))

    boxes = face_detect_with_cache(image)
    if args.verbose:
        sys.stderr.write("[i] # detected faces: {}\n".format(len(boxes)))

    for bn, box in enumerate(boxes):
        box = box[:4]
        rep = face_recognize_with_cache(image, box)                
        if rep is not None:
            for knn, known_name in enumerate(known):
                for krn, known_rep in enumerate(known[known_name]):
                    d = np.array(rep) - np.array(known_rep)
                    sqd = np.dot(d, d)
                    if args.verbose:
                        sys.stderr.write("[i] Comparing face in box {} to {} ({}). Got distance {}\n\t{}\n".format(box, known_name, krn, sqd, known[known_name][known_rep]))

                    if sqd < args.find_distance:
                        # found face within search distance
                        print("{}\t{}\t{}\t{}".format(image_realpath, known_name, box, sqd))

                        if args.output_dir is not None:
                            out_name, out_ext = os.path.splitext(image_realpath)
                            out_file = os.path.join(args.output_dir, os.path.basename(out_name)+"_"+imagehashhex(image.data)[:10]+out_ext)
                            try:
                                shutil.copyfile(image_realpath, out_file)
                                sys.stderr.write("[i] Saved input image to {}\n".format(out_file))
                            except IOError as e:
                                sys.stderr.write("[!] Copy file error: {}\n".format(e))
                            
                        break

def find_known_faces_in_video(video_path, known):
    #TODO
    return "TODO video find face"

def find_known_faces(input, known, root_input=False):
    def string_input():
        if input=='':
            if not root_input:
                return "[!] Empty non root input"
            sys.stderr.write("camera has to be done\n")
            #TODO camera
            return

        ainput = os.path.realpath(input)
        if os.path.isdir(ainput):
            if root_input or args.recursive:
                sys.stderr.write("[i] looking in {} for input files\n".format(ainput))
                return find_known_faces([os.path.join(ainput, file) for file in os.listdir(ainput)], known)
        elif os.path.isfile(ainput):
            kind = filetype.guess(ainput)
            if kind is None:
                return "[!] Input \"{}\" is not supported".format(ainput)
            mime = kind.mime.split("/")[0]

            supported = {
                'image': find_known_faces_in_image,
                'video': find_known_faces_in_video,
            }
            if mime not in supported.keys():
                return "[!] Input \"{}\" is not supported mime group \"{}\". Only {} are supported".format(ainput, mime, supported.keys())

            return supported[mime](ainput, known)
        else:
            return "[!] Input \"{}\" is not a directory, or a file".format(ainput)
        
        return

    def list_input():
        for item in input:
            res = find_known_faces(item, known, root_input)
            if res:
                sys.stderr.write("[!] Input item \"{}\" returned: {}\n".format(item, res))
        return

    input_types = {
        str:string_input,
        list:list_input,
    }
    if not type(input) in input_types:
        return "[!] Unknown input type: {}".format(type(input))

    return input_types[type(input)]()

def _main():

    if args.output_dir is not None and not os.path.isdir(args.output_dir):
        return "[!] Output directory does not exist {}".format(args.output_dir)

    # known_faces{face_name_string:{representation_vector_as_tuple:{full_path_filename_string:(box_tuple,)}}}
    known_faces = {}
    def add_known_face_from_image(face_name, image_path):
        img = cv2.imread(image_path)
        if img.size == 0:
            sys.stderr.write("[!] Could not load image: {}\n".format(image_path))
            return

        if args.verbose:
            sys.stderr.write("[i] Find face image: {}\n".format(image_path))

        boxes = face_detect_with_cache(img)

        if args.verbose:
            sys.stderr.write("[i] # detected faces: {}\n".format(len(boxes)))

        if len(boxes) != 1:
            sys.stderr.write("[w] Find face image \"{}\" doesn't contain only one face\n".format(image_path, len(boxes)))
            return

        box = boxes[0][:4]
        rep = face_recognize_with_cache(img, box) 
        if rep is None:
            sys.stderr.write("[!] Find face image \"{}\": detected face {} can't align for recognition\n".format(image_path, box))
            return

        #if args.verbose:
        #    sys.stderr.write("[i] Find face image \"{}\": detected face {} representation vector:\n{}\n".format(image_path, box, rep))

        if face_name in known_faces:
            if rep in known_faces[face_name]:
                if os.path.realpath(image_path) in known_faces[face_name][rep]:
                    known_faces[face_name][rep][os.path.realpath(image_path)]+=(box,)
                else:
                    known_faces[face_name][rep][os.path.realpath(image_path)]=(box,)
            else:
                known_faces[face_name][rep]={os.path.realpath(image_path):(box,)}
        else:
            known_faces[face_name]={}
            known_faces[face_name][rep]={os.path.realpath(image_path):(box,)}
        return

    if args.face is not None:
        if args.verbose:
            sys.stderr.write("[i] Face to find: {}\n".format(args.face))

        if os.path.isdir(args.face):
            face_name = os.path.basename(os.path.realpath(args.face))
            for possible_face_file in os.listdir(args.face):
                possible_face_file = os.path.join(os.path.realpath(args.face), possible_face_file)
                if os.path.isfile(possible_face_file):
                    kind = filetype.guess(possible_face_file)
                    if kind is not None and kind.mime.split("/")[0] == 'image':
                        add_known_face_from_image(face_name, possible_face_file)
        elif os.path.isfile(args.face):
            face_name = os.path.splitext(os.path.basename(args.face))[0]
            if face_name == '':
                face_name = args.face 

            add_known_face_from_image(face_name, os.path.realpath(args.face))
        else:
            sys.stderr.write("[!] Face argument \"{}\" is not a file or directory\n".format(args.face))




    # print("known_faces: {}".format(known_faces))
    if len(known_faces) == 0:
        return "[!] No known faces to find"

    sys.stderr.write("[i] Looking for {} in {}\n".format(known_faces.keys(), args.input))

    return find_known_faces(args.input, known_faces, root_input=True)


if __name__ == '__main__':
    sys.exit(_main())
