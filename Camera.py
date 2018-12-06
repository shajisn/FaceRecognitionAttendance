import threading
import os
import re

import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import cv2

import detect_and_align
import id_data

from time import sleep
from utils import base64_to_pil_image


class Camera(object):
    """
    Manipulate camera image here..
    convert images frame from web-browser image to CV2 
        format so that recognition can be done on the frames
    """

    def __init__(self):
        self.pnet = None
        self.rnet = None
        self.onet = None
        self.id_dataset = None
        self.images_placeholder = None
        self.sess = None

        self.to_process = []
        self.to_output = []

        self.init_tensor_flow()

        thread = threading.Thread(target=self.keep_processing, args=())
        thread.daemon = True
        thread.start()

    def __exit__(self, exc_type, exc_value, traceback):
        print("Closing Tensor-flow session...")
        self.sess.close()

    def init_tensor_flow(self):
        print("Initializing Tensor-flow ...")
        self.graph = tf.Graph()
        self.sess = tf.Session()

        print("Tensorflow session created ")
        self.pnet, self.rnet, self.onet = detect_and_align.create_mtcnn(self.sess, None)
        print("Loading model file...")
        # Load the model
        self.load_model('./model/')
        print("Model file loaded...")
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        self.id_dataset = id_data.get_id_data('./ids/', self.pnet, self.rnet, self.onet, self.sess,
                                              self.embeddings, self.images_placeholder,
                                              self.phase_train_placeholder)
        self.print_id_dataset_table()

    def get_embedding_distance(self, emb1, emb2):
        dist = np.sqrt(np.sum(np.square(np.subtract(emb1, emb2))))
        return dist

    def print_id_dataset_table(self):
        nrof_samples = len(self.id_dataset)

        print('Images:')
        for i in range(nrof_samples):
            print('%1d: %s' % (i, self.id_dataset[i].image_path))
        print('')

        print('Distance matrix')
        print('         ', end='')
        for i in range(nrof_samples):
            name = os.path.splitext(os.path.basename(self.id_dataset[i].name))[0]
            print('     %s   ' % name, end='')
        print('')

        for i in range(nrof_samples):
            name = os.path.splitext(os.path.basename(self.id_dataset[i].name))[0]
            print('%s       ' % name, end='')

            for j in range(nrof_samples):
                dist = self.get_embedding_distance(self.id_dataset[i].embedding, self.id_dataset[j].embedding)
                print('  %1.4f      ' % dist, end='')
            print('')

    def load_model(self, model):
        model_exp = os.path.expanduser(model)
        if os.path.isfile(model_exp):
            print('Model filename: %s' % model_exp)
            with gfile.FastGFile(model_exp, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
        else:
            print('Model directory: %s' % model_exp)
            meta_file, ckpt_file = self.get_model_filenames(model_exp)

            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)

            saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
            saver.restore(self.sess, os.path.join(model_exp, ckpt_file))

    def get_model_filenames(self, model_dir):
        files = os.listdir(model_dir)
        meta_files = [s for s in files if s.endswith('.meta')]
        if len(meta_files) == 0:
            raise ValueError('No meta file found in the model directory (%s)' % model_dir)
        elif len(meta_files) > 1:
            raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
        meta_file = meta_files[0]
        meta_files = [s for s in files if '.ckpt' in s]
        max_step = -1
        for f in files:
            step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
            if step_str is not None and len(step_str.groups()) >= 2:
                step = int(step_str.groups()[1])
                if step > max_step:
                    max_step = step
                    ckpt_file = step_str.groups()[0]
        return meta_file, ckpt_file

    def find_matching_id(self, embedding):
        threshold = 1.1
        min_dist = 10.0
        matching_id = None

        for iddata in self.id_dataset:
            dist = self.get_embedding_distance(iddata.embedding, embedding)

            if dist < threshold and dist < min_dist:
                min_dist = dist
                matching_id = iddata.name
        return matching_id, min_dist

    def process_one(self):
        if not self.to_process:
            return

        # input is an ascii string.
        input_str = self.to_process.pop(0)

        # convert it to a pil image
        input_img = base64_to_pil_image(input_str)

        ################## where the hard work is done ############
        # output_img is an PIL image
        # output_img = input_img #self.makeup_artist.apply_makeup(input_img)

        # output_str is a base64 string in ascii
        # output_str = pil_image_to_base64(output_img)

        # convert eh base64 string in ascii to base64 string in _bytes_
        # self.to_output.append(binascii.a2b_base64(output_str))

        open_cv_image = np.array(input_img)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        print("Processing frame...")

        face_patches, padded_bounding_boxes, landmarks = detect_and_align.align_image(open_cv_image,
                                                                                      self.pnet,
                                                                                      self.rnet,
                                                                                      self.onet)
        matching_id = "Unknown"
        if len(face_patches) > 0:
            face_patches = np.stack(face_patches)
            feed_dict = {self.images_placeholder: face_patches, self.phase_train_placeholder: False}
            embs = self.sess.run(self.embeddings, feed_dict=feed_dict)

            print('Matches in frame:')
            for i in range(len(embs)):
                bb = padded_bounding_boxes[i]

                matching_id, dist = self.find_matching_id(embs[i, :])
                if matching_id:
                    print('Hi %s! Distance: %1.4f' % (matching_id, dist))
                else:
                    matching_id = 'Unknown'
                    print('Unknown! Couldn\'t fint match.')

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(open_cv_image, matching_id, (bb[0], bb[3]), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.rectangle(open_cv_image, (bb[0], bb[1]), (bb[2], bb[3]), (255, 0, 0), 2)
        else:
            print("No face patches")

        match_dict = {}
        match_dict[matching_id] = open_cv_image
        # adding matching_name=>frame to array
        self.to_output.append(match_dict)

    def keep_processing(self):
        while True:
            self.process_one()
            sleep(0.01)

    def enqueue_input(self, input):
        self.to_process.append(input)

    def get_frame(self):
        while not self.to_output:
            sleep(0.05)
        return self.to_output.pop(0)
