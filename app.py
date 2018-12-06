import os
import re
import logging
import json
from datetime import datetime
from sys import stdout

from flask import Flask, render_template, request, Response
from flask_socketio import SocketIO, emit

import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from scipy import misc
import cv2

from models import User, db_session, Attendance
from Camera import Camera

import detect_and_align
import id_data
from utils import base64_to_pil_image

app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(stdout))
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
print(APP_ROOT)

POSTGRES = {
    'user': 'postgres',
    'pw': 'ease@inapp1',
    'db': 'facerecognition',
    'host': 'localhost',
    'port': '5432',
}
dbUrl = 'postgresql://%(user)s:%(pw)s@%(host)s:%(port)s/%(db)s' % POSTGRES
print("Connecting to DB " + dbUrl)
app.config['SQLALCHEMY_DATABASE_URI'] = dbUrl
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

socketio = SocketIO(app)

camera = Camera()


# socketio.run(app)

@socketio.on('input image', namespace='/video')
def video_message(input):
    # print(input)
    input = input.split(",")[1]
    # print(input)
    camera.enqueue_input(input)
    # Camera.enqueue_input(base64_to_pil_image(input))


@socketio.on('connect', namespace='/video')
def test_connect():
    app.logger.info("client connected")


@app.route('/')
def index():
    return render_template("index.html", title="Admin Login")


@app.after_request
def set_response_headers(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


@app.route('/login', methods=['POST'])
def login():
    username = request.form['user']
    password = request.form['password']

    registered_user = User.query.filter_by(userID=username, password=password).first()
    if registered_user is None:
        return render_template("index.html", title="Admin Login", msg="The username or password is incorrect")

    return render_template("task.html", msg="")


@app.route('/register', methods=['POST'])
def register():
    return render_template("signup.html", title="Add Employee Details")


@app.after_request
def set_response_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/signup', methods=['POST'])
def signup():
    user_id = str(request.form["user"])
    password = str(request.form["password"])
    email = str(request.form["email"])
    name = str(request.form["name"])
    check_user = User.query.filter_by(userID=user_id, password=password).first()
    if check_user is None:
        new_user = User(user_id, password, email, name)
        db_session.add(new_user)
        db_session.commit()
        return render_template("upload_photo.html", title="Upload Photo", user_id=name)
    else:
        return render_template("upload_photo.html", title="Upload Photo", user_id=name)


@app.route("/upload_photo.html", methods=['POST'])
def upload():
    # dict = request.form
    # for key in dict:
    #     print('form key ' + dict[key])

    target = os.path.join(APP_ROOT, "ids/")
    if not os.path.isdir(target):
        os.mkdir(target)

    user_id = str(request.form['username'])
    model = os.path.join(target, user_id + "/")
    if not os.path.isdir(model):
        os.mkdir(model)

    images_data = request.form['canvasImage']
    json_data = json.loads(images_data)

    # print(image_data)
    # for i in range(len(image_data)):
    #     print(i)
    for i in range(0, 4):
        name = "image_" + str(i)
        print("Image Name : " + name)
        image_data = json_data[name]
        if image_data:
            content = image_data.split(';')[1]
            image_encoded = content.split(',')[1]
            input_img = base64_to_pil_image(image_encoded)

            basename = user_id + name
            suffix = datetime.now().strftime("%y%m%d_%H%M%S.jpg")
            filename = "_".join([basename, suffix])
            print("Saving to: " + model + filename)
            input_img.save(model + filename)

    return render_template("task.html")


@app.route('/verify', methods=['POST', 'GET'])
def verify():
    return render_template("verify.html")


@app.route('/attendance', methods=['POST', 'GET'])
def mark_attendance():
    user_name = str(request.form["username"])
    check_user = User.query.filter_by(name=user_name).first()
    if check_user is not None:
        new_attendance = Attendance(check_user.userID, datetime.utcnow())
        db_session.add(new_attendance)
        db_session.commit()
    return render_template("verify.html")


@app.route('/logout', methods=['POST'])
def logout():
    return render_template("index.html", title="Admin Login", msg1="Logged out please login again")


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


def gen():
    while True:
        match_dict = camera.get_frame()
        print("Returning converted frames...", match_dict)
        for i, (match_name, frame) in enumerate(match_dict.items()):
            print("index: {}, key: {}".format(i, match_name))
            with app.test_request_context():
                socketio.emit("response", match_name, namespace='/video')
            encimg = cv2.imencode('.jpg', frame)[1].tostring()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + encimg + b'\r\n')


def gen1():
    """Video streaming generator function."""

    print("Starting to generate frames!!!")
    with tf.Graph().as_default():
        with tf.Session() as sess:
            print("Initialize tensor")
            pnet, rnet, onet = detect_and_align.create_mtcnn(sess, None)
            print("Loading model file...")
            # Load the model
            load_model('./model/')
            print("Model file loaded...")
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            id_dataset = id_data.get_id_data('./ids/', pnet, rnet, onet, sess, embeddings, images_placeholder,
                                             phase_train_placeholder)
            print_id_dataset_table(id_dataset)

            while True:
                frame = camera.get_frame()
                print("Processing frame...")

                face_patches, padded_bounding_boxes, landmarks = detect_and_align.align_image(frame, pnet, rnet, onet)

                if len(face_patches) > 0:
                    face_patches = np.stack(face_patches)
                    feed_dict = {images_placeholder: face_patches, phase_train_placeholder: False}
                    embs = sess.run(embeddings, feed_dict=feed_dict)

                    print('Matches in frame:')
                    for i in range(len(embs)):
                        bb = padded_bounding_boxes[i]

                        matching_id, dist = find_matching_id(id_dataset, embs[i, :])
                        if matching_id:
                            print('Hi %s! Distance: %1.4f' % (matching_id, dist))
                        else:
                            matching_id = 'Unkown'
                            print('Unkown! Couldn\'t fint match.')

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame, matching_id, (bb[0], bb[3]), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

                        cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (255, 0, 0), 2)
                else:
                    print("No face patches")

                # cnt = cv2.imencode('.jpeg', frame)[1]
                # b64 = base64.encodebytes(cnt)

                encimg = cv2.imencode('.jpg', frame)[1].tostring()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + encimg + b'\r\n')


@app.route("/train", methods=['POST'])
def train():
    camera = None
    camera = Camera()
    # with tf.Graph().as_default():
    #     with tf.Session() as sess:
    #         # Create Multi-task Cascaded Convolutional Networks
    #         pnet, rnet, onet = detect_and_align.create_mtcnn(sess, None)
    #         # Load the model
    #         load_model('./model/')
    #         images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    #         embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    #         phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    #
    #         id_dataset = id_data.get_id_data('./ids/', pnet, rnet, onet, sess, embeddings, images_placeholder,
    #                                          phase_train_placeholder)
    #         print_id_dataset_table(id_dataset)
    #
    #         test_run(pnet, rnet, onet, sess, images_placeholder, phase_train_placeholder, embeddings, id_dataset,
    #                  None)

    return render_template("task.html", title="Home", msg="Training completed")


def load_model(model):
    model_exp = os.path.expanduser(model)
    if os.path.isfile(model_exp):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))


def get_embedding_distance(emb1, emb2):
    dist = np.sqrt(np.sum(np.square(np.subtract(emb1, emb2))))
    return dist


def print_id_dataset_table(id_dataset):
    nrof_samples = len(id_dataset)

    print('Images:')
    for i in range(nrof_samples):
        print('%1d: %s' % (i, id_dataset[i].image_path))
    print('')

    print('Distance matrix')
    print('         ', end='')
    for i in range(nrof_samples):
        name = os.path.splitext(os.path.basename(id_dataset[i].name))[0]
        print('     %s   ' % name, end='')
    print('')
    for i in range(nrof_samples):
        name = os.path.splitext(os.path.basename(id_dataset[i].name))[0]
        print('%s       ' % name, end='')
        for j in range(nrof_samples):
            dist = get_embedding_distance(id_dataset[i].embedding, id_dataset[j].embedding)
            print('  %1.4f      ' % dist, end='')
        print('')


def test_run(pnet, rnet, onet, sess, images_placeholder, phase_train_placeholder, embeddings, id_dataset, test_folder):
    if test_folder is None:
        return

    image_names = os.listdir(os.path.expanduser(test_folder))
    image_paths = [os.path.join(test_folder, img) for img in image_names]
    nrof_images = len(image_names)
    aligned_images = []
    aligned_image_paths = []

    for i in range(nrof_images):
        image = misc.imread(image_paths[i])
        face_patches, _, _ = detect_and_align.align_image(image, pnet, rnet, onet)
        aligned_images = aligned_images + face_patches
        aligned_image_paths = aligned_image_paths + [image_paths[i]] * len(face_patches)

    aligned_images = np.stack(aligned_images)

    feed_dict = {images_placeholder: aligned_images, phase_train_placeholder: False}
    embs = sess.run(embeddings, feed_dict=feed_dict)

    for i in range(len(embs)):
        misc.imsave('outfile' + str(i) + '.jpg', aligned_images[i])
        matching_id, dist = find_matching_id(id_dataset, embs[i, :])
        if matching_id:
            print('Found match %s for %s! Distance: %1.4f' % (matching_id, aligned_image_paths[i], dist))
        else:
            print('Couldn\'t fint match for %s' % (aligned_image_paths[i]))


def get_model_filenames(model_dir):
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


def find_matching_id(id_dataset, embedding):
    threshold = 1.1
    min_dist = 10.0
    matching_id = None

    for iddata in id_dataset:
        dist = get_embedding_distance(iddata.embedding, embedding)

        if dist < threshold and dist < min_dist:
            min_dist = dist
            matching_id = iddata.name
    return matching_id, min_dist


def get_embedding_distance(emb1, emb2):
    dist = np.sqrt(np.sum(np.square(np.subtract(emb1, emb2))))
    return dist


# start application
if __name__ == '__main__':
    app.secret_key = 'secretkey'
    # app.run(host="0.0.0.0", port=8181, debug=False, ssl_context='adhoc')
    app.run(host="0.0.0.0", port=8181, debug=True)
