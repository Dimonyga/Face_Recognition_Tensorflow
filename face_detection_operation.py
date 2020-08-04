from multiprocessing import Process, Queue, cpu_count
from os.path import join, exists, basename
from os import mkdir, listdir, environ
import time
import glob
import cv2
import numpy as np
from PIL import Image

def get_files(person):
    file_types = ["*.png", "*.PNG", "*.JPEG", "*.jpeg", "*.jpg", "*.JPG"]
    files=[]
    for file_type in file_types:
        files += glob.glob(join(person, file_type))
    return files

def save_cropped_face(images_root_folder):

    if not exists(images_root_folder):
        return Exception("Input Images folder is not exist.")
    people = listdir(images_root_folder)
    task_queue = Queue()
    cpus = cpu_count()
    processes = cpus*[None]
    for task in people:
        print(f"add task {task}")
        task_queue.put(task)
    for cpu in range(cpus):
        processes[cpu-1] = Process(target=worker, args=(task_queue,))
        processes[cpu-1].start()
        #Process(target=worker, args=(task_queue,)).start()
    for process in processes:
        process.join()


def worker(qinput):
    while True:
        person = qinput.get()
        process_people(person)
        #qinput.task_done()

def process_people(person):
    start = time.time()
    print(f"Start processsing {person}")
    required_size=(224, 224)
    cropped_folder='dataset'
    images_root_folder='people'
    files = get_files(join(images_root_folder, person))
    faces = 0
    fnum = len(files)-1

    detector = get_detector()

    for image_file in files:
        img = cv2.imread(image_file)
        results = detector.detect_faces(img)
        if not results:
            continue

        x, y, width, height = results[0]['box']
        face = img[y:y + height, x:x + width]
        try:
            image = Image.fromarray(face)
        except ValueError:
            continue
        image = image.resize(required_size)
        face_array = np.asarray(image)

        if not exists(cropped_folder):
            mkdir(cropped_folder)

        if not exists(join(cropped_folder, 'ds')):
            mkdir(join(cropped_folder, 'ds'))
        output_file_name = basename(image_file)
        persondir=join('ds',person)
        if not exists(join(cropped_folder, persondir)):
           mkdir(join(cropped_folder,persondir))
        cv2.imwrite(
            join(cropped_folder, persondir, output_file_name),
            face_array)
        faces += 1
    end = time.time()
    tm=end - start
    ips=fnum/tm
    print(f"Done {person} in {tm} sec. ({ips} per second), detected {faces} from {fnum} files")

def get_detector():
    from mtcnn import MTCNN
    detector = MTCNN(steps_threshold=[0.55, 0.86, 0.78])
    return detector

def get_detected_face(filename, required_size=(224, 224)):
    img = cv2.imread(filename)
    detector = get_detector()
    results = detector.detect_faces(img)
    x, y, width, height = results[0]['box']
    face = img[y:y + height, x:x + width]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array, face


if __name__ == "__main__":
    environ["CUDA_VISIBLE_DEVICES"] = "-1"
    environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    save_cropped_face("people")
