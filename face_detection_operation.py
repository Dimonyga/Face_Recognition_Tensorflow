from multiprocessing import Process, Queue, cpu_count
from os.path import join, exists, basename
from os import makedirs, listdir, environ, rename
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
    while not qinput.empty():
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
    fnum = len(files)

    detector = get_detector()

    filldir=join(cropped_folder, 'full')
    testdir=join(cropped_folder, 'testing')
    trandir=join(cropped_folder, 'training')


    if not exists(cropped_folder):
        makedirs(cropped_folder)
    
    if not exists(join(filldir, person)):
        makedirs(join(filldir, person))
    if not exists(join(testdir, person)):
        makedirs(join(testdir, person))
    if not exists(join(trandir, person)):
        makedirs(join(trandir, person))

    for image_file in files:
        img = cv2.imread(image_file)
        results = detector.detect_faces(img)
        if not results:
            img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
            rres = detector.detect_faces(img)
            if not rres:
                continue
            else:
                results = rres

        x, y, width, height = results[0]['box']
        face = img[y:y + height, x:x + width]
        try:
            image = Image.fromarray(face)
        except ValueError:
            continue
        image = image.resize(required_size)
        face_array = np.asarray(image)
        output_file_name = basename(image_file)
        persondir=join('full',person)
        if not exists(join(cropped_folder, persondir)):
           mkdir(join(cropped_folder,persondir))
        cv2.imwrite(
            join(cropped_folder, persondir, output_file_name),
            face_array)
        faces += 1
    end = time.time()
    tm=end - start
    ips=fnum/tm
    #split dataset 90/10
    i=0
    for face in get_files(join(cropped_folder, persondir)):
        if i < 9:
            rename(face, face.replace(filldir, trandir))
            i += 1
        else:
            rename(face, face.replace(filldir, testdir))
            i = 0

    print(f"Done {person} in {tm} sec. ({ips} per second), detected {faces} from {fnum} files")

def get_detector():
    from mtcnn import MTCNN
    detector = MTCNN(
#            min_face_size=20,
            steps_threshold=[0.73, 0.8, 0.8],
#            scale_factor=0.709,
            )
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
    environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    save_cropped_face("people")
