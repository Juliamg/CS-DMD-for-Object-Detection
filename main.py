import numpy as np
import os
import cv2
import random
import argparse

from sklearn.decomposition import PCA
from skimage.color import rgb2gray
from src_algorithm import src_algorithm
from DMD_video_sampling import parse_videos
from face_detection import process_DMD_snapshots

IGNORE_FILES = ['.DS_Store']
shape = (5, 5)
prep_dataset = True

random.seed(30)

def prep_train_test(train_path, test_path, options: dict):
    init_data_matrix = True
    TrainSet = {}
    class_label_train = []
    TestSet = {}
    class_label_test = []
    test_file = []

    dims = options['dims'] #either a tuple for downsampling or an integer for pca

    for folder in os.listdir(train_path):
        init_class_matrix = True
        if folder in IGNORE_FILES:
            continue
        class_folder = train_path + folder + os.sep
        class_vector = os.listdir(class_folder)

        for img_file in class_vector:
            if img_file in IGNORE_FILES:
                continue
            class_label_train.append(folder)
            img_path = train_path + folder + os.sep + img_file # absolute path to image
            X_orig = cv2.imread(img_path)

            try: # convert to greyscale if image is rgb
                X_orig.shape[2]
                X_orig = rgb2gray(X_orig)
            except IndexError:
                pass

            if options['feature_selection'] == 'downsampling':
                X = cv2.resize(X_orig, dims, interpolation = cv2.INTER_AREA)
            else:
                X = X_orig

            X = X.reshape(-1, 1)
            X = normalize_data_column(X)

            if init_class_matrix:
                D_c = X # initialize data matrix
                init_class_matrix = False
            else:
                D_c = np.hstack((D_c, X))

        if init_data_matrix: # this will run the first time
            D_train = D_c
            init_data_matrix = False
        else:
            D_train = np.hstack((D_train, D_c))

    # Now process test data
    init_data_matrix = True

    for folder in os.listdir(test_path):
        init_class_matrix = True
        if folder in IGNORE_FILES:
            continue
        class_folder = test_path + folder + os.sep
        class_vector = os.listdir(class_folder)

        for img_file in class_vector:
            if img_file in IGNORE_FILES:
                continue
            class_label_test.append(folder)
            img_path = test_path + folder + os.sep + img_file  # absolute path to image
            test_file.append(img_path)
            X_orig = cv2.imread(img_path)

            try:  # convert to greyscale if image is rgb
                X_orig.shape[2]
                X_orig = rgb2gray(X_orig)
            except IndexError:
                pass

            if options['feature_selection'] == 'downsampling':
                X = cv2.resize(X_orig, dims, interpolation = cv2.INTER_AREA)
            else:
                X = X_orig

            X = X.reshape(-1, 1)
            X = normalize_data_column(X)

            if init_class_matrix:
                D_c = X  # initialize data matrix
                init_class_matrix = False
            else:
                D_c = np.hstack((D_c, X))

        if init_data_matrix:  # this will run the first time
            D_test = D_c
            init_data_matrix = False
        else:
            D_test = np.hstack((D_test, D_c))

    if options['feature_selection'] == 'pca':
        D_train, D_test = pca_dim_reduction(D_train, D_test, dims)

    TrainSet['X'] = D_train
    TrainSet['y'] = np.array(class_label_train)
    TestSet['X'] = D_test
    TestSet['y'] = np.array(class_label_test)
    TestSet['files'] = test_file

    return TrainSet, TestSet

def normalize_data_column(img_matrix):
    normalized_img = img_matrix / np.sqrt(np.sum(img_matrix ** 2))
    return normalized_img

def pca_dim_reduction(train_imgs, test_imgs, n_features):
    img_matrix_train = train_imgs.transpose()
    img_matrix_test = test_imgs.transpose()
    pca = PCA(n_components=n_features, svd_solver='randomized', whiten=True).fit(img_matrix_train)
    pca.fit(img_matrix_train)
    resized_matrix_train = pca.transform(img_matrix_train)
    resized_matrix_test = pca.transform(img_matrix_test)
    resized_matrix_train = resized_matrix_train.transpose()
    resized_matrix_test = resized_matrix_test.transpose()

    return resized_matrix_train, resized_matrix_test

def split_train_test(data_folder):

    if not os.path.isdir(os.path.join(data_folder, "train")):
        print("Train folder with data does not exist, exiting..")
        return

    test_path = os.path.join(data_folder, "test")
    train_path = os.path.join(data_folder, "train")

    if not os.path.isdir(test_path):
        os.mkdir(test_path)

    for dir in os.listdir(train_path): #list of subjects
        i = 0
        if dir in IGNORE_FILES:
            continue

        if not os.path.isdir(os.path.join(test_path, dir)): # create folder for each subjet in test folder
            os.mkdir(os.path.join(test_path, dir))

        subject_path = os.path.join(train_path, dir)

        for file in os.listdir(subject_path):

            l = len(os.listdir(subject_path))
            i += 1

            if l > 5:
                if int(len(os.listdir(subject_path)) * 0.3) % i == 0:
                    k = random.choice(os.listdir(os.path.join(train_path, dir)))
                    #print(f"Sample chosen from {dir}, is {k}, num files: {l}")
                    #print(os.path.join(train_path, dir, k), os.path.join(test_path, dir, k))
                    os.rename(os.path.join(train_path, dir, k),
                              os.path.join(test_path, dir, k))

            elif l <= 5:
                if i == 1: ## Sample a single test file (should optimally have more training data than this, so one test sample is sufficient)
                    k = random.choice(os.listdir(os.path.join(train_path, dir)))
                    #print(f"Sample chosen from {dir}, is {k}, num files: {l}")
                    #print(os.path.join(train_path, dir, k), os.path.join(test_path, dir, k))

                    os.rename(os.path.join(train_path, dir, k),
                              os.path.join(test_path, dir, k))

def run_extract_pipeline(videos_folder, dest_folder, train: bool):
    print("Running DMD and Face Detection on input videos")

    if train:
        data_path = parse_videos(videos_folder, dest_folder, train_folder=True)
        process_DMD_snapshots(data_path) # Absolute path to where data resides

        breakpoint()
        ### Split train to test dataset ###
        split_train_test(dest_folder)  # Takes input root folder

    else:
        data_path = parse_videos(videos_folder, dest_folder, train_folder=False) # To test existing database of faces against a recorded video
        process_DMD_snapshots(data_path)                                          # that was not in training

    return data_path

def get_parser():
    parser = argparse.ArgumentParser(description='Specify how you want to run the Object Detection pipeline.')
    parser.add_argument('--train', type=str, required=True, help='Enter y/n for running full pipeline, creating training and test set from folder')
    parser.add_argument('--extract_new', type=str, required=True, help='Enter y/n to run full pipeline on new data against trained model')
    parser.add_argument('--run_new', type=str, required=True, help='Enter y/n for running only src on extracted new dataset, (n) for trained dataset')
    parser.add_argument('--video_folder', required=True, type=str, help='Name of video folder located in project root folder')

    return parser

def main(args=None):

    """

    Main entrypoint

    """

    parser = get_parser()
    args = parser.parse_args(args)

    data_folder = 'Data' # This must exist in project root folder - all processed data is sent here
    video_folder = args.video_folder
    src = os.path.join(os.getcwd(), video_folder)
    dest = os.path.join(os.getcwd(), data_folder)

    if args.train == 'y':
        train_path = run_extract_pipeline(src, dest, train=True) + os.sep
        test_path = os.path.join(dest, 'test') + os.sep

    if args.extract_new == 'y':
        train_path = os.path.join(dest, 'train') + os.sep
        dest = dest + os.sep + 'NewVideos' + os.sep # dedicated folder for new incoming videos to be tested on existing training data
        test_path = run_extract_pipeline(src, dest, train=False)

    if args.train == 'n' and args.extract_new == 'n' and args.run_new == 'y':
        train_path = os.path.join(dest, 'train') + os.sep
        test_path = os.path.join(os.getcwd(), 'Data/NewVideos') + os.sep

    elif args.train == 'n' and args.extract_new == 'n' and args.run_new == 'n':
        train_path = os.path.join(dest, 'train') + os.sep
        test_path = os.path.join(dest, 'test') + os.sep

    ### Feature selection option ###
    #options = {'feature_selection': 'downsampling', 'dims': shape} # feature selection can be wither pca (eigenfaces) or downsampling
    options = {'feature_selection': 'pca', 'dims': 18}

    TrainSet, TestSet = prep_train_test(train_path, test_path, options)

    ### Parameters for src algorithm ###
    num_classes = len(set(TrainSet['y']))
    num_test_samples = len(list(TestSet['y']))
    sigma = 0.00001
    thresh_certainty = 0.75 # Threshold for how "certain" the src algorithm should be when predicting the class.
                           # Certainty in the prediction that falls below this threshold is discarded
                           # Increasing this value will lead to stricter predictions. Put zero for no threshold

    print(f"Running SRC classifier with certainty threshold: {thresh_certainty}, and feature selection: {options['feature_selection']}")

    rec_rate, failed = src_algorithm(TrainSet, TestSet, num_classes, num_test_samples, sigma, thresh_certainty)
    print("Failed to classify images: \n")
    for f in failed:
        print(f)

if __name__ == '__main__':
    main()