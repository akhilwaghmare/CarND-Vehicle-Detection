import glob
import numpy as np
import cv2
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.externals import joblib

#------------------------------------------------#

#-------------------Constructing SVM Classifier-------------------#

input_scalar = StandardScaler()

def color_conversion(image, color_space):
    conversion_parameter = ''
    if color_space == 'RGB':
        return image
    elif color_space == 'HSV':
        conversion_parameter = cv2.COLOR_RGB2HSV
    elif color_space == 'LUV':
        conversion_parameter = cv2.COLOR_RGB2LUV
    elif color_space == 'HLS':
        conversion_parameter = cv2.COLOR_RGB2HLS
    elif color_space == 'YUV':
        conversion_parameter = cv2.COLOR_RGB2YUV
    elif color_space == 'YCrCb':
        conversion_parameter = cv2.COLOR_RGB2YCrCb
    return cv2.cvtColor(image, conversion_parameter)

def normalize(image):
    return (image - image.mean()) / (image.max() - image.min())

def preprocess(input):
    global input_scalar
    preprocessed_input = input_scalar.transform(input)
    preprocessed_input = normalize(preprocessed_input)
    return preprocessed_input

def get_spatial_features(image, spatial_size):
    features = cv2.resize(image, spatial_size).ravel()
    return features

def get_histogram_features(image, histogram_bins):
    bins_range = (0, 256)

    channel1_histogram = np.histogram(image[:, :, 0], bins=histogram_bins, range=bins_range)
    channel2_histogram = np.histogram(image[:, :, 1], bins=histogram_bins, range=bins_range)
    channel3_histogram = np.histogram(image[:, :, 2], bins=histogram_bins, range=bins_range)

    features = np.concatenate((channel1_histogram[0], channel2_histogram[0], channel3_histogram[0]))
    return features

def get_hog_features(image, orientations, pixels_per_cell, cells_per_block):
    features = hog(image, orientations=orientations, pixels_per_cell=(pixels_per_cell, pixels_per_cell), cells_per_block=(cells_per_block, cells_per_block), transform_sqrt=True, feature_vector=True)
    return features

def extract_features(images, color_space, spatial_size, histogram_bins, orientations, pixels_per_cell, cells_per_block, hog_channel, spatial_features, histogram_features, hog_features):
    features = []
    for image in images:
        feature_image = color_conversion(image, color_space)
        image_features = []

        if spatial_features == True:
            image_features.append(get_spatial_features(feature_image, spatial_size))
        if histogram_features == True:
            image_features.append(get_histogram_features(feature_image, histogram_bins))
        if hog_features == True:
            if hog_channel == 'ALL':
                all_hog_features = []
                for i in range(feature_image.shape[2]):
                    channel_hog_features = get_hog_features(feature_image[:, :, i], orientations, pixels_per_cell, cells_per_block)
                    all_hog_features.append(channel_hog_features)
                all_hog_features = np.ravel(all_hog_features)
            else:
                all_hog_features = get_hog_features(feature_image, orientations, pixels_per_cell, cells_per_block)
            image_features.append(all_hog_features)
        features.append(np.concatenate(image_features))
    return features

def default_extract_features_from_files(image_names):
    # HOG Parameters
    color_space = 'YCrCb'
    orientations = 5
    pixels_per_cell = 8
    cells_per_block = 5
    hog_channel = 'ALL'

    spatial_size = (16, 16)
    histogram_bins = 24
    spatial_features = False
    histogram_features = False
    hog_features = True

    images = []
    for filename in image_names:
        images.append(mpimg.imread(filename))

    return extract_features(images, color_space, spatial_size, histogram_bins, orientations, pixels_per_cell, cells_per_block, hog_channel, spatial_features, histogram_features, hog_features)

def default_extract_features(images):
    # HOG Parameters
    color_space = 'YCrCb'
    orientations = 5
    pixels_per_cell = 8
    cells_per_block = 5
    hog_channel = 'ALL'

    spatial_size = (16, 16)
    histogram_bins = 24
    spatial_features = False
    histogram_features = False
    hog_features = True

    return extract_features(images, color_space, spatial_size, histogram_bins, orientations, pixels_per_cell, cells_per_block, hog_channel, spatial_features, histogram_features, hog_features)

def classifier():
    global input_scalar

    if Path('model.pkl').is_file():
        if Path('scalar.pkl').is_file():
            input_scalar = joblib.load('scalar.pkl')
            return joblib.load('model.pkl')

    # Get training images
    car_images = glob.glob('vehicles/*/*.png')
    not_car_images = glob.glob('non-vehicles/*/*.png')

    car_features = default_extract_features_from_files(car_images)
    not_car_features = default_extract_features_from_files(not_car_images)

    # Features vector to feed into SVM
    X = np.vstack((car_features, not_car_features)).astype(np.float64)
    input_scalar = StandardScaler().fit(X)
    joblib.dump(input_scalar, 'scalar.pkl')
    X = preprocess(X)

    # Label vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(not_car_features))))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svc = LinearSVC()
    svc.fit(X_train, y_train)
    joblib.dump(svc, 'model.pkl')
    return svc

#-------------------Sliding Window Implementation-------------------#

def search_mask(image):
    mask = np.zeros_like(image[:,:,0])
    vertices = np.array([[(700,400), (1000,720), (1280,720), (1280,400)]])
    mask = cv2.fillPoly(mask, vertices, 1)
    return mask

def get_window_size():
    return (64, 64)

def find_windows(image):
    # Parameters
    window_size = get_window_size()
    overlap_size = (0.5, 0.5)
    x_search = [600, 1280]
    y_search = [400, 720]
    mask = search_mask(image)

    del_x = np.int(window_size[0] * (1 - overlap_size[0]))
    del_y = np.int(window_size[1] * (1 - overlap_size[1]))

    x_windows_count = np.int((x_search[1] - x_search[0]) / del_x) - 1
    y_windows_count = np.int((y_search[1] - y_search[0]) / del_y) - 1

    windows = []

    for i in range(x_windows_count):
        for j in range(y_windows_count):
            x_start = i * del_x + x_search[0]
            x_end = x_start + window_size[0]
            y_start = j * del_y + y_search[0]
            y_end = y_start + window_size[1]

            if mask[int(y_start)][int(x_start)] > 0:
                windows.append(((x_start,y_start), (x_end,y_end)))

    return windows

def search_windows(image, windows, classifier):
    car_windows = []

    for window in windows:
        window_image = cv2.resize(image[window[0][1]:window[1][1], window[0][0]:window[1][0]], get_window_size())
        features = default_extract_features([window_image])
        features = preprocess(np.array(features).reshape(1,-1))

        prediction = classifier.predict(features)
        if prediction == 1:
            car_windows.append(window)

    return car_windows

def draw_boxes(image, boxes):
    color = (0,0,255)
    thickness = 6

    output_image = np.copy(image)

    for box in boxes:
        cv2.rectangle(output_image, box[0], box[1], color, thickness)

    return output_image

#-------------------Using heatmaps to remove false positives-------------------#

def add_to_heatmap(heatmap, new_boxes):
    for box in new_boxes:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap

def threshold_heatmap(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap

def generate_heatmap(all_frame_boxes, recent_frames, threshold, heatmap_template):
    frame_heatmap = heatmap_template
    for frame in all_frame_boxes[-recent_frames:]:
        frame_heatmap = add_to_heatmap(frame_heatmap, frame)
    frame_heatmap = threshold_heatmap(frame_heatmap, threshold)
    return frame_heatmap

def generate_heatmaps(all_frame_boxes, recent_frames, threshold, heatmap_template):
    heatmaps = []

    for index, frame_boxes in enumerate(all_frame_boxes):
        frame_index_start = max(index - recent_frames, 0)
        heatmap = heatmap_template
        for frame in all_frame_boxes[frame_index_start:index+1]:
            heatmap = add_to_heatmap(heatmap, frame)
        heatmap = threshold_heatmap(heatmap, threshold)
        heatmaps.append(heatmap)

    return heatmaps

def draw_bounding_box(image, labels):
    for car in range(1, labels[1]+1):
        car_pixels = (labels[0] == car).nonzero()
        car_pixels_y = np.array(car_pixels[0])
        car_pixels_x = np.array(car_pixels[1])
        bounding_box = ((np.min(car_pixels_x), np.min(car_pixels_y)), (np.max(car_pixels_x), np.max(car_pixels_y)))
        cv2.rectangle(image, bounding_box[0], bounding_box[1], (0,0,255), 6)
    return image
