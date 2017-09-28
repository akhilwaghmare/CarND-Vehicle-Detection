from helpers import *
from scipy.ndimage.measurements import label

car_classifier = LinearSVC()
windows = []

bounding_boxes = []

def setup():
    global car_classifier, windows
    car_classifier = classifier()
    windows = find_windows(mpimg.imread('test_images/test1.jpg'))
    print("Setup complete.")

def image_pipeline(image):
    global car_classifier, bounding_boxes, windows
    orig_output_image = np.copy(image)
    output_image = orig_output_image.astype(np.float32)/255

    car_windows = search_windows(output_image, windows, car_classifier)
    bounding_boxes.append(car_windows)

    box_image = draw_boxes(orig_output_image, car_windows)

    return box_image

def video_pipeline(image):
    global car_classifier, bounding_boxes, windows
    orig_output_image = np.copy(image)
    output_image = orig_output_image.astype(np.float32)/255

    car_windows = search_windows(output_image, windows, car_classifier)
    bounding_boxes.append(car_windows)

    recent_frames = 20
    if len(bounding_boxes) < recent_frames + 1:
        recent_frames = len(bounding_boxes) - 1

    frame_heatmap = np.zeros_like(image[:,:,0])
    frame_heatmap = generate_heatmap(bounding_boxes, recent_frames, 5, frame_heatmap)

    labels = label(frame_heatmap)
    box_image = draw_bounding_box(image, labels)

    return box_image

setup()
output = image_pipeline(mpimg.imread('test_images/test1.jpg'))
plt.imshow(output)
plt.show()
