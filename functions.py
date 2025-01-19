import cv2
import mediapipe as mp
import numpy as np
from random import randint
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load object detection labels
labels = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def resize_with_aspect_ratio(image, width=None, height=None):
    """
    Resizes an image while keeping the aspect ratio.

    :param image: Input image
    :param width: Desired width (optional)
    :param height: Desired height (optional)
    :return: Resized image
    """
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        # Calculate the ratio to scale by height
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        # Calculate the ratio to scale by width
        r = width / float(w)
        dim = (width, int(h * r))

    # Resize the image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


def predict_digit(img_path):
    """
    Predict the digit in the input image.

    :param img_path: Image path
    :return: Predicted digit
    """
    # Load model
    digit_model = load_model("models/digit_recognition_model.keras")

    # Pre-process the image data
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
    img_normalized = img_resized / 255.0
    img_array = np.expand_dims(
        img_normalized, axis=-1
    )  # Add the channel dimension (grayscale)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension for

    # Make the prediction
    prediction = digit_model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    return predicted_digit


def detect_objects(img_path):
    """
    Perform object detection on the input image.

    :param img_path: Image path
    :return: Output image with detections (if any)
    """
    model_dir = "models/faster_rcnn_inception_v2_coco_2018_01_28/saved_model"
    model = tf.saved_model.load(model_dir)
    object_detection_model = model.signatures["serving_default"]

    image = np.array(cv2.imread(img_path))
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    detection = object_detection_model(input_tensor)

    # Parse the detection results
    boxes = detection["detection_boxes"].numpy()
    classes = detection["detection_classes"].numpy().astype(int)
    scores = detection["detection_scores"].numpy()

    for i in range(classes.shape[1]):
        class_id = int(classes[0, i])
        score = scores[0, i]
        if score > 0.5:  # Confidence threshold
            h, w, _ = image.shape
            ymin, xmin, ymax, xmax = boxes[0, i]

            # Convert normalized coordinates to image coordinates
            xmin = int(xmin * w)
            xmax = int(xmax * w)
            ymin = int(ymin * h)
            ymax = int(ymax * h)

            # Draw the bounding box and display labels
            class_name = labels[class_id]
            random_color = (randint(0, 256), randint(0, 256), randint(0, 256))
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), random_color, 2)
            label = f"Class: {class_name}, Score: {score:.2f}"

            # Calculate text size and add background rectangle for clarity
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            text_x = xmin
            text_y = ymin - 10 if ymin - 10 > 10 else ymin + 20

            # Draw background rectangle for text
            cv2.rectangle(
                image,
                (text_x, text_y - text_height - 5),
                (text_x + text_width, text_y + baseline),
                random_color,
                -1,
            )

            # Put the text on the image
            cv2.putText(
                image,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
    image = resize_with_aspect_ratio(image, height=300)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def face_keypoint_detection(img_path):
    """
    Locates faces and facial features within a frame.

    :param img_path: Image path
    :return: Output image with detections (if any)
    """

    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )

    image = cv2.imread(img_path)
    image = resize_with_aspect_ratio(image, height=300)

    if image is None:
        print(f"Error: Unable to load image from {img_path}")
    else:
        # Convert the image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Perform face detection
        results = face_detection.process(rgb_image)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = (
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih),
                )

                # Draw bounding box on the image
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Draw keypoints on the image
                for keypoint in detection.location_data.relative_keypoints:
                    keypoint_x = int(keypoint.x * iw)
                    keypoint_y = int(keypoint.y * ih)
                    cv2.circle(image, (keypoint_x, keypoint_y), 1, (0, 0, 255), -1)

        else:
            print("No face detected in the image.")

    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
