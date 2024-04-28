import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

# Path to the input image, change it to your specific input image file
IMAGE_PATH = 'path_to_your_image.jpg'

# Paths to the model configuration and pre-trained model weights
# You should download these from the TensorFlow model zoo
CONFIG_PATH = 'path_to_model_config.pbtxt'
CHECKPOINT_PATH = 'path_to_checkpoint.ckpt'

# Load the pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(CHECKPOINT_PATH).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

# Load image
image_np = cv2.imread(IMAGE_PATH)
image_np_expanded = np.expand_dims(image_np, axis=0)

# Perform the detection
input_tensor = tf.convert_to_tensor(image_np_expanded, dtype=tf.float32)
detections = detect_fn(input_tensor)

# Visualize detection results
label_map_path = configs['eval_input_config'].label_map_path
category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)

viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np,
    detections['detection_boxes'][0].numpy(),
    (detections['detection_classes'][0].numpy() + 1).astype(int),
    detections['detection_scores'][0].numpy(),
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=200,
    min_score_thresh=.30,
    agnostic_mode=False)

cv2.imshow('Object Detection', image_np)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Count batteries (assuming battery has a specific class id, e.g., 'battery': 1)
n_batteries = np.sum((detections['detection_classes'][0].numpy() + 1).astype(int) == 1)
print('Number of batteries detected:', n_batteries)
