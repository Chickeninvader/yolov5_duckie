import torch
from matplotlib import pyplot as plt
from PIL import Image

# Load the trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp8/weights/best.pt')

# Run inference on a custom image
img_path = "/Users/khoavo2003/Desktop/paper_work/duckietown/object_detection_duckietown/dataset/duckietown_dataset/images/frame_000002.png"  # Replace with the path to your image
results = model(img_path)

# Print results
results.print()  # Print detected objects and their confidence scores

# Display image with predictions
results.show()  # Opens a window with the annotated image

# Save predictions
# results.save(save_dir='runs/detect/exp_custom')  # Saves predictions to specified folder

# Optionally access individual predictions
detections = results.pandas().xyxy[0]  # Bounding boxes and confidence as a Pandas DataFrame
print(detections)
