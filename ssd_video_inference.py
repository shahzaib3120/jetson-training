from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys

if len(sys.argv) < 5:
    print('Usage: python run_ssd_example.py <net type> <model path> <label path> <video path>')
    sys.exit(0)

net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]
video_path = sys.argv[4]

class_names = [name.strip() for name in open(label_path).readlines()]

# Create the network based on the specified type
if net_type == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
elif net_type == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd, mb1-ssd-lite, mb2-ssd-lite, or sq-ssd-lite.")
    sys.exit(1)

# Load the model
net.load(model_path)

# Create the appropriate predictor for the network
if net_type == 'vgg16-ssd':
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd':
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd-lite':
    predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'mb2-ssd-lite':
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'sq-ssd-lite':
    predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
else:
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)

# Open the video file
video_capture = cv2.VideoCapture(video_path)
if not video_capture.isOpened():
    print(f"Failed to open video file: {video_path}")
    sys.exit(1)

# Get video properties
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video_capture.get(cv2.CAP_PROP_FPS)

# Set up video writer to save output video
output_video_path = "run_ssd_example_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

print(f"Processing video: {video_path}")

# Process each frame from the video
while True:
    ret, orig_frame = video_capture.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)

    # Predict boxes, labels, and probabilities
    boxes, labels, probs = predictor.predict(rgb_frame, 10, 0.4)

    # Draw bounding boxes and labels on the frame
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        # Convert box coordinates to int
        box = [int(x) for x in box]
        # Draw the bounding box
        cv2.rectangle(orig_frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
        # Prepare the label text
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        # Draw the label text
        cv2.putText(orig_frame, label, (box[0] + 20, box[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # Show the frame with the drawn bounding boxes and labels
    # resize the frame to fit the screen
    show_frame = cv2.resize(orig_frame, (800, 600))
    cv2.imshow('SSD Object Detection', show_frame)

    # Write the frame to the output video
    video_writer.write(orig_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"Video processing completed. The output video is saved as {output_video_path}.")
