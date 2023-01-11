import os
import supervisely as sly
import cv2

# in this directory stored 7 copy of one video with different frame rotations
videos_dir = "./rotated"

new_videos_dir = "./correct_orientation"
fourcc = cv2.VideoWriter_fourcc(*"MP4V")

if sly.fs.dir_exists(new_videos_dir):
    sly.fs.clean_dir(new_videos_dir)
else:
    sly.fs.mkdir(new_videos_dir)

frames_set = set()  # for test at the end

for file in os.listdir(videos_dir):
    new_path = os.path.join(new_videos_dir, f"new-{file}")
    filepath = os.path.join(videos_dir, file)

    cap = cv2.VideoCapture(filepath)

    # set CAP_PROP_ORIENTATION_AUTO flag for VideoCapture
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

    frameSize = (int(cap.get(3)), int(cap.get(4)))
    fps = cap.get(5)

    frames_set.add(frameSize)  # for test at the end

    writer = cv2.VideoWriter(new_path, fourcc, fps, frameSize)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        writer.write(frame)

    # Release everything if job is finished
    writer.release()
    cap.release()
    cv2.destroyAllWindows()

# should be only 2 shape sizes options: 1280x720 or 720x1280
assert len(frames_set) == 2
