import datetime
import time

import cv2

# Open the default camera
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # auto mode
print("Auto Exposure:", cam.get(cv2.CAP_PROP_AUTO_EXPOSURE))

# Set camera resolution (best effort; actual capture size may vary depending on hardware)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)

# Setup video writer
output_size = (512, 512)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
save_path = "videos"
video_filename = datetime.datetime.now().strftime(f"{save_path}/vid_%Y-%m-%d_%H-%M-%S") + ".mp4"
FPS = 3.0
out = cv2.VideoWriter(video_filename, fourcc, FPS, output_size)  # 3 fps
print(f"Recording video to: {video_filename}")

try:
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to get new frame")
            time.sleep(1)
            continue

        # Resize frame to 512x512
        frame_resized = cv2.resize(frame, output_size)

        display_frame = frame_resized
        cv2.imshow("frame", display_frame)

        # Write frame to video file
        out.write(frame_resized)

        delay_ms = int(1000 / FPS)
        c = cv2.waitKey(delay_ms) & 0xFF
        if c in [13, 27] or chr(c) in "qQ":  # Enter, Esc, q, Q
            break
finally:
    cam.release()
    out.release()
    cv2.destroyAllWindows()
    print("Recording stopped, resources released.")
