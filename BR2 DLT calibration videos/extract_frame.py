import os
import cv2

def extract_frame(video_file_path, time_in_seconds, cam_num, x_pos):
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
            raise ValueError("could not find video at provided destination")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if (time_in_seconds*fps > total_frames):
        raise ValueError("please ensure that the provided time in seconds is less than the total length of the video in seconds")        
    
    frame_number = 0
    output_file = ""
    while cap.isOpened():
        _, frame = cap.read()
        if not _:
            break
        frame_number += 1
        if frame_number == int(time_in_seconds*fps):
            output_file = f"cam{cam_num}_x{x_pos}.jpg"
            if not cv2.imwrite(output_file, frame):
                print("image could not be saved.")     
                
    return os.path.join(os.getcwd(), output_file)  