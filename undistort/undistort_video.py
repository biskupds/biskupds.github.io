import os
import glob
import time

import numpy as np
import cv2
import click
from tqdm import tqdm


def path_contains_cam(filepath):
    return (
        os.path.splitext(filepath)[0].endswith("cam1")
        or os.path.splitext(filepath)[0].endswith("cam2")
        or os.path.splitext(filepath)[0].endswith("cam3")
        or os.path.splitext(filepath)[0].endswith("cam4")
        or os.path.splitext(filepath)[0].endswith("cam5")
    )


def get_cam_params(filepath):
    mtx = np.zeros((3, 3))
    dist = np.zeros(5)
    # using file name we first import mtx, dist variables from correct csv
    with open(
        glob.glob("./csv/*cam" + os.path.splitext(filepath)[0][-1] + ".csv")[0]
    ) as file:
        for i in range(0, 3):
            mtx[i] = np.fromstring(file.readline(), dtype=np.float64, sep=",")
        dist = np.fromstring(file.readline(), dtype=np.float64, sep=",")
    return mtx, dist


def create_videowriter(fps, filepath, width, height):
    # Get the video's width, height, and frames per second (fps)
    output_file = "undistorted-" + os.path.splitext(filepath)[0][-4:] + ".mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Video codec
    writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    return writer, output_file


def undistort_frame(frame, mtx, dist, size):
    # get parameter
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, size, 1, size)
    # undistort
    undistorted_frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    cropped_frame = undistorted_frame[y : y + h, x : x + w]
    return cropped_frame, roi


@click.command()
@click.option(
    "--filepath",
    prompt="Enter complete file path to video",
    help='be sure that the video title is formatted "...cam1.mov" or "...cam2.mp4"',
)
def undistort_video(filepath):
    """reads and undistorts video or image based on calibrated parameters custom to each camera. returns file path to the newly saved undistorted output"""
    # checking if file is named correctly
    if not path_contains_cam(filepath):
        click.echo("Can't figure out which camera this is from")
        exit()
    # checking if video exists and can be read
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        click.echo("Error opening video file")
        exit()
    # Video configurations
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_size = (video_width, video_height)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    mtx, dist = get_cam_params(filepath)

    ret, frame = cap.read()
    _, (_, _, undistorted_width, undistorted_height) = undistort_frame(
        frame, mtx, dist, video_size
    )

    # iterate through frames and apply undistortion
    writer, output_file = create_videowriter(
        video_fps, filepath, undistorted_width, undistorted_height
    )

    pbar = tqdm(total=num_frames)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        pbar.update(1)

        # cv2.imshow("original", frame[::4, ::4, :])

        undistorted_frame, _ = undistort_frame(frame, mtx, dist, video_size)

        # cv2.imshow("undistorted", undistorted_frame)
        # cv2.waitKey(1)

        writer.write(undistorted_frame.astype("uint8"))

    # save file and return file path
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    out = os.path.join(os.getcwd(), output_file)
    click.echo("Success! Your file can be found at: " + out)
    return out


if __name__ == "__main__":
    undistort_video()
