import urllib.request
import os
import repnet

# CONSTANTS
PATH_TO_TRAINED_MODEL_DIR = "./model"
VIDEO_PATH = "./data/test.mp4"
TEMP_DIR = "./tmp"

## PARAMS

# FPS while recording video from webcam.
WEBCAM_FPS = 16

# Time in seconds to record video on webcam.
RECORDING_TIME_IN_SECONDS = 8

# Threshold to consider periodicity in entire video.
THRESHOLD = 0.2

# Threshold to consider periodicity for individual frames in video.
WITHIN_PERIOD_THRESHOLD = 0.5

# Use this setting for better results when it is
# known action is repeating at constant speed.
CONSTANT_SPEED = False

# Use median filtering in time to ignore noisy frames.
MEDIAN_FILTER = True

# Use this setting for better results when it is
# known the entire video is periodic/reapeating and
# has no aperiodic frames.
FULLY_PERIODIC = False

# Plot score in visualization video.
PLOT_SCORE = False

# Visualization video's FPS.
VIZ_FPS = 30


def load_trained_repnet():
    download_urls = [
        "https://storage.googleapis.com/repnet_ckpt/checkpoint",
        "https://storage.googleapis.com/repnet_ckpt/ckpt-88.data-00000-of-00002",
        "https://storage.googleapis.com/repnet_ckpt/ckpt-88.data-00001-of-00002",
        "https://storage.googleapis.com/repnet_ckpt/ckpt-88.index",
    ]

    if not os.path.exists(PATH_TO_TRAINED_MODEL_DIR):
        os.makedirs(PATH_TO_TRAINED_MODEL_DIR)

    # download all checkpoint files from google storage
    for url in download_urls:
        file_path = os.path.join(PATH_TO_TRAINED_MODEL_DIR, url.split("/")[-1])
        if not os.path.exists(file_path):
            print("Downloading %s" % url)
            urllib.request.urlretrieve(url, filename=file_path)


def run_repnet(model, imgs, vid_fps):
    print("Running RepNet...")
    (
        pred_period,
        pred_score,
        within_period,
        per_frame_counts,
        chosen_stride,
    ) = repnet.get_counts(
        model,
        imgs,
        strides=[1, 2, 3, 4],
        batch_size=20,
        threshold=THRESHOLD,
        within_period_threshold=WITHIN_PERIOD_THRESHOLD,
        constant_speed=CONSTANT_SPEED,
        median_filter=MEDIAN_FILTER,
        fully_periodic=FULLY_PERIODIC,
    )
    print("Visualizing results...")
    repnet.viz_reps(
        imgs,
        per_frame_counts,
        pred_score,
        interval=1000 / VIZ_FPS,
        plot_score=PLOT_SCORE,
    )

    return (
        pred_period,
        pred_score,
        within_period,
        per_frame_counts,
        chosen_stride,
    )


def create_count_video(imgs, vid_fps, model_output):
    (_, pred_score, within_period, per_frame_counts, _,) = model_output

    # Debugging video showing scores, per-frame frequency prediction and
    # within_period scores.
    tmp_vid_location = os.path.join(TEMP_DIR, "debug_video.mp4")
    if os.path.exists(tmp_vid_location):
        os.remove(tmp_vid_location)
    elif not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    repnet.create_count_video(
        imgs,
        per_frame_counts,
        within_period,
        score=pred_score,
        fps=vid_fps,
        output_file=tmp_vid_location,
        delay=1000 / VIZ_FPS,
        plot_count=True,
        plot_within_period=True,
        plot_score=True,
    )
    repnet.show_video(tmp_vid_location)


def main():
    load_trained_repnet()
    if not os.path.exists(VIDEO_PATH):
        print("Video does not exist")
        exit(1)
    imgs, vid_fps = repnet.read_video(VIDEO_PATH)
    model_output = run_repnet(repnet.RepNet(), imgs, vid_fps)
    create_count_video(imgs, vid_fps, model_output)


if __name__ == "__main__":
    main()
