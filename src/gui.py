import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from pathlib import Path
from PIL import ImageTk, Image
from functools import partial
import copy

import cv2
import numpy as np


MAX_EXAMPLE_VIDS = 4  # TODO(11jolek11): REMOVE


def explore_examples(video_dir: str, target_shape: tuple[int, int] = (120, 120)):
    video_dir = Path(video_dir)
    video_filenames = []
    video_list = []

    if video_dir.is_dir():
        video_filenames = list(video_dir.glob("*.mp4"))

    i = 0  # TODO(11jolek11): REMOVE

    for video_filename in video_filenames:

        if i >= MAX_EXAMPLE_VIDS:
            break

        i += 1

        frame = None
        cap = cv2.VideoCapture(str(video_filename.absolute()))
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        if cap.isOpened() and video_length > 0:
            # FIXME(11jolek11): sample thumbnail from random frame not from first!!!
            for _ in range(video_length):
                flag, frame = cap.read()
                if flag:
                    break

        if frame.shape != target_shape:
            frame = cv2.resize(frame, target_shape)

        video = dict()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = Image.fromarray(frame)

        video["file"] = video_filename
        video["thumbnail"] = frame

        video_list.append(video)

    print(f"Found {len(video_list)} videos")

    return video_list


def play_video(video_path: str):
    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(video_path)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video file")

        # Read until video is completed
    while cap.isOpened():

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release
    # the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


class Gui:
    def __init__(self):
        self._root = tk.Tk()
        self._root.geometry("700x700")
        self._filepath = Path()
        self._temp_images = None
        self._temp_preview_handlers = None

    def ask_file_path(self):
        self._filepath = Path(askopenfilename())

    def create_run_frame(self, parent_frame):
        frame = ttk.Frame(parent_frame, width=200, height=200)
        frame.columnconfigure(0, weight=2)
        frame.columnconfigure(1, weight=1)

        file_upload_btn = ttk.Button(frame, text="Upload", command=self.ask_file_path)
        file_upload_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

        run_btn = ttk.Button(frame, text="Run")
        run_btn.pack(side=tk.LEFT, fill=tk.X)

        return frame

    def create_presentation_frame(self, parent_frame):
        pass

    def create_example_videos_frame(self, parent_frame):
        frame = ttk.Frame(parent_frame, width=150, height=150)
        labels = []
        labels_images = []
        preview_video_handlers = []

        examples = explore_examples("C:/Users/dabro/PycharmProjects/scientificProject/data/videos/Normal-001")

        welcome_label = ttk.Label(frame, text="Wybierz video z przykładów")
        welcome_label.pack(side=tk.TOP, fill=tk.X, expand=True)

        for video in examples:
            subframe = ttk.Frame(frame, width=150, height=120)

            img = ImageTk.PhotoImage(video["thumbnail"])
            label = ttk.Label(subframe, image=img)
            label.pack(side=tk.LEFT, expand=True)
            labels.append(label)
            labels_images.append(img)

            print(str(video["file"].absolute()))

            preview_button = ttk.Button(subframe, text='Preview', command=partial(play_video, str(video["file"].absolute())))

            preview_button.pack(side=tk.RIGHT)

            subframe.pack()

        return frame, labels_images, preview_video_handlers

    def build(self):
        # self._root.columnconfigure(0)
        self.create_run_frame(self._root).pack(anchor=tk.NW)
        frame, self._temp_images, self._temp_preview_handlers = self.create_example_videos_frame(self._root)
        frame.pack(anchor=tk.W, expand=True)

    def run(self):
        self._root.mainloop()


if __name__ == '__main__':
    gui = Gui()
    gui.build()
    gui.run()

    # explore_examples("C:/Users/dabro/PycharmProjects/scientificProject/data/videos/Normal-001")
