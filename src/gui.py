import tkinter as tk
from functools import partial
from pathlib import Path
from tkinter import ttk
from tkinter.filedialog import askopenfilename

import PIL
import cv2
from PIL import ImageTk, Image

MAX_EXAMPLE_VIDS = 4


def explore_examples(video_dir: str, target_shape: tuple[int, int] = (120, 120)):
    video_dir = Path(video_dir)
    video_filenames = []
    video_list = []

    if video_dir.is_dir():
        video_filenames = list(video_dir.glob("*.mp4"))

    i = 0

    for video_filename in video_filenames:

        if i >= MAX_EXAMPLE_VIDS:
            break

        i += 1

        frame = None
        cap = cv2.VideoCapture(str(video_filename.absolute()))
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        if cap.isOpened() and video_length > 0:
            # FIXME(11jolek11): sample thumbnail from random frame not from first frame!!!
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

    print(f"Found {len(video_list)} example videos")

    return video_list


def play_video(video_path: str):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")

        # Read until video is completed
    while cap.isOpened():

        ret, frame = cap.read()
        if ret:
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
        self.examples_dir = Path()

        self.labels_images = []
        self.preview_video_handlers = []

        self._temp_original_img_frame = []
        self._temp_recon_img_frame = []

        self.results = []

        self._canvas = None
        self._scrollbar = None
        self._scrollable_frame = None

        self.run_btn = ttk.Button()

        self.run_action = None

    def set_examples_dir(self, examples_dir: str):
        if Path(examples_dir).is_dir():
            self.examples_dir = Path(examples_dir)
        else:
            print("Invalid examples directory")

    def get_root(self):
        return self._root

    def assign_action_to_play_button(self, funct):
        print("Assigning action to play button")
        self.run_btn.configure(command=funct)

    def add_action(self, funct):
        self.run_action = funct

    def assign_action_with_filename_to_play_button(self):
        print("Assigning action to play button")
        self.run_btn.configure(command=self.run_action(self._filepath))

    def _create_result_frame(self, parent_frame, part_name: str, original_frame: PIL.Image, recon_frame: PIL.Image,
                             state: str):
        frame = ttk.Frame(parent_frame)
        part_name_label = ttk.Label(frame, text=part_name)
        state_label = ttk.Label(frame, text=state)

        original_img_frame = ImageTk.PhotoImage(original_frame)
        self._temp_original_img_frame.append(original_img_frame)
        original_img_label = ttk.Label(frame, image=original_img_frame)

        recon_img_frame = ImageTk.PhotoImage(recon_frame)
        self._temp_recon_img_frame.append(recon_img_frame)
        recon_img_label = ttk.Label(frame, image=recon_img_frame)

        part_name_label.pack(side=tk.LEFT)
        original_img_label.pack(side=tk.LEFT)
        recon_img_label.pack(side=tk.LEFT)
        state_label.pack(side=tk.LEFT)

        return frame

    def push_record_on_scroll(self, part_name: str, original_frame: PIL.Image, recon_frame: PIL.Image, state: str):
        frame = self._create_result_frame(self._scrollable_frame, part_name, original_frame, recon_frame, state)
        self.results.append(frame)
        frame.pack()
        self._root.update()

    def ask_file_path(self):
        choice = askopenfilename()
        self._filepath = Path(choice)
        print("Chosen filename: ", str(self._filepath))
        self.assign_action_with_filename_to_play_button()

    def create_run_frame(self, parent_frame):
        frame = ttk.Frame(parent_frame, width=200, height=200)
        frame.columnconfigure(0, weight=2)
        frame.columnconfigure(1, weight=1)

        file_upload_btn = ttk.Button(frame, text="Upload", command=self.ask_file_path)
        file_upload_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.run_btn = ttk.Button(frame, text="Run")
        self.run_btn.pack(side=tk.LEFT, fill=tk.X)

        return frame

    def create_example_videos_frame(self, parent_frame):
        frame = ttk.Frame(parent_frame, width=150, height=150)
        labels = []

        examples = explore_examples(str(self.examples_dir.absolute()))

        welcome_label = ttk.Label(frame, text="Wybierz video z przykładów")
        welcome_label.pack(side=tk.TOP, fill=tk.X, expand=True)

        for video in examples:
            subframe = ttk.Frame(frame, width=150, height=120)

            img = ImageTk.PhotoImage(video["thumbnail"])
            label = ttk.Label(subframe, image=img)
            label.pack(side=tk.LEFT, expand=True)
            labels.append(label)
            self.labels_images.append(img)

            print(str(video["file"].absolute()))

            preview_button = ttk.Button(subframe, text='Preview',
                                        command=partial(play_video, str(video["file"].absolute())))

            preview_button.pack(side=tk.RIGHT)

            subframe.pack()

        return frame

    def create_presentation_frame(self, parent_frame):
        frame = ttk.Frame(parent_frame)

        legend_frame = ttk.Frame(frame)

        legend_names = ["Część", "Orginał", "Rekonstrukcja", "Status"]
        legend_labels = []

        for legend in legend_names:
            temp_label = ttk.Label(legend_frame, text=legend)
            temp_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
            legend_labels.append(temp_label)

        legend_frame.pack()

        self._canvas = tk.Canvas(frame)
        self._scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self._canvas.yview)
        self._scrollable_frame = ttk.Frame(self._canvas)

        self._scrollable_frame.bind(
            "<Configure>",
            lambda e: self._canvas.configure(
                scrollregion=self._canvas.bbox("all")
            )
        )

        self._canvas.create_window((0, 0), window=self._scrollable_frame, anchor="nw")

        self._canvas.configure(yscrollcommand=self._scrollbar.set)

        for result in self.results:
            result.pack()

        # frame.pack()
        self._canvas.pack(side="left", fill="both", expand=True)
        self._scrollbar.pack(side="right", fill="y")

        return frame

    def build(self):
        user_frame = tk.Frame(self._root)

        self.create_run_frame(user_frame).pack(anchor=tk.NW)
        frame = self.create_example_videos_frame(user_frame)
        frame.pack(anchor=tk.W, expand=True)

        user_frame.pack(side=tk.LEFT)

        self.create_presentation_frame(self._root).pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def run(self):
        self._root.mainloop()


if __name__ == '__main__':

    def hello(filename):
        return lambda: print("hello world {}".format(filename))

    gui = Gui()
    gui.set_examples_dir("C:/Users/dabro/PycharmProjects/scientificProject/data/videos/Normal-001")
    gui.build()
    gui.add_action(hello)
    gui.assign_action_with_filename_to_play_button()

    gui.run()
