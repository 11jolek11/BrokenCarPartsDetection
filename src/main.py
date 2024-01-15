import PIL.Image
import numpy as np

from .demo import Demo, DisHandle
from .gui import Gui
from .video import VideoFrameExtract
import cv2


class Main:
    def __init__(self):
        self.gui = Gui()
        self.gui.set_examples_dir("./example_videos")
        self.gui.build()
        self.demo = Demo()
        self.video_reader = VideoFrameExtract()
        self.dishandle = DisHandle()

        self.runner = None

    def get_file_and_run(self, file_name):
        def internal(file):
            self.video_reader.read(file)
            frames, _ = self.video_reader.select_frames(10)

            for frame_no, frame in enumerate(frames):
                original, reconstructed = self.demo.forward(frame)
                for part in reconstructed.keys():
                    cutoff = PIL.Image.fromarray(reconstructed[part]["cutoff"].astype(np.uint8))
                    recon = reconstructed[part]["recon"]

                    recon_np = np.array(recon)
                    transformed_part_np = np.array(reconstructed[part]["transformed_part"])

                    diff_img = cv2.subtract(recon_np, transformed_part_np)
                    diff = int(np.argwhere(diff_img > 0).shape[0])
                    classification = self.dishandle.classify([part, diff])

                    class_label = ""

                    if classification > 0.5:
                        class_label = "OK"
                    else:
                        class_label = "BROKEN"

                    self.gui.push_record_on_scroll(part, cutoff, recon, str(frame_no), class_label)

        return lambda: internal(file_name)

    def run(self):
        self.gui.add_action(self.get_file_and_run)
        self.gui.assign_action_with_filename_to_play_button()
        self.gui.run()


if __name__ == "__main__":
    main_window = Main()
    main_window.run()
