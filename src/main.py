import PIL.Image

from gui import Gui
from demo import Demo
from video import VideoFrameExtract
import numpy as np


class Main:
    def __init__(self):
        self.gui = Gui()
        self.gui.set_examples_dir("C:/Users/dabro/PycharmProjects/scientificProject/data/videos/Normal-001")
        self.gui.build()
        self.demo = Demo()
        self.video_reader = VideoFrameExtract()

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

                    self.gui.push_record_on_scroll(part, cutoff, recon, str(frame_no))

        return lambda: internal(file_name)

    def run(self):
        self.gui.add_action(self.get_file_and_run)
        self.gui.assign_action_with_filename_to_play_button()
        self.gui.run()


if __name__ == "__main__":
    main_window = Main()
    main_window.run()
