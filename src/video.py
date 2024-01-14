from pathlib import Path

import cv2


class VideoFrameExtract:
    def __init__(self):
        self.video = None
        self.raw_path = Path()
        self.frames = []
        self.video_meta = {}

    def read(self, raw_path: Path):
        self.raw_path = Path(raw_path)
        self.video = None
        if self.raw_path.is_file():
            self.video = cv2.VideoCapture(str(self.raw_path))
        else:
            raise FileNotFoundError(f"{raw_path} does not exist")

    def select_frames(self, sample_rate: int):
        frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        if sample_rate == 0 or sample_rate > frame_count:
            raise ValueError("Sample rate")

        correct_frame_count = 0

        for f_no in range(0, frame_count, sample_rate):
            self.video.set(cv2.CAP_PROP_POS_FRAMES, f_no)

            ret, frame = self.video.read()

            if ret:
                self.frames.append(frame)
                correct_frame_count += 1

        return self.frames, correct_frame_count


if __name__ == '__main__':
    video_reader = VideoFrameExtract()
    video_reader.read("C:/Users/dabro/PycharmProjects/scientificProject/data/videos/Normal-001/000001.mp4")
    frames, no_of_frames = video_reader.select_frames(10)

    print(f"Frames no.: {no_of_frames}")

    for single_frame in frames:
        cv2.imshow('frame', single_frame)
        cv2.waitKey(0)
