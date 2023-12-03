import Augmentor


p = Augmentor.Pipeline(
    "C:/Users/dabro/PycharmProjects/scientificProject/notebooks/CarPartsDatasetExperimentDir",
    "C:/Users/dabro/PycharmProjects/scientificProject/notebooks/CarPartsDatasetExperimentDir/output"
                       )
p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.flip_left_right(probability=0.6)
p.flip_top_bottom(probability=0.5)
p.skew_tilt(probability=0.75)
p.skew_corner(probability=0.4)
p.rotate_random_90(probability=0.8)
# p.gaussian_distortion(probability=0.3)
p.sample(1000)
p.process()
