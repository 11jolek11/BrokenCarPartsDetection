
# BrokenCarPartsDetection
# Overview
This project is designed to automate the detection of damages in vehicles using advanced image processing and machine learning techniques. The system can identify and classify various types of car damages from images, providing a fast and accurate assessment that can be used in various applications such as insurance claims, vehicle inspections, and maintenance services.

## Documentation
### Technology Stack
Programming Language: Python

Framework: PyTorch

Image Processing: OpenCV

Machine Learning: Restricted Boltzmann Machine

### Problem-Solving Journey

Initially, classic image processing methods were employed to build a classifier for detecting damaged parts of vehicles. However, after multiple attempts, these methods did not yield satisfactory results. Consequently, a new approach was adopted: reconstructing parts of the vehicle to identify damage more effectively.

Model Selection
To achieve this, both Variational Autoencoders (VAEs) and Restricted Boltzmann Machines (RBMs) were tested. Ultimately, RBMs were chosen due to their superior performance in reconstructing undamaged parts of the vehicle, allowing for a clearer identification of damaged areas when discrepancies between the reconstruction and the original image were detected.

### Data Management
Data is managed using DVC (Data Version Control), allowing for efficient tracking and versioning of datasets. The following datasets are used in this project:

Training Data: Images of undamaged cars are used to train the model to understand the normal, undamaged condition of a vehicle.
Validation Data: Images of damaged cars are used to validate the model's ability to detect and classify various types of damage.

### Results

Despite efforts, the final results did not meet expectations. The model had difficulty accurately reconstructing damaged areas, likely due to insufficient training caused by limited computational resources and inadequate data. This hindered the model's ability to effectively detect and classify car damages.
