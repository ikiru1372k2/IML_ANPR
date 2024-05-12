
---

## ANPR (Automatic Number Plate Recognition) Project

This project implements an end-to-end Automatic Number Plate Recognition (ANPR) system using deep learning techniques for object detection and optical character recognition (OCR).

### Project Overview

The project includes the following components:

1. **Data Preparation (`image_csv.ipynb`)**:
   - This notebook is responsible for preparing the dataset for training the object detection model.
   - It reads images from the `data/images/` directory and generates a CSV file (`labels.csv`) containing image file paths and corresponding bounding box annotations for license plate regions.
   - The annotations typically include the coordinates (xmin, xmax, ymin, ymax) of the bounding boxes around license plates.

2. **Object Detection and Model Training (`object_detection.ipynb`)**:
   - In this notebook, the `InceptionResNetV2` model is utilized for transfer learning to detect license plate regions in images.
   - The pre-trained `InceptionResNetV2` model (without the top layers) is loaded from TensorFlow's model zoo and fine-tuned on the annotated dataset (`labels.csv`).
   - The dataset is split into training and validation sets for model training.
   - The model is trained using the Mean Squared Error (MSE) loss function and Adam optimizer to predict bounding box coordinates (xmin, xmax, ymin, ymax) for license plate regions.
   - The trained model is saved as `object_detection_model.h5` in the `models/` directory for later use.

3. **License Plate Detection and OCR (`predict.ipynb`)**:
   - This notebook demonstrates how to use the trained object detection model (`object_detection_model.h5`) to perform license plate detection and OCR on new images.
   - It loads the saved model and uses it to predict bounding boxes for license plate regions in the specified image.
   - Bounding boxes are then used to extract license plate regions from the image.
   - Optical Character Recognition (OCR) is applied to recognize text from the extracted license plate images using Tesseract OCR.
   - The recognized text (license plate number) is displayed as output along with the visual representation of the detected license plate region.

### Usage

1. **Data Preparation**:
   - Open and execute `image_csv.ipynb` to generate the `labels.csv` file based on your dataset.

2. **Model Training**:
   - Open and run `object_detection.ipynb` to train the object detection model using the generated `labels.csv`.
   - Make sure to download and place the pre-trained weights for `InceptionResNetV2` in the `models/` directory.

3. **License Plate Detection and OCR**:
   - Open and execute `predict.ipynb` to load the trained model and perform license plate detection and OCR on new images.
   - Update the `image_path` variable in the notebook to specify the path to the image you want to process for license plate detection and OCR.

### File Structure

```
anpr_project/
│
├── data/
│   ├── images/
│   ├── labels.csv
│
├── models/
│   ├── object_detection_model.h5
│
├── notebooks/
│   ├── image_csv.ipynb
│   ├── object_detection.ipynb
│   ├── predict.ipynb
│
├── README.md
```

### Requirements

- Python 3.x
- Jupyter Notebook
- TensorFlow
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Tesseract OCR

### Note

- Customize the notebook usage based on your dataset and specific requirements.
- Ensure all necessary dependencies are installed before running the notebooks.
- Update the `image_path` variable in `predict.ipynb` with the path to the image you want to process for license plate detection and OCR.

---
