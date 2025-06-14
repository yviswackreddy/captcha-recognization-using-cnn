CAPTCHA Recognition System

 Overview
This project implements a CAPTCHA recognition system using deep learning, specifically a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The system is designed to recognize text-based CAPTCHAs by preprocessing images, segmenting individual characters, and classifying them. It includes robust image preprocessing techniques for noise reduction and character segmentation, achieving high accuracy in automated CAPTCHA solving.

 Features
Image Preprocessing: Converts CAPTCHA images to grayscale, applies adaptive thresholding, and reduces noise using Gaussian blur.
Character Segmentation: Uses contour detection to identify and extract individual characters from CAPTCHA images.
CNN Model: A deep learning model with three convolutional layers, max pooling, and dense layers for character classification.
Data Preparation: Processes a dataset of individual character images for training.
Recognition Pipeline: Combines preprocessing, segmentation, and prediction to recognize full CAPTCHA texts.
Training Script: Trains the model on a user-provided dataset and saves it for future use.

Requirements
- Python 3.8+
- TensorFlow
- OpenCV
- NumPy
- Scikit-learn

Install dependencies using:
```bash
pip install tensorflow opencv-python numpy scikit-learn
```
 Dataset
The system requires a dataset of individual character images organized in a directory structure like:
```
data_dir/
├── A/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── B/
│   ├── image1.png
│   └── ...
├── 0/
├── 1/
└── ...
```
- Each folder should correspond to a character (uppercase letters A-Z and digits 0-9).
- Images should be grayscale, containing a single character.
- Multiple samples per character are recommended for robust training.

 Usage
1. Prepare Dataset:
   - Organize character images in the directory structure above.
   - Update the `data_dir` path in `captcha_recognition.py` to point to your dataset.

2. Train the Model:
   - Run the script to train the model:
     ```bash
     python captcha_recognition.py
     ```
   - The model will be trained and saved as `captcha_model.h5`.

3. Test on a CAPTCHA:
   - Update the `captcha_image` path in the script to point to a test CAPTCHA image.
   - Run the script to predict the CAPTCHA text:
     ```bash
     python captcha_recognition.py
     ```
   - The predicted CAPTCHA text will be printed to the console.

## Configuration
- Number of Characters: Adjust `num_chars` in the script (default is 4) to match the length of your CAPTCHAs.
- Model Training: Modify `epochs` in the training function for better accuracy (default is 10).
- Image Size: The model expects 28x28 character images. Adjust preprocessing and model input shape if using different sizes.

## Notes
- For CAPTCHAs with heavy distortion or overlapping characters, you may need to enhance the segmentation and preprocessing steps.
- Increase training data or epochs for improved accuracy.
- The model supports uppercase letters (A-Z) and digits (0-9) by default. Modify the character set in the script if needed.

## License
This project is licensed under the MIT License.

## Acknowledgments
Built using TensorFlow, Keras, and OpenCV for deep learning and image processing.
