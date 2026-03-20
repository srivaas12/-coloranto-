# AI Image Colorization Web Application

## Overview

This project is a deep learning-based web application that performs automatic image colorization. It converts grayscale or low-color images into realistic colorized versions using pretrained convolutional neural network models. The application is built with an interactive interface that allows users to upload images and visualize results in real time.

## Features

* Image colorization using pretrained ECCV16 and SIGGRAPH17 models
* Interactive web interface built with Streamlit
* Side-by-side comparison and interactive before–after slider
* Support for multiple image uploads
* GPU acceleration (optional) for faster inference
* Downloadable colorized outputs

## Technology Stack

* Python
* PyTorch
* Streamlit
* NumPy
* Pillow
* scikit-image

## How It Works

The application takes an input image and processes it through the following pipeline:

1. Image preprocessing and resizing
2. Conversion to grayscale luminance channel
3. Passing the image through pretrained colorization models
4. Postprocessing to reconstruct the color image
5. Displaying and exporting the results

Two models are used:

* ECCV16: Produces smooth and consistent colorization
* SIGGRAPH17: Produces sharper and more vibrant outputs

## Installation

Clone the repository:

```
git clone https://github.com/your-username/ai-colorization-app.git
cd ai-colorization-app
```

Create a virtual environment and activate it:

```
python -m venv venv
venv\Scripts\activate   # Windows
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the application:

```
streamlit run app.py
```

## Usage

* Upload an image using the file uploader
* View the original and colorized outputs
* Compare results using the interactive slider
* Download the generated images

## Project Structure

```
.
├── app.py
├── requirements.txt
├── colorizers/
│   ├── eccv16.py
│   ├── siggraph17.py
│   ├── util.py
├── .streamlit/
│   └── config.toml
```

## Acknowledgements

This project utilizes pretrained image colorization models originally proposed in:

* Richard Zhang et al., “Colorful Image Colorization,” ECCV 2016
* Richard Zhang et al., “Real-Time User-Guided Image Colorization with Learned Deep Priors,” SIGGRAPH 2017

The implementation is inspired by open-source contributions in the computer vision community and adapted into an interactive web application using Streamlit.

## Author

Sri Vasudevan R
