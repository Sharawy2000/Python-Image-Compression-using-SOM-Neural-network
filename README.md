<p align="center">
  <a href="https://github.com/Sharawy2000/Python-Image-Compression-using-SOM-Neural-network" target="_blank">
  </a>
</p>

## 📷 Python Image Compression using SOM Neural Network

## About the Project

This project implements an image compression algorithm using a Self-Organizing Map (SOM) neural network. The SOM algorithm helps in reducing the size of an image by clustering similar pixels together, which is particularly useful for image storage and transmission.

## Installation Instructions

To install and run this project, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/Sharawy2000/Python-Image-Compression-using-SOM-Neural-network.git
    cd Python-Image-Compression-using-SOM-Neural-network
    ```

2. **Create a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## 📄 Description

This project implements an image compression algorithm using a Self-Organizing Map (SOM) neural network. The SOM algorithm helps in reducing the size of an image by clustering similar pixels together, which is particularly useful for image storage and transmission.


## 📸 Usage


To compress an image using the SOM neural network, run the following command:

bash
Copy code
python compress_image.py --input <path_to_input_image> --output <path_to_output_image> --dimensions <width>x<height>
Example:

bash
Copy code
python compress_image.py --input images/sample.jpg --output images/compressed_sample.jpg --dimensions 100x100
This will compress the input image and save the compressed image to the specified output path.



## ✨ Features

Image Compression: Reduce image size by clustering similar pixels.
Custom Dimensions: Specify the desired dimensions for the compressed image.
Easy Integration: Simple command-line interface for compressing images.



## 🤝 Contributing


We welcome contributions! To contribute, follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Make your changes and commit them (git commit -m 'Add some feature').
Push to the branch (git push origin feature-branch).
Create a new Pull Request.



## 📜 License


This project is licensed under the MIT License. See the LICENSE file for more details.



## 🙏 Acknowledgments


SOM Algorithm


```bash
python compress_image.py --input <path_to_input_image> --output <path_to_output_image> --dimensions <width>x<height>
