import sys
import time

import numpy as np
from PIL import Image


def progressbar(it, prefix="", size=60, out=sys.stdout):
    count = len(it)
    start = time.time()

    def show(j):
        x = int(size * j / count)
        remaining = ((time.time() - start) / j) * (count - j)
        mins, sec = divmod(remaining, 60)
        time_str = f"{int(mins):02}:{sec:03.1f}"
        print(f"{prefix}[{'█' * x}{'.' * (size - x)}] {j}/{count} Est wait {time_str}", end='\r', file=out,
              flush=True)

    show(0.1)
    for i, item in enumerate(it):
        yield item
        show(i + 1)
    print("\n", flush=True, file=out)


class SOM:
    def __init__(self, input_size, output_size, num_iterations, learning_rate, sigma):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.random((output_size[0], output_size[1], input_size))
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.sigma = sigma

    def train(self, data):
        for iteration in progressbar(range(self.num_iterations), "Training: "):
            # Select a random data point
            data_point = data[np.random.randint(len(data))]

            # Find the best matching unit (BMU)
            bmu_index = self.find_bmu(data_point)

            # Update weights of BMU and its neighbors
            self.update_weights(bmu_index, data_point, iteration)

    def find_bmu(self, data_point):
        # Calculate Euclidean distances between data point and SOM weights
        distances = np.linalg.norm(self.weights - data_point, axis=2)

        # Find the index of the best matching unit (BMU)
        bmu_index = np.unravel_index(np.argmin(distances, axis=None), distances.shape)

        return bmu_index

    def update_weights(self, bmu_index, data_point, iteration):
        # Calculate neighborhood radius
        radius = self.sigma * np.exp(-iteration / self.num_iterations)

        # Update weights of BMU and its neighbors
        for i in range(self.output_size[0]):
            for j in range(self.output_size[1]):
                # Calculate the distance between neuron and BMU
                distance = np.linalg.norm(np.array([i, j]) - np.array(bmu_index))

                # Calculate the influence of the neuron on BMU
                influence = np.exp(-distance ** 2 / (2 * radius ** 2))

                # Update weights
                self.weights[i, j] += influence * self.learning_rate * (data_point - self.weights[i, j])

    def compress_image(self, input_image_path, output_image_path):
        # Load the image
        image = Image.open(input_image_path)

        # Convert image to numpy array
        image_array = np.array(image)

        # Normalize pixel values
        image_array = image_array / 255.0

        # Get image dimensions
        height, width, channels = image_array.shape

        # Reshape image array
        image_array = image_array.reshape(-1, channels)

        # Train the SOM
        self.train(image_array)

        # Create a compressed image
        compressed_image = np.zeros(image_array.shape)

        # Assign each pixel of the image to the closest neuron's weight
        for i in progressbar(range(image_array.shape[0]), "Compressing: "):
            bmu_index = self.find_bmu(image_array[i])
            compressed_image[i] = self.weights[bmu_index]

        # Convert compressed image back to the original scale
        compressed_image *= 255

        # Reshape compressed image array
        compressed_image = compressed_image.reshape(height, width, channels)

        # Save the compressed image
        compressed_image = compressed_image.astype(np.uint8)
        compressed_image = Image.fromarray(compressed_image)
        compressed_image.save(output_image_path)

        print("Image compressed and saved successfully!")
