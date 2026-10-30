# DCGAN Image Generation using Fashion-MNIST

A Deep Convolutional Generative Adversarial Network (DCGAN) implemented with Keras to generate synthetic grayscale fashion-item images based on the Fashion-MNIST dataset.

The project trains two neural networks together:
- **Generator:** takes a 100-dimensional random noise vector and generates a 28 × 28 grayscale image.
- **Discriminator:** receives an image and predicts whether it is real (from Fashion-MNIST) or generated (fake).

The generator learns to create increasingly realistic images by trying to fool the discriminator.

## Project Overview

| Item | Details |
|---|---|
| Project type | Deep Learning / Generative AI |
| Model | Deep Convolutional Generative Adversarial Network (DCGAN) |
| Dataset | Fashion-MNIST |
| Image size | 28 × 28 pixels |
| Image channels | 1 (grayscale) |
| Latent dimension | 100 |
| Training epochs | 20 |
| Batch size | 128 |
| Framework | Keras |
| Language | Python |

## Dataset

The project uses the Fashion-MNIST dataset available through `keras.datasets`. It contains 28 × 28 grayscale images of fashion items. The notebook loads the training images directly; a separate manual dataset download is not required when the Keras dataset can be accessed.

Before training, images are converted to `float32` and normalized from the pixel range `[0, 255]` to `[-1, 1]`, matching the generator's `tanh` output.

## Model Architecture

### 1. Generator

The generator receives a random vector of length 100 and transforms it into a 28 × 28 × 1 image.

Main stages:
1. Dense layer expands the latent vector to `7 × 7 × 128`.
2. Reshape layer creates a 7 × 7 feature map.
3. Transposed convolution upsamples the feature map to 14 × 14.
4. A second transposed convolution upsamples it to 28 × 28.
5. A final convolution with `tanh` activation produces the grayscale image.

### 2. Discriminator

The discriminator classifies an input 28 × 28 grayscale image as real or fake.

Main stages:
1. Convolution layer with 64 filters.
2. Two convolution layers with 128 filters, using stride 2 for downsampling.
3. LeakyReLU activations.
4. Flatten and Dropout layers.
5. Dense output layer with sigmoid activation for binary classification.

### 3. Training

The discriminator is trained using real images labelled `1` and generated images labelled `0`. The generator is trained through the combined GAN model, with the discriminator frozen in that combined model. During generator training, generated images are assigned target label `1`, encouraging the generator to fool the discriminator.

The notebook uses the Adam optimizer with a learning rate of `0.0002` and `beta_1=0.5`, and binary cross-entropy loss.

## Repository Structure

Keep the files in the following structure when uploading the project to GitHub:

```text
DCGAN-Fashion-MNIST/
│
├── asiignment_dgcan.ipynb
├── generator_model_005.h5
├── generator_model_010.h5
├── generator_model_015.h5
├── generator_model_020.h5
│
├── generated_plot_e005.png
├── generated_plot_e010.png
├── generated_plot_e015.png
├── generated_plot_e020.png
│
└── README.md
```

> There are two uploaded files named `generator_model_015.h5` (one has an additional `(1)` in its local filename). Keep only one copy in the GitHub repository and give it the name `generator_model_015.h5`.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/<your-repository-name>.git
   cd <your-repository-name>
   ```

2. Install the required Python packages:

   ```bash
   pip install tensorflow numpy matplotlib jupyter
   ```

   The notebook imports Keras through `keras`. If your environment does not provide the standalone `keras` package with TensorFlow, install a compatible Keras version for your TensorFlow installation.

## How to Run

1. Open the project folder in VS Code or a terminal.
2. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

3. Open `asiignment_dgcan.ipynb`.
4. Run the notebook cells in order, from top to bottom.
5. The notebook loads Fashion-MNIST, defines the generator and discriminator, trains the GAN for 20 epochs, and plots the training losses.
6. During training, the code saves a generator checkpoint and a 7 × 7 grid of generated images every 5 epochs.

Training from scratch may take time depending on your CPU/GPU and TensorFlow configuration.

## Generated Samples

The repository includes generated image grids saved at selected training checkpoints:

| Training checkpoint | Image grid |
|---|---|
| Epoch 5 | `generated_plot_e005.png` |
| Epoch 10 | `generated_plot_e010.png` |
| Epoch 15 | `generated_plot_e015.png` |
| Epoch 20 | `generated_plot_e020.png` |

These grids contain 49 generated grayscale images each. They can be viewed directly from the repository.

## Saved Generator Checkpoints

The following generator model files are included:

- `generator_model_005.h5` — checkpoint saved after epoch 5.
- `generator_model_010.h5` — checkpoint saved after epoch 10.
- `generator_model_015.h5` — checkpoint saved after epoch 15.
- `generator_model_020.h5` — checkpoint saved after epoch 20.

To load a saved generator in Python:

```python
from keras.models import load_model

generator = load_model("generator_model_020.h5")
```

To generate new images:

```python
import numpy as np
import matplotlib.pyplot as plt

latent_dim = 100
noise = np.random.randn(16, latent_dim)
generated_images = generator.predict(noise)

# Convert values from [-1, 1] to [0, 1] for display
generated_images = (generated_images + 1) / 2.0

plt.figure(figsize=(6, 6))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(generated_images[i].squeeze(), cmap="gray")
    plt.axis("off")

plt.tight_layout()
plt.show()
```

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Jupyter Notebook
- Fashion-MNIST dataset

## Learning Outcomes

- Understanding the working principle of Generative Adversarial Networks.
- Building generator and discriminator networks using convolutional layers.
- Applying transposed convolutions for image upsampling.
- Training GAN models using adversarial learning.
- Monitoring generator and discriminator losses.
- Saving and loading trained generator checkpoints.
- Generating synthetic grayscale images from random latent vectors.

## Notes

- The included `.h5` files are saved generator checkpoints, not the complete training notebook or dataset.
- The notebook downloads Fashion-MNIST through Keras when required.
- Generated image quality can vary; GAN training does not guarantee perfect or class-controlled outputs.
- The notebook's training loop saves checkpoints and sample grids every 5 epochs.

## Author

**Sarthak Andhale**

---

If you use or extend this project, feel free to fork the repository and experiment with the model architecture, latent dimension, training duration, and hyperparameters.
