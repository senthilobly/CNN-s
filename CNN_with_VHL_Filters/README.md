
# 🧠 Simple Convolutional Neural Network (CNN) Using NumPy

This project demonstrates a basic **Convolutional Neural Network (CNN)** pipeline implemented from scratch using **NumPy**, without any deep learning libraries like TensorFlow or PyTorch.

It covers the core operations of a CNN:
- Convolution using predefined kernels
- Max Pooling for downsampling
- Fully Connected Layer
- Softmax Activation for classification

---

## 📌 Features

- Convolution using 3 filters:
  - **Vertical Edge Detection (Sobel Y)**: Detects vertical lines or edges.
  - **Horizontal Edge Detection (Sobel X)**: Detects horizontal lines or edges.
  - **Laplacian Edge Detection**: Captures all-direction edge intensity by computing second-order derivatives.

These filters are commonly used in image processing to detect edges and structure from images.

---

## 🏗️ Architecture

```
Input Image (28x28 grayscale)
        ↓
3 Convolution Filters (3x3)
        ↓
Feature Maps (3x26x26)
        ↓
Max Pooling (2x2, stride=2)
        ↓
Pooled Feature Maps (3x13x13)
        ↓
Flattened Vector (507 values)
        ↓
Fully Connected Layer (10 Neurons)
        ↓
Softmax Activation
        ↓
Class Probabilities (10 classes)
```

---

## 📂 Files

| File         | Description |
|--------------|-------------|
| `cnn_numpy.py` | Main Python script implementing the CNN using NumPy |
| `README.md`  | Project documentation and usage guide |

---

## 📦 Dependencies

- Python 3.x
- NumPy
- Matplotlib

Install requirements (if needed):

```bash
pip install numpy matplotlib
```

---

## ▶️ How It Works

1. **Input**: A randomly generated grayscale image of size `28x28`
2. **Convolution**: Applied 3 different kernels to extract features:
   - Vertical (Sobel Y)
   - Horizontal (Sobel X)
   - Laplacian (for all edges)
3. **Max Pooling**: Downsamples each feature map using a `2x2` kernel and stride of `2`
4. **Flattening**: Converts the pooled maps into a single vector
5. **Fully Connected Layer**: Multiplies the flattened vector with randomly initialized weights and biases
6. **Softmax**: Converts raw outputs into probability scores across 10 classes

---

## 📊 Output

- **Feature Maps**: Displays the filtered results after convolution for each kernel
- **Max-Pooled Output**: Reduces spatial dimensions to improve efficiency
- **Final Output**: Prints softmax probabilities for 10 hypothetical classes

---

## 🧪 Sample Output

```bash
Input Image Shape: (28, 28)
Output Shape: (3, 26, 26)

Output Shape after max pooling: (3, 13, 13)
Flattened Size: (507,)
Weight shape: (10, 507)
Bias shape: (10,)
Fully Connected Output (10 classes): [ 1.23, -0.56, ..., 3.45 ]
Class Probabilities after Softmax: [0.01, 0.15, ..., 0.20]
```

Visual Output:

- Filtered outputs for:
  - Vertical Edges
  - Horizontal Edges
  - Laplacian Edges
- Each visualized using Matplotlib.

---

## 📝 Notes

- This is a minimal working example for **educational purposes**.
- No backpropagation or training is included — weights and biases are random.
- Extend it to include:
  - Multiple layers
  - Training via gradient descent
  - Real image datasets (e.g., MNIST)

---

## 📸 Screenshots

*(Add image showing feature maps if available)*

---

## 👨‍💻 Author

**Senthilkumar Ravi**  
Learning CNNs from scratch using NumPy

---

## 📚 References

- Sobel and Laplacian Edge Detection
- Neural Network basics
- Deep Learning by Goodfellow et al.
- [D2L.ai](https://d2l.ai/) - Dive into Deep Learning

---
