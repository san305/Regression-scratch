# Linear Regression from Scratch 📉

A lightweight implementation of Simple Linear Regression using only **NumPy**. This project demonstrates the fundamental mathematics behind machine learning optimization without relying on high-level libraries like Scikit-Learn.

## 🚀 Overview
This repository contains a manual implementation of the **Gradient Descent** algorithm to find the line of best fit ($y = mx + b$) for a given dataset.

### Key Features
*   **Mean Squared Error (MSE)** calculation for cost tracking.
*   **Gradient Descent** optimization for weight ($m$) and bias ($b$) updates.
*   **Prediction** function for new data points.
*   **Vectorized implementation** using NumPy for performance.

## 🧠 The Math

The model minimizes the Cost Function (MSE) by iteratively updating parameters using partial derivatives:

1.  **Prediction:** 
    $$\hat{y}_i = m x_i + b$$

2.  **Cost Function (MSE):** 
    $$J(m,b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

3.  **Gradients:**
    *   $$\frac{\partial J}{\partial m} = -\frac{2}{n} \sum_{i=1}^{n} x_i (y_i - \hat{y}_i)$$
    *   $$\frac{\partial J}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)$$

4.  **Parameter Updates** (Gradient Descent):
    *   $$m := m - \alpha \frac{\partial J}{\partial m}$$
    *   $$b := b - \alpha \frac{\partial J}{\partial b}$$
    
    where $\alpha$ is the learning rate

## 📊 Example Usage

```python
import numpy as np
from linear_regression import LinearRegression

# Sample data
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# Create and train model
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)

# Make predictions
predictions = model.predict(np.array([6, 7, 8]))
print(predictions)  # Output: [12. 14. 16.]
