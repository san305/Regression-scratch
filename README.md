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

1.  **Prediction:** $\hat{y} = mx + b$
2.  **Cost Function (MSE):** $J(m,b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$
3.  **Gradients:**
    *   $\frac{\partial J}{\partial m} = -\frac{2}{n} \sum x(y - \hat{y})$
    *   $\frac{\partial J}{\partial b} = -\frac{2}{n} \sum (y - \hat{y})$
