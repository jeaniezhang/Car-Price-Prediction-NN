# Prediction Competition: Car-Price-Prediction-NN

This project is a submission for the 'Machine Learning in Economics' prediction competition. 
The goal was to predict the natural logarithm of a used car's price using only a **single-hidden-layer neural network**

## Project Workflow

1.  **Data Preprocessing:** Combined two large training datasets (total of 600k+ rows).
2.  **Advanced Feature Engineering:** Wrote custom functions to parse over 15 text-based features (e.g., "296 hp @ 6,200 RPM" -> `296`) into numerical inputs.
3.  **Model Architecture:** Built a robust TensorFlow model implementing L2 regularization, BatchNormalization, and Dropout to prevent overfitting.
4.  **Hyperparameter Tuning:** Ran a tuning loop to find the optimal hidden layer size, testing [64, 128, 256, 512] nodes. The 256-node layer was found to be optimal.
5.  **Training:** Used modern callbacks including `EarlyStopping` and `ReduceLROnPlateau` for efficient training.

## Final Results

The competition was judged on **Mean Squared Error (MSE)** on the final hold-out test set.

* **My Best Validation MSE:** 0.01193
* **My Best Validation R-squared:** 0.9276

* **Final Leaderboard R-squared:** 0.9272
