# Numpy Uncertainty Sampling
NumPy implementations of common Active Learning strategies for Uncertainty Sampling

The four types of Uncertainty Sampling in this repository are:

1. Least Confidence: difference between the most confident prediction and 100% confidence

2. Margin of Confidence: difference between the top two most confident predictions

3. Ratio of Confidence: ratio between the top two most confident predictions

4. Entropy: difference between all predictions, as defined by information theory

They were implemented for my text, Human-in-the-Loop Machine Learning:

https://www.manning.com/books/human-in-the-loop-machine-learning

Note that I started writing the text using NumPy but switched to PyTorch, so you will also find implementations of these algorithms and many more in this repo:

https://github.com/rmunro/pytorch_active_learning/



