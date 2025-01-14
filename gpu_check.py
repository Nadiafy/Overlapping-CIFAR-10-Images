import tensorflow as tf
import time

# Start a simple computation and measure execution time
with tf.device("/GPU:0"):  # Force using the first GPU
    start_time = time.time()
    tensor = tf.random.uniform([10000, 10000])
    result = tf.matmul(tensor, tensor)
    print(f"GPU computation time: {time.time() - start_time:.4f} seconds")

with tf.device("/CPU:0"):  # Force using the CPU
    start_time = time.time()
    tensor = tf.random.uniform([10000, 10000])
    result = tf.matmul(tensor, tensor)
    print(f"CPU computation time: {time.time() - start_time:.4f} seconds")

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # Default, enables full logs

# Re-run your TensorFlow code
