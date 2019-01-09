
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
import time

n = 8000
dtype = tf.float32
with tf.device("/GPU:0"):
    matrix1 = tf.Variable(tf.ones((n, n), dtype=dtype))
    matrix2 = tf.Variable(tf.ones((n, n), dtype=dtype))
    product = tf.matmul(matrix1, matrix2)

config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))

with tf.Session(config=config) as sess1:
    sess1.run(tf.global_variables_initializer())
    iters = 10
    start = time.time()
    for i in range(iters):
        sess1.run(product)
end = time.time()
ops = n**3 + (n-1)*n**2 # n^2*(n-1) additions, n^3 multiplications
elapsed = (end - start)
rate = iters*ops/elapsed/10**9
print('\n %d x %d matmul took: %.2f sec, %.2f G ops/sec' % (n, n,
                                                                elapsed/iters,
                                                                rate,))
