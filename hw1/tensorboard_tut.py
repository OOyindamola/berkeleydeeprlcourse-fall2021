import tensorflow as tf
tf.compat.v1.disable_eager_execution() 
# create graph
a = tf.constant(2)
b = tf.constant(3)
c = tf.add(a, b)
# launch the graph in a session
with tf.compat.v1.Session() as sess:
    print(sess.run(c))
