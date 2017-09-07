
# coding: utf-8

# In[11]:


# Softmax example in TF using the classical Iris dataset
# Download iris.data from https://archive.ics.uci.edu/ml/datasets/Iris

import tensorflow as tf
import os

# this time weights form a matrix, not a column vector, one "weight vector" per class.
W = tf.Variable(tf.zeros([4, 3]), name="weights")
# so do the biases, one per class.
b = tf.Variable(tf.zeros([3], name="bias"))


def combine_inputs(X):
    return tf.matmul(X, W) + b


def inference(X):
    return tf.nn.softmax(combine_inputs(X))


def loss(X, Y):
    '''
    https://www.tensorflow.org/versions/r0.12/api_docs/python/nn/classification#sparse_softmax_cross_entropy_with_logits
    tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name=None)
    
    https://www.tensorflow.org/versions/r1.2/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits
    sparse_softmax_cross_entropy_with_logits(
    _sentinel=None,
    labels=None,
    logits=None,
    name=None
    )
    '''
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=combine_inputs(X), labels=Y))


def read_csv(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer([os.path.join(os.getcwd(), file_name)])

    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)

    # decode_csv will convert a Tensor from type string (the text line) in
    # a tuple of tensor columns with the specified defaults, which also
    # sets the data type for each column
    decoded = tf.decode_csv(value, record_defaults=record_defaults)

    # batch actually reads the file and loads "batch_size" rows in a single tensor
    return tf.train.shuffle_batch(decoded,
                                  batch_size=batch_size,
                                  capacity=batch_size * 50,
                                  min_after_dequeue=batch_size)


def inputs():

    '''
    http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
    
    5.1,3.5,1.4,0.2,Iris-setosa
    4.9,3.0,1.4,0.2,Iris-setosa

    '''
    sepal_length, sepal_width, petal_length, petal_width, label =        read_csv(100, "iris.data", [[0.0], [0.0], [0.0], [0.0], [""]])

    # convert class names to a 0 based class index.
    label_number = tf.to_int32(tf.argmax(tf.to_int32(tf.stack([
        tf.equal(label, ["Iris-setosa"]),
        tf.equal(label, ["Iris-versicolor"]),
        tf.equal(label, ["Iris-virginica"])
    ])), 0))

    # Pack all the features that we care about in a single matrix;
    # We then transpose to have a matrix with one example per row and one feature per column.
    features = tf.transpose(tf.stack([sepal_length, sepal_width, petal_length, petal_width]))

    return features, label_number


def train(total_loss):
    learning_rate = 0.01
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


def evaluate(sess, X, Y):

    predicted = tf.cast(tf.argmax(inference(X), 1), tf.int32)

    print (sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32))))


# Launch the graph in a session, setup boilerplate
with tf.Session() as sess:

    tf.global_variables_initializer().run()

    X, Y = inputs()

    total_loss = loss(X, Y)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # actual training loop
    training_steps = 1000
    for step in range(training_steps):
        sess.run([train_op])
        # for debugging and learning purposes, see how the loss gets decremented thru training steps
        if step % 10 == 0:
            print("loss: ", sess.run([total_loss]))

    evaluate(sess, X, Y)
    
    import time
    time.sleep(5)
    
    coord.request_stop()
    coord.join(threads)
    sess.close()


# In[ ]:




