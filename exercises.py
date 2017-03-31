import tensorflow as tf
import input_data

def main(parms):

  # Load the MNIST dataset
  mnist = input_data.read_data_sets('MNIST_data', one_hot=True, splits = parms['splits'])

  n_data = parms['n_data']                                      # the number of data points
  n_data_feat = parms['n_data_feat']                            # the dimensionality of the feature vector
  n_labels = parms['n_labels']                                  # the number of labels
  assert(mnist.train.images.shape[0] == n_data)
  assert(mnist.train.images.shape[1] == n_data_feat)
  assert(mnist.train.labels.shape[0] == n_data)
  assert(mnist.train.labels.shape[1] == n_labels)

  # define the input and target placeholders
  data_p = tf.placeholder(tf.float32, shape = [None, n_data_feat])  # placeholder for the data that will be fed in batches,
                                                                    # therefore the first dimension is None, since the
                                                                    # size of the batch can vary
  target_p = tf.placeholder(tf.float32, shape = [None, n_labels])   # placeholder for the labels that will be fed in batches

  # all the models will produce logits, which are argumnets of the softmax function
  # softmax function converts a set of real numbers into a categorical distribution
  if (parms['model_type'] == 'logreg'):
    # -- model #1 --
    # define the model parameters
    W = tf.Variable(tf.truncated_normal([n_data_feat, n_labels], stddev = parms['init_variance'])) # for multiclass logisitc regression we have only
                                                                                                   # one fully-connected layer
                                                                                                   # (initialized with truncated normal distribution);
                                                                                                   # it maps features to labels, therefore the dimensions
                                                                                                   # of the weight matrix are [n_data_feat, n_labels]
    b = tf.Variable(tf.zeros([n_labels]))                                                          # we have one bias per class
    # define the model architecture
    logits = tf.matmul(data_p, W) + b                                                              # the definition of the mapping from data to logits
                                                                                                   # via model parameters (W,b);
  elif (parms['model_type'] == 'fc'):
    # -- model #2 --
    fc1_W = tf.Variable(tf.truncated_normal([n_data_feat, 64], stddev = parms['init_variance'])) # as in logistic regression case, just
                                                                                                 # here the first layer maps into a hidden 64-dimensional layer
    fc1_b = tf.Variable(tf.zeros([64]))
    fc2_W = tf.Variable(tf.truncated_normal([64, n_labels], stddev = parms['init_variance']))    # the last layer maps from the hidden layer into the number of
                                                                                                 # classes; this can be thought as logistic regression on the
                                                                                                 # features from the hidden layers
    fc2_b = tf.Variable(tf.zeros([n_labels]))
    # model arch
    h_fc1 = tf.nn.relu(tf.matmul(data_p, fc1_W) + fc1_b)            # hidden layers features are non-linear map of input featues,
                                                                    # here we use ReLU(x)=max(0,x) as non-linearity
    logits = tf.matmul(h_fc1, fc2_W) + fc2_b                        # same as for logistic regression, just here inputs are features from hidden layer
  elif (parms['model_type'] == 'conv'):
    # -- model #3 --
    conv1_w = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev = parms['init_variance']))  # in convolutional network the paramters are filter kernels,
                                                                                                # here we map an input image (in MNIST that is 28x28x1 image,
                                                                                                # last channel is x1 since it's grayscale image) into 32 feature
                                                                                                # maps at the output. The kernel size is 5x5, the number of
                                                                                                # input maps is 1, and the number of output maps is 32.
    conv1_b = tf.Variable(tf.zeros([32]))
    conv2_w = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev = parms['init_variance'])) # second convolutional layer operates on 32 feature maps produced
                                                                                                # by the first convolutional layer and maps them into 64 feature maps;
                                                                                                # the size of the kernel is 5x5. Note that the each of 64 kernels in
                                                                                                # this layer operates on all 32 input maps, so each kernel has
                                                                                                # 5*5*32=800 weights per kernel or 64*800=51200 weights for all kernels
                                                                                                # in the second convolutional layer
    conv2_b = tf.Variable(tf.zeros([64]))
    fc1_W = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev = parms['init_variance']))   # after two convolutional layers we have a fully-connected layer
    fc1_b = tf.Variable(tf.zeros([1024]))
    fc2_W = tf.Variable(tf.truncated_normal([1024, n_labels], stddev = parms['init_variance'])) # and another one that maps hidden variables from the third hidden layer
                                                                                                # into the logits
    fc2_b = tf.Variable(tf.zeros([n_labels]))
    # model arch
    image = tf.reshape(data_p, [-1, 28, 28, 1])                                                                 # since the images are stored as vectors we first
                                                                                                                # reshape them into a tensor; first dimension of the
                                                                                                                # tensor is the number of images in the batch,
                                                                                                                # the second dimension is image height, the third
                                                                                                                # the image widht and the last one is the number of
                                                                                                                # input channels
    h_conv1 = tf.nn.relu(tf.nn.conv2d(image, conv1_w, strides = [1, 1, 1, 1], padding = 'SAME') + conv1_b)      # we apply the convolution on the input and apply
                                                                                                                # ReLU non-linearity, padding 'SAME' denotes that the
                                                                                                                # image will be padded with zeros so that the output
                                                                                                                # has the same spatial dimensions as input (for MNIST
                                                                                                                # that is 28x28)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')           # we after convolution we apply max-pooling with
                                                                                                                # kernel size 2x2 in spatial domain, no pooling over
                                                                                                                # batches (first dimension) or channels (last dimension),
                                                                                                                # with a stride 2 (non-overlapping max pooling)
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, conv2_w, strides = [1, 1, 1, 1], padding = 'SAME') + conv2_b)    # the second convolutional layer is functionally the same
                                                                                                                # as the previous one
    h_pool2 = tf.nn.max_pool(h_conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')           # same for max pooling
    h_fc1 = tf.nn.relu(tf.matmul(tf.reshape(h_pool2, [-1, 7*7*64]), fc1_W) + fc1_b)                             # here we first reshape feature maps that are output
                                                                                                                # of second convolutional layer into one vector per
                                                                                                                # data point: since inputs are 28x28 and they pass through
                                                                                                                # two 2x2 non-overlapping max-pooling layers the spatial
                                                                                                                # size of feature maps is 28/(2*2)=7, so the feature maps
                                                                                                                # for one image are of dimension 7x7x64; then we
                                                                                                                # feed these reshaped features into a fully-connected
                                                                                                                # layer
    logits = tf.matmul(h_fc1, fc2_W) + fc2_b                                                                    # finally we apply an output layer, as in previous examples,
                                                                                                                # to obtain the logits

  # evaluate loss from net output and target
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=target_p))                # our loss is cross-entropy loss between the net predictions
                                                                                                                # and the true labels (target_p); for numberical stability
                                                                                                                # instead of supplying the posteriors (passing the logits
                                                                                                                # through the softmax) we supply the logits into the function
                                                                                                                # tf.nn.softmax_cross_entropy
  # prediction is softmax of logits
  prediction = tf.nn.softmax(logits)
  # define the evaluation measure
  correct_prediction = tf.equal(tf.argmax(target_p, 1), tf.argmax(prediction, 1))                               # we evaluate the prediction accuracy of the network by first
                                                                                                                # determining for each data point the class with maximum
                                                                                                                # posterior, and comparing that class with the ground-truth:
                                                                                                                # if the predicted class is the same as ground truth prediction
                                                                                                                # for that data point is correct (1), otherwise not correct (0)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))                                            # accuracy is just a proportion of correctly predicted labels

  # define the optimizer
  if (parms['optimizer_type'] == 'graddesc'):
    optimizer = tf.train.GradientDescentOptimizer(parms['lambda'])                                              # the simplest optimizer is pure stochastic gradient descent
  elif (parms['optimizer_type'] == 'momentum'):
    optimizer = tf.train.MomentumOptimizer(parms['lambda'], momentum = 0.9)                                     # a bit more sophisticated optimizer is SGD with momentum,
                                                                                                                # in a nutshell, this prevents the gradients to deviate too
                                                                                                                # much from the gradients computed in the previous steps, i.e.
                                                                                                                # the parameters are intert
  elif (parms['optimizer_type'] == 'ADAM'):
    optimizer = tf.train.AdamOptimizer(parms['lambda'])                                                         # ADAM is currently the most advanced optimizer, gives
                                                                                                                # decent results for majority of the networks and converges
                                                                                                                # much faster than the previous two
  # TODO: try decaying learning rate (see TF API: tf.train.exponential_decay)
  optimization_step = optimizer.minimize(loss)                                                                  # the optimization step is performed by minimizing the loss
                                                                                                                # function; behind the scenes function minimize actually calls
                                                                                                                # two functions: first it calls compute_gradients, that computes
                                                                                                                # all needed gradients using backprop, second it calls
                                                                                                                # apply_gradients that applies gradients according to the
                                                                                                                # chosen optimizer

  saver = tf.train.Saver()                                                                                      # an object used for saving the parameters of the network
                                                                                                                # into a checkpoint file

  sess = tf.Session()                                                                                           # TensorFlow session object, which contais the graph and
                                                                                                                # within which all the operations will be evaluated

  sess.run(tf.initialize_all_variables())                                                                       # initialize all graph variables (weights, biases)

  cur_epoch = 0
  best_val_accuracy = 0
  lookahead_counter = 0
  # start training
  while (True):
    # get the next batch
    batch_data, batch_targets = mnist.train.next_batch(parms['batch_size'])                                     # for the batching we use the class from the mnist.py
                                                                                                                # please see the code there to see how the batches are
                                                                                                                # formed and prepared
    # evaluate net outputs, loss and modify net params
    feed_dict = {data_p: batch_data, target_p: batch_targets}
    _, batch_loss, batch_predictions = sess.run([optimization_step, loss, prediction], feed_dict = feed_dict)   # get the batch loss and batch predictions given the current
                                                                                                                # batch data and labels at input (in feed_dict)
    # if we have finished a pass through training data (an epoch)
    if (mnist.train.epochs_completed != cur_epoch):
      if (parms['low_memory']):
        val_accuracy = 0.0
        while(mnist.validation.epochs_completed == cur_epoch):
          batch_val_data, batch_val_targets = mnist.validation.next_batch(parms['batch_size'])
          val_accuracy += sum(sess.run(correct_prediction, feed_dict = {data_p: batch_val_data, target_p: batch_val_targets}))
        val_accuracy /= mnist.validation.num_examples
        print('Epoch #%d: validation_accuracy = %f' % (cur_epoch, val_accuracy))
      else:
        # evaluate loss and accuracy on the validation data
        [val_loss, val_accuracy] = sess.run([loss, accuracy], feed_dict = {data_p: mnist.validation.images, target_p: mnist.validation.labels})
        print('Epoch #%d: validation loss = %f, validation_accuracy = %f' % (cur_epoch, val_loss, val_accuracy))
      if (val_accuracy >= best_val_accuracy):
        # if this is the best accuracy on validation data so far, save the model ...
        print('Found best validation accuracy, saving model')
        best_val_accuracy = val_accuracy
        saver.save(sess, parms['checkpoint_path'])
        # ... and reset lookahead counter
        lookahead_counter = 0
      else:
        # if this is not the best model we might be overfitting already, so increase lookahead counter
        lookahead_counter += 1
      cur_epoch = mnist.train.epochs_completed
    if (mnist.train.epochs_completed == parms['max_epochs'] or
        lookahead_counter == parms['max_lookahead']):
      # if we have reached maximum number of epochs or we have hit lookahead counter wall (which prevents overfitting), quit training
      # this is called "early stopping" and it is a standard way of regularizing deep models
      break

  # load the best model
  saver.restore(sess, parms['checkpoint_path'])
  # evaluate performance on test set
  if (parms['low_memory']):
    test_accuracy = 0.0
    while(not mnist.test.epochs_completed):
      batch_test_data, batch_test_targets = mnist.test.next_batch(parms['batch_size'])
      test_accuracy += sum(sess.run(correct_prediction, feed_dict = {data_p: batch_test_data, target_p: batch_test_targets}))
    test_accuracy /= mnist.test.num_examples
  else:
    # evaluate the model that performs best on validation data on the test data
    test_accuracy = sess.run(accuracy, feed_dict = {data_p: mnist.test.images, target_p: mnist.test.labels})
  print('Test accuracy: %f' % (test_accuracy))

if __name__ == '__main__':

  splits = [55000, 5000, 5000]                         # [n_train, n_val, n_test]

  parms = { 'splits' : splits,                         # the sizes of splits (train, val, test)
            'n_data' : splits[0],                      # the number of training data points
            'n_data_feat' : 784,                       # the dimension of input (MNIST are 28x28 images flattened into 784-dimensional vector)
            'n_labels' : 10,                           # there are 10 labels (digits 0..9)
            'model_type' : 'conv',                     # one of the three models: 'logreg', 'fc' and 'conv' (please see the code above)
            'lambda' : 1e-4,                           # the step size of the gradient descent (0.5 for 'SGD', 1e-4 for 'ADAM')
            'init_variance': 0.1,                      # variance of the weight initializer
            'optimizer_type' : 'ADAM',                 # the optimizer used, can be: 'graddesc', 'momentum' and 'ADAM' (please see the code above)
            'max_epochs' : 200,                        # maximal number of epochs (runs through the dataset)
            'max_lookahead' : 5,                       # the size of lookahead: how many steps to try to improve validation performace before quitting
            'batch_size' : 50,                         # the size of the mini-batch for stochastic gradient descent
            'checkpoint_path' : '/tmp/model.ckpt',     # the path to the checkpoint file where the weights of the best performing network are saved
            'low_memory': False }                      # if there is not enough memory to fit all the validation/data in memory the evaluation will be done in chunks

  main(parms)
