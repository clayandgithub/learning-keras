from keras.models import Sequential
from keras.layers import Dense, Activation, Merge, Dropout
from utils import *
from keras.utils.np_utils import to_categorical

class Config:
    epsilon = 0.01  # learning rate for gradient descent
    reg_lambda = 0.01  # regularization strength
    layers = [2, 4, 2] # number of nodes in each layer
    nb_epoch = 2000
    print_loss = True
    random_seed = 6
    num_samples = 5000

def test_binary_classification():
    X, y = generate_data(Config.random_seed, Config.num_samples)
    y = y.reshape(Config.num_samples, 1)

    model = Sequential()
    model.add(Dense(output_dim = 4, input_dim = 2, activation = 'tanh'))
    model.add(Dense(output_dim = 1, input_dim = 4, activation = 'tanh'))
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics = ['accuracy'])

    print 'start training...'
    start_time = time.time()
    model.fit(X, y, nb_epoch = Config.nb_epoch, batch_size = 5000)
    end_time = time.time()
    print 'training time : ' + str(end_time - start_time)
    
    visualize(X, y, lambda x : model.predict_classes(x, batch_size = 5000))

def test_multi_classification():
    # for a multi-input model with 10 classes:

    left_branch = Sequential()
    left_branch.add(Dense(32, input_dim=784))

    right_branch = Sequential()
    right_branch.add(Dense(32, input_dim=784))

    merged = Merge([left_branch, right_branch], mode='concat')

    model = Sequential()
    model.add(merged)
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # generate dummy data
    data_1 = np.random.random((1000, 784))
    data_2 = np.random.random((1000, 784))

    # these are integers between 0 and 9
    labels = np.random.randint(10, size=(1000, 1))
    # we convert the labels to a binary matrix of size (1000, 10)
    # for use with categorical_crossentropy
    labels = to_categorical(labels, 10)

    # train the model
    # note that we are passing a list of Numpy arrays as training data
    # since the model has 2 inputs
    model.fit([data_1, data_2], labels, nb_epoch=1000, batch_size=1000)

def test_mlp_binary_classification():
    model = Sequential()
    model.add(Dense(64, input_dim=2, init='uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    X, y = generate_data(Config.random_seed, Config.num_samples)
    y = y.reshape(Config.num_samples, 1)

    print 'start training...'
    start_time = time.time()
    model.fit(X, y, nb_epoch = Config.nb_epoch, batch_size = 5000)
    end_time = time.time()
    print 'training time : ' + str(end_time - start_time)
    
    visualize(X, y, lambda x : model.predict_classes(x, batch_size = 5000))

test_mlp_binary_classification()