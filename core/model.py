import tensorflow as tf

class CNN_model(tf.keras.Model):
    def __init__(self):
        super(CNN_model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=(3,3),
                                            strides=(1,1),
                                            activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=(3,3),
                                            strides=(1,1),
                                            activation='relu')
        self.max_pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2))
        self.conv3 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(3,3),
                                            strides=(1,1),
                                            activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(3,3),
                                            strides=(1,1),
                                            activation='relu')
        self.max_pool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2))
        self.flatten1 = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units = 512,
                                            activation='relu')
        self.dense2 = tf.keras.layers.Dense(units = 128,
                                            activation='relu')
        self.dense3 = tf.keras.layers.Dense(units = 10,
                                            activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.max_pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool2(x)
        x = self.flatten1(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)

    def model(self):
        x = tf.keras.layers.Input(shape=(28, 28, 1))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
