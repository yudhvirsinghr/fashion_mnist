import tensorflow as tf

csv_logger = tf.keras.callbacks.CSVLogger(filename ='./csv_log/training.log',
        separator=',', append=False)

tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs',
        histogram_freq=0, write_graph=True, write_images=True,
        update_freq='epoch', profile_batch=2, embeddings_freq=0,
        embeddings_metadata=None)
