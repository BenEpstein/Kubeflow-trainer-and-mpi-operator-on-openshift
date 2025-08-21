# tf_mnist_hvd.py
import tensorflow as tf
import horovod.tensorflow.keras as hvd

# 1) Horovod init
hvd.init()

# 2) One GPU per local MPI process
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[hvd.local_rank()], True)

# 3) Data
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_train = x_train[..., None]  # (N, 28, 28, 1)

# Shard the dataset by rank to avoid duplication
num_workers = hvd.size()
my_rank = hvd.rank()
x_shard = x_train[my_rank::num_workers]
y_shard = y_train[my_rank::num_workers]

ds = tf.data.Dataset.from_tensor_slices((x_shard, y_shard)).shuffle(10000).batch(128)

# 4) Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28,28,1)),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

# 5) Optimizer scaled by number of workers
base_lr = 0.01
opt = tf.keras.optimizers.SGD(learning_rate=base_lr * num_workers, momentum=0.9)
opt = hvd.DistributedOptimizer(opt)

model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# 6) Horovod callbacks: broadcast initial weights, average metrics
callbacks = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(root_rank=0),
    hvd.callbacks.MetricAverageCallback(),
]

# Only rank 0 prints progress
verbose = 1 if hvd.rank() == 0 else 0
model.fit(ds, epochs=3, callbacks=callbacks, verbose=verbose)
if hvd.rank() == 0:
    model.save("mnist_hvd_tf.keras")
