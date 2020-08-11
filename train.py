import os
import sys
import copy
import datetime
import numpy as np
import tensorflow as tf
import tensorflow.keras as tk

from utils.dataset import BSD68
from utils.unet_residual import UNetR as Model

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
experiment_id = sys.argv[2]

if os.path.exists(os.path.join("saver",experiment_id)) is False:
    os.mkdir(os.path.join("saver",experiment_id))

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def train():
    config = {
            "lr":4e-4,
            "momentum":0.9,
            "epoches":200,
            "accum_iters":1,
            "init_model_path":None,

            "batch_size": 128,
            "example_count":-1,
            "input_size":(180,180), #(w,h)

            "loss_interval":10,
            "image_interval":30,
            "metric_interval":60,
            "save_interval":50, # epoch interval

            "unet_depth":3,
            "unet_filter_num":96,
            "unet_use_batchnorm":True,
            "unet_padding_method":"same",
            "batch_norm_decay":0.99,
            "batch_norm_epsilon":0.001,
            }
    if False: # test
        config["epoches"] = 10
        config["batch_size"] = 2
        config["example_count"] = 10
        config["loss_interval"] = 5
        config["image_interval"] = 5
        config["metric_interval"] = 10
        config["save_interval"] = 4

    # load dataset
    bsd68 = BSD68(div_factor=2**(config["unet_depth"]-1))
    train_dataset = bsd68.load_train_dataset(batch_size=config["batch_size"],example_count=config["example_count"])
    val_dataset = bsd68.load_val_dataset()

    # create network
    model = Model(copy.deepcopy(config))
    print(f"created model with config:{config}")

    # optimizer
    #optimizer=tk.optimizers.SGD(learning_rate=config["lr"],momentum=config["momentum"])
    optimizer=tk.optimizers.Adam(learning_rate=config["lr"])

    # training
    iteration_index = 0
    accumulated_grads = [tf.Variable(initial_value=tf.zeros_like(w),trainable=False,shape=w.shape,dtype=w.dtype,name=f"{w.name}_accumulated_gradient") for w in model.weights]
    start_time = datetime.datetime.now()
    for epoch_index in range(config["epoches"]):
        print(f"epoch_index:{epoch_index}")
        for x,y in train_dataset:
            if iteration_index % config["accum_iters"] == config["accum_iters"]-1:
                apply_grads = True
            else:
                apply_grads = False

            with tf.GradientTape() as grad_tape:
                y_ = model.call(x,training=True)
                gt,mask = tf.split(y,2,axis=-1)
                total_loss = tf.math.reduce_mean( tf.math.reduce_sum((y_*mask-gt)**2,axis=(1,2))/tf.math.reduce_sum(mask,axis=(1,2)) ) # mse

            grads = grad_tape.gradient(total_loss,model.weights)
            for g_index in range(len(grads)):
                if grads[g_index] is None: continue
                accumulated_grads[g_index].assign_add(grads[g_index] / config["accum_iters"])

            if apply_grads is True:
                optimizer.apply_gradients(zip(accumulated_grads,model.weights))
                for grad_index in range(len(accumulated_grads)):
                    accumulated_grads[grad_index].assign(tf.zeros_like(accumulated_grads[grad_index]))

            if iteration_index % 5 == 0:
                print(f"epoch:{epoch_index}/{config['epoches']},iter:{iteration_index},loss:{total_loss.numpy()}")

            iteration_index += 1

    # save model
    model.save_weights(filepath=os.path.join("saver",experiment_id,"final"))
    end_time = datetime.datetime.now()
    training_time = end_time - start_time
    print(f"the end of the train:{training_time.days}d{training_time.seconds//3600}h{training_time.seconds%3600//60}m{training_time.seconds%60}s")

if __name__ == "__main__":
    train()
