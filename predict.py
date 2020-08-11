import os
import sys
import copy
import pickle
import datetime
import numpy as np
import tensorflow.keras as tk

from utils.dataset import BSD68
from utils.unet_residual import UNetR as Model
from utils.psnr import PSNRMetric

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
experiment_id = sys.argv[2]

if len(sys.argv) <= 3:
    saved_weights_paths = [os.path.join("saver",experiment_id,"final")]
else:
    saved_weights_paths = sys.argv[3:]

def load_trained_npy_weights(model,pretrained_weights_path):
    pretrained_weights = np.load(pretrained_weights_path,allow_pickle=True)
    weights = []
    for single_variable in model.variables:
        #print(f"{single_variable.name} shape:{single_variable.shape}--{pretrained_weights[single_variable.name].shape}")
        weights.append( pretrained_weights[single_variable.name] )
 
    model.set_weights(weights)
    print(f"load trained npy weights from {pretrained_weights_path}")

def predict():
    config = {
            "epoches":1,
            "batch_size": 1,
            "input_size":(None,None),
            "saved_weights_paths": saved_weights_paths,
            "max_example_count":-1,
            "category_num":1,

            "unet_depth":3,
            "unet_padding_method":"same",
            "unet_use_batchnorm":True,
            "unet_filter_num":96,

            "batch_norm_decay":0.99,
            "batch_norm_epsilon":0.001,
            }

    # create network
    model = Model(copy.deepcopy(config))

    # load dataset
    bsd68 = BSD68(div_factor=2**(config["unet_depth"]-1))
    test_dataset = bsd68.load_test_dataset()

    # predict 
    print(f"predict config:{config}")
    metric = PSNRMetric(minval=0.0,maxval=255.0)
    start_time = datetime.datetime.now()
    for single_saved_weights_path in config["saved_weights_paths"]:
        # load weights
        print(f"load weights from {single_saved_weights_path}")
        if single_saved_weights_path.endswith("npy") or single_saved_weights_path.endswith("npz"):
            load_trained_npy_weights(model,single_saved_weights_path)
        else:
            model.load_weights(single_saved_weights_path)

        metric.reset_states()
        for index,example in enumerate(test_dataset):
            if index%100 == 0: print(f"cur example:{index}/68")
            x,y = example
            y_ = model.predict(x)
            y_ = bsd68.postprocess(y_,target_size=(y.shape[2],y.shape[1]))
            metric.update_state(y,y_)

        psnr = metric.result().numpy()
        print(f"psnr:{psnr:6.4} for {single_saved_weights_path}\n")

    end_time = datetime.datetime.now()
    delta_time = end_time - start_time
    print(f"total time:{delta_time.seconds//60}m{delta_time.seconds%60}s")
    print("the end of the predict")

if __name__ == "__main__":
    predict()
