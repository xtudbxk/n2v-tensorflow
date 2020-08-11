import pickle
import numpy as np
import tensorflow as tf
import tensorflow.keras as tk

class UNetR(tk.Model):
    def __init__(self,config=None,**kwargs):
        config = {} if config is None else config
        if "inputs" in kwargs or "outputs" in kwargs: 
            super(UNetR,self).__init__(**kwargs)
        else:
            inputs,outputs = self.create_body(config)
            super(UNetR,self).__init__(inputs=inputs,outputs=outputs,**kwargs)
        self.config = pickle.dumps(config)

    @staticmethod
    def create_body(config):
        # inputs
        input_h,input_w = config.get("input_size",[572,572])
        input_c = config.get("input_channel",1)
        inputs = tk.Input(shape=[input_h,input_w,input_c], batch_size=config.get("batch_size",1))

        x = UNetR.create_unet_backbone(inputs,config)
        x = UNetR.create_unet_fc(x,config)
        x = inputs+x

        return inputs,x

    @staticmethod
    def create_unet_backbone(x,config):
        unet_depth = config.get("unet_depth",5)
        conv_num_per_depth = config.get("unet_conv_num",2)
        filter_num_base = config.get("unet_filter_num",64)

        skip_layers = []
        # down
        for layer_depth in range(1,unet_depth):
            for conv_char_index in range(conv_num_per_depth):
                x = UNetR.create_unet_conv(x,filter_num_base,layer_depth,chr(conv_char_index+97),config)
            skip_layers.append(x)
            x = tk.layers.MaxPool2D(pool_size=(2,2),strides=(2,2),name=f"unet_{layer_depth}{chr(conv_char_index+97)}_pool")(x)

        # middle
        layer_depth = unet_depth
        for conv_char_index in range(conv_num_per_depth-1):
            x = UNetR.create_unet_conv(x,filter_num_base,layer_depth,chr(conv_char_index+97),config)
        x = UNetR.create_unet_conv(x,filter_num_base//2,layer_depth,chr(conv_num_per_depth+96),config)

        # up with skip layers
        for layer_depth in reversed(range(1,unet_depth)):
            x = UNetR.create_unet_upsample(x,config)
            if config.get("unet_padding_method","valid") == "valid":
                start_index = (skip_layers[layer_depth-1].shape[1] - x.shape[1])//2
                end_index = start_index+x.shape[1]
                x = tk.layers.Concatenate(axis=-1)([x,skip_layers[layer_depth-1][:,start_index:end_index,start_index:end_index,:]])
            else: # padding method: same
                x = tk.layers.Concatenate(axis=-1)([x,skip_layers[layer_depth-1]])
            for conv_char_index in range(conv_num_per_depth-1):
                x = UNetR.create_unet_conv(x,filter_num_base,layer_depth,chr(conv_char_index+97),config,name_prefix="unet_reversed")
            if layer_depth == 1:
                x = UNetR.create_unet_conv(x,filter_num_base,layer_depth,chr(conv_num_per_depth+96),config,name_prefix="unet_reversed")
            else:
                x = UNetR.create_unet_conv(x,filter_num_base//2,layer_depth,chr(conv_num_per_depth+96),config,name_prefix="unet_reversed")

        return x
                
    @staticmethod
    def create_unet_conv(x,filter_num_base,layer_depth,conv_char,config,name_prefix="unet"):
        x = tk.layers.Conv2D(filters=filter_num_base*2**(layer_depth-1),kernel_size=(3,3),strides=(1,1),padding=config.get("unet_padding_method","valid"),use_bias=True,kernel_regularizer=config.get("kernel_regularizer",None),name=f"{name_prefix}_{layer_depth}{conv_char}")(x)
        if config.get("unet_use_batchnorm",False) is True:
            x = tk.layers.BatchNormalization(momentum=config.get("batch_norm_decay",0.95),epsilon=config.get("batch_norm_epsilon",1e-5),name=f"{name_prefix}_{layer_depth}{conv_char}_batchnorm")(x)
        x = tk.layers.Activation('relu',name=f"{name_prefix}_{layer_depth}{conv_char}_relu")(x)
        return x

    @staticmethod
    def create_unet_upsample(x,config):
        x = tk.layers.UpSampling2D(size=(2,2))(x)
        return x

    @staticmethod
    def create_unet_fc(x,config):
        x = tk.layers.Conv2D(filters=config.get("unet_output_channel",1),kernel_size=(1,1),strides=(1,1),padding=config.get("unet_padding_method","valid"),use_bias=True,name="unet_fc",kernel_regularizer=config.get("kernel_regularizer",None))(x)
        return x

if __name__ == "__main__":
    import numpy as np
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "10"
    fake_data = np.ones([1,572,572,1])
    unetr = UNetR(config={"unet_padding_method":"same","unet_depth":3,"unet_use_batchnorm":True})
    for single_w in unetr.weights:
        print(f"{single_w.name},{single_w.dtype}")
    y = unetr.predict(fake_data)
    print(f"output shape:{y.shape}")
    
