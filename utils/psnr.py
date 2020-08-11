import numpy as np
import tensorflow as tf
import tensorflow.keras as tk

def compute_psnr(img1,img2,n_bits=8):
    mse = np.mean((img1-img2)**2)
    return 10*np.log10( (2**n_bits-1)**2 / mse )

class PSNRMetric(tk.metrics.Metric):
    def __init__(self,minval=0.0,maxval=1.0):
        super(PSNRMetric,self).__init__()
        self.minval,self.maxval = minval,maxval
        self.sum_psnr = self.add_weight(name="sum_psnr",initializer="zeros",dtype=tf.float32)
        self.img_count = self.add_weight(name="img_count",initializer="zeros",dtype=tf.float32)

    def update_state(self,y_true,y_pred,sample_weight=None):
        y_true = y_true-self.minval
        y_pred = y_pred-self.minval
        psnr = tf.image.psnr(y_true,y_pred,max_val=self.maxval-self.minval)
        self.sum_psnr.assign_add(tf.math.reduce_sum(psnr))
        self.img_count.assign_add(tf.cast(tf.shape(psnr)[0],tf.float32))

    def result(self):
        return self.sum_psnr / self.img_count
