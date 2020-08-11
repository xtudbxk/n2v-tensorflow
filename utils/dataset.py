import os
import random
import numpy as np
import tensorflow as tf

from . import preprocess

class BSD68():
    def __init__(self,div_factor=4,train_data_path=None,val_data_path=None,test_data_path=None):
        self.div_factor = div_factor
        self.train_data_path = train_data_path is None and "data/BSD68_reproducibility_data/train/DCNN400_train_gaussian25.npy" or train_data_path
        self.val_data_path = val_data_path is None and "data/BSD68_reproducibility_data/val/DCNN400_validation_gaussian25.npy" or val_data_path
        self.test_data_path = test_data_path is None and ["data/BSD68_reproducibility_data/test/bsd68_gaussian25.npy","data/BSD68_reproducibility_data/test/bsd68_groundtruth.npy"] or test_data_path
        assert isinstance(self.test_data_path,list), print(f"test_path is not a list:{self.test_data_path}")
        self.means,self.stds = self.get_means_and_stds()

    def get_means_and_stds(self):
        x = np.load(self.train_data_path) # [NHWC] or [NHW]
        if len(x.shape) <= 3:
            x = x[...,np.newaxis]
        return preprocess.get_means_and_stds(x)

    def postprocess(self,img,target_size=(None,None)): # [NHWC]
        if target_size[0] is not None:
            img = preprocess.pad_or_crop(img,target_size)
        img = preprocess.denormalize(img,self.means,self.stds)
        return img

    def download(self):
        print('''
1. open [BSD68_reproducibility_data.zip](https://cloud.mpi-cbg.de/index.php/s/pbj89sV6n6SyM29/download/) in web viewer and download it to the folder "data".
2. uncompress the "BSD68_reproducibility_data.zip" in "data" folder.
''')

    def load_test_dataset(self,example_count=-1):
        # create tf dataset
        gaussian_data = np.load(self.test_data_path[0],allow_pickle=True)
        groundtruth_data = np.load(self.test_data_path[1],allow_pickle=True)
        if example_count > 0:
            gaussian_data = gaussian_data[:example_count]
            groundtruth_data = groundtruth_data[:example_count]
        def generator():
            for gaussian_example,groundtruth_example in zip(gaussian_data,groundtruth_data):
                gaussian_example = gaussian_example.astype(np.float32)[np.newaxis,...,np.newaxis]

                target_size = (
                        ((gaussian_example.shape[2]-1)//self.div_factor+1)*self.div_factor, \
                        ((gaussian_example.shape[1]-1)//self.div_factor+1)*self.div_factor
                        )
                gaussian_example = preprocess.pad_or_crop(gaussian_example,target_size)
                gaussian_example = preprocess.normalize(gaussian_example,self.means,self.stds)[0]

                groundtruth_example = groundtruth_example.astype(np.float32)[...,np.newaxis]

                yield gaussian_example,groundtruth_example
    
        dataset = tf.data.Dataset.from_generator(generator,output_types=(tf.float32,tf.float32),output_shapes=(tf.TensorShape([None,None,1]),tf.TensorShape([None,None,1])))
        dataset = dataset.batch(1)
    
        return dataset

    def _load_dataset(self,data_path,shuffle=False,batch_size=1,input_size=(180,180),example_count=-1,mask_values=True,crop_size=(64,64),pixel_percentage=0.00198,sample_radius=2): #crop_size:(w,h)
        x = np.load(data_path,allow_pickle=True)
        print(x.shape)
        if example_count > 0:
            x = x[:example_count]
        def generator():
            indexs = list(range(x.shape[0]))
            if shuffle is True:
                random.shuffle(indexs)
            for example_index in indexs:
                example = x[example_index]
                example = example[np.newaxis,...,np.newaxis]
                example = preprocess.normalize(example,self.means,self.stds)
                if mask_values is True:
                    input_example,output_example = self._mask_values(example,crop_size,pixel_percentage,sample_radius)
                else:
                    input_example,output_example = example,example

                yield input_example[0],output_example[0]


        if mask_values is True:
            dataset = tf.data.Dataset.from_generator(generator,output_types=(tf.float32,tf.float32),output_shapes=(tf.TensorShape([crop_size[1],crop_size[0],1]),tf.TensorShape([crop_size[1],crop_size[0],2])))
        else:
            dataset = tf.data.Dataset.from_generator(generator,output_types=(tf.float32,tf.float32),output_shapes=(tf.TensorShape([input_size[1],input_size[0],1]),tf.TensorShape([input_size[1],input_size[0],1])))
        dataset = dataset.batch(batch_size)
    
        return dataset

    def _mask_values(self,x,crop_size,pixel_percentage,sample_radius): # x: [1,h,w,1] 
            # subsample patch
            w_start = np.random.randint(0,x.shape[2]-crop_size[0]+1)
            h_start = np.random.randint(0,x.shape[1]-crop_size[1]+1)
            x_patch = x[:,h_start:h_start+crop_size[1],w_start:w_start+crop_size[0],:]
            #print(f"h_start:{h_start},w_start:{w_start}")

            # pick N random pixels
            box_size = np.round(np.sqrt(1/pixel_percentage)).astype(np.int) # compute a suitable box size in case receptive files of different pixels overlapped
            pixels_count_w = int(np.ceil(crop_size[0]/box_size))
            pixels_count_h = int(np.ceil(crop_size[1]/box_size))
            random_pixels_h = []
            random_pixels_w = []
            #print(f"box_size:{box_size}, pixels_count_w:{pixels_count_w}, pixels_count_h:{pixels_count_h}")
            for i in range(pixels_count_w):
                for j in range(pixels_count_h):
                    coord_h = j*box_size+np.random.randint(0,box_size) 
                    coord_w = i*box_size+np.random.randint(0,box_size) 
                    if coord_h < x_patch.shape[1] and coord_w < x_patch.shape[2]:
                        random_pixels_h.append(coord_h)
                        random_pixels_w.append(coord_w)
            #print(f"random_pixels_h:{random_pixels_h}")
            #print(f"random_pixels_w:{random_pixels_w}")

            # the input
            input_example = np.copy(x_patch)
            def get_mask_values(random_pixels_h,random_pixels_w,sample_radius):
                mask_values = []
                for index_h,index_w in zip(random_pixels_h,random_pixels_w):
                    sample_subpatch_start_h = max(0,index_h-sample_radius)
                    sample_subpatch_end_h = sample_subpatch_start_h+2*sample_radius+1
                    shift_h = min(0,x_patch.shape[1]-sample_subpatch_end_h)
                    sample_subpatch_start_h += shift_h
                    sample_subpatch_end_h += shift_h

                    sample_subpatch_start_w = max(0,index_w-sample_radius)
                    sample_subpatch_end_w = sample_subpatch_start_w+2*sample_radius+1
                    shift_w = min(0,x_patch.shape[2]-sample_subpatch_end_w)
                    sample_subpatch_start_w += shift_w
                    sample_subpatch_end_w += shift_w
                    
                    #print(f"sample_subpatch_start_h:{sample_subpatch_start_h},sample_subpatch_end_h:{sample_subpatch_end_h}")
                    #print(f"sample_subpatch_start_w:{sample_subpatch_start_w},sample_subpatch_end_w:{sample_subpatch_end_w}")

                    mask_values.append( x_patch[:,np.random.randint(sample_subpatch_start_h,sample_subpatch_end_h),np.random.randint(sample_subpatch_start_w,sample_subpatch_end_w),:] )
                mask_values = np.stack(mask_values,axis=1)
                #print(f"mask_values:{mask_values}")
                return mask_values

            #print(f"input_example shape:{input_example[:,random_pixels_h,random_pixels_w,:].shape}")
            input_example[:,random_pixels_h,random_pixels_w,:] = get_mask_values(random_pixels_h,random_pixels_w,sample_radius)

            # the output
            output_example = np.zeros(x_patch.shape,x_patch.dtype)
            output_example[:,random_pixels_h,random_pixels_w,:] = x_patch[:,random_pixels_h,random_pixels_w,:]
            output_mask = np.zeros(x_patch.shape,x_patch.dtype)
            output_mask[:,random_pixels_h,random_pixels_w,:] = 1
            output_example = np.concatenate([output_example,output_mask],axis=-1)

            return input_example,output_example

    def load_train_dataset(self,shuffle=True,batch_size=1,input_size=(180,180),example_count=-1):
        return self._load_dataset(self.train_data_path,shuffle,batch_size,input_size,example_count,mask_values=True)

    def load_val_dataset(self,shuffle=False,batch_size=1,input_size=(180,180),example_count=-1):
        return self._load_dataset(self.val_data_path,shuffle,batch_size,input_size,example_count,mask_values=False)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "10"
    bsd68 = BSD68(div_factor=4)

    def test_loading_data(bsd68):
        train_d = bsd68.load_train_dataset(batch_size=2,example_count=5)
        print("train dataset:")
        for x,y in train_d:
            print(f"x:{x.shape},{x.dtype}")
            print(f"y:{y.shape},{y.dtype}")
            print(f"postpreprocess:{bsd68.postprocess(x).shape},{y.shape}")
    
        #val_d = bsd68.load_val_dataset()
        #print("val dataset:")
        #for x,y in val_d.take(3):
        #    print(f"x:{x.shape},{x.dtype}")
        #    print(f"y:{y.shape},{y.dtype}")
        #    print(f"postpreprocess:{bsd68.postprocess(x).shape},{y.shape}")
    
        #test_d = bsd68.load_test_dataset()
        #print("test dataset:")
        #for x,y in test_d.take(3):
        #    print(f"x:{x.shape},{x.dtype}")
        #    print(f"y:{y.shape},{y.dtype}")
        #    print(f"postpreprocess:{bsd68.postprocess(x,(y.shape[2],y.shape[1])).shape},{y.shape}")
    #test_loading_data(bsd68)
    
    def test_masking_values(bsd68):
        x = np.arange(100,dtype=np.float32).reshape([1,10,10,1])
        a,b = bsd68._mask_values(x,crop_size=(9,9),pixel_percentage=0.1,sample_radius=1)
        print(f"a shape:{a.shape},{a.dtype}, b shape:{b.shape},{b.dtype}")
        for array in [x,a,b[...,0:1],b[...,1:2]]:
            for m in range(array.shape[1]):
                for n in range(array.shape[2]):
                    print(f"{array[0,m,n,0]:4.5} ",end="")
                print("\n")
            print("\n")

    test_masking_values(bsd68)
