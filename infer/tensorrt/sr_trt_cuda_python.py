'''
 FILENAME:      sr_trt_cuda_python.py

 AUTHORS:       Pan Shaohua

 START DATE:    Friday September 9th 2022

 CONTACT:       shaohua.pan@quvideo.com

 INFO:          
'''

import ctypes
import os
import shutil
import random
import sys
import threading
import time
import cv2
import numpy as np
from cuda import cudart
import tensorrt as trt

import warnings
warnings.filterwarnings("ignore")


def get_img_path_batches(batch_size, img_dir):
    ret = []
    batch = []
    for root, dirs, files in os.walk(img_dir):
        for name in files:
            if len(batch) == batch_size:
                ret.append(batch)
                batch = []
            batch.append(os.path.join(root, name))
    if len(batch) > 0:
        ret.append(batch)
    return ret


class SRTRT(object):
    '''description: A SR class that warps TensorRT op'''

    def __init__(self, engine_file_path):
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # deserialize the engine from file
        with open(engine_file_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        # create a stream on this device
        _, stream = cudart.cudaStreamCreate()
        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            print('binding: ', binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))

            # allocate host and device buffers
            host_mem = np.empty(size, dtype=dtype)
            _, cuda_mem = cudart.cudaMallocAsync(host_mem.nbytes, stream)

            # append the device buffer to device binding
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)


                input_shape = engine.get_binding_shape(binding)

            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

                output_shape = engine.get_binding_shape(binding)

        # store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size
        self.input_shape = input_shape
        self.output_shape = output_shape

    def infer(self, raw_image_generator):

        threading.Thread.__init__(self)
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess
        batch_image_raw = []
        batch_origin_h = []
        batch_origin_w = []
        batch_input_image = np.empty(shape=[self.batch_size, 3, self.input_h, self.input_w])
        for i, image_raw in enumerate(raw_image_generator):
            input_image, image_raw, origin_h, origin_w = self.preprocess_image(image_raw)
            batch_image_raw.append(image_raw)
            batch_origin_h.append(origin_h)
            batch_origin_w.append(origin_w)
            np.copyto(batch_input_image[i], input_image)
        batch_input_image = np.ascontiguousarray(batch_input_image)

        # Copy input image to host buffer
        np.copyto(host_inputs[0], batch_input_image.ravel())
        start = time.time()
        # Transfer input data  to the GPU.
        cudart.cudaMemcpyAsync(cuda_inputs[0], host_inputs[0].ctypes.data, host_inputs[0].nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        # Run inference.
        context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream)
        # Transfer predictions back from the GPU.
        cudart.cudaMemcpyAsync(host_outputs[0].ctypes.data, cuda_outputs[0], host_outputs[0].nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        # Synchronize the stream
        cudart.cudaStreamSynchronize(stream)
        end = time.time()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]

        output = output.reshape(self.output_shape)

        batch_image_raw = self.post_process(output)

        return batch_image_raw, end - start

    def destroy(self):
        # Remove any stream and cuda mem
        cudart.cudaStreamDestroy(self.stream)
        cudart.cudaFree(self.cuda_inputs[0])
        cudart.cudaFree(self.cuda_outputs[0])

    def get_raw_image(self, image_path_batch):
        """
        description: Read an image from image path
        """
        for img_path in image_path_batch:
            yield cv2.imread(img_path)

    def get_raw_image_zeros(self, image_path_batch=None):
        """
        description: Ready data for warmup
        """
        for _ in range(self.batch_size):
            yield np.zeros([self.input_h, self.input_w, 3], dtype=np.uint8)

    def preprocess_image(self, raw_bgr_image):
        """
        description: Convert BGR image to RGB,
                     resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        image_raw = raw_bgr_image
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def post_process(self, output):
        """
        description: postprocess the prediction
        """
        output_imgs = []
        output_shape = output.shape
        for i in range(output_shape[0]):
            tmp = output[i, ...]
            tmp = np.transpose(tmp, [1, 2, 0])
            tmp = tmp[:, :, ::-1] * 255.0
            img = np.uint8(tmp)
            output_imgs.append(img)

        return output_imgs



class inferThread(threading.Thread):
    def __init__(self, sr_wrapper, image_path_batch):
        threading.Thread.__init__(self)
        self.sr_wrapper = sr_wrapper
        self.image_path_batch = image_path_batch

    def run(self):
        batch_image_raw, use_time = self.sr_wrapper.infer(self.sr_wrapper.get_raw_image(self.image_path_batch))
        for i, img_path in enumerate(self.image_path_batch):
            parent, filename = os.path.split(img_path)
            save_name = os.path.join('output', filename)
            # Save image
            cv2.imwrite(save_name, batch_image_raw[i])
        print('input->{}, time->{:.2f}ms, saving into output/'.format(self.image_path_batch, use_time * 1000))


class warmUpThread(threading.Thread):
    def __init__(self, sr_wrapper):
        threading.Thread.__init__(self)
        self.sr_wrapper = sr_wrapper

    def run(self):
        batch_image_raw, use_time = self.sr_wrapper.infer(self.sr_wrapper.get_raw_image_zeros())
        print('warm_up->{}, time->{:.2f}ms'.format(batch_image_raw[0].shape, use_time * 1000))


if __name__ == "__main__":
    # load custom plugin and engine
    # PLUGIN_LIBRARY = "build/libmyplugins.so"
    engine_file_path = "onnx_files/real_esrgan_anime.engine"

    if len(sys.argv) > 1:
        engine_file_path = sys.argv[1]
    if len(sys.argv) > 2:
        PLUGIN_LIBRARY = sys.argv[2]

    # ctypes.CDLL(PLUGIN_LIBRARY)
    cudart.cudaDeviceSynchronize()


    if os.path.exists('output/'):
        shutil.rmtree('output/')
    os.makedirs('output/')
    # a YoLov5TRT instance
    sr_wrapper = SRTRT(engine_file_path)
    try:
        print('batch size is', sr_wrapper.batch_size)

        image_dir = "samples/"
        image_path_batches = get_img_path_batches(sr_wrapper.batch_size, image_dir)

        for i in range(10):
            # create a new thread to do warm_up
            thread1 = warmUpThread(sr_wrapper)
            thread1.start()
            thread1.join()
        for batch in image_path_batches:
            # create a new thread to do inference
            thread1 = inferThread(sr_wrapper, batch)
            thread1.start()
            thread1.join()
    finally:
        # destroy the instance
        sr_wrapper.destroy()