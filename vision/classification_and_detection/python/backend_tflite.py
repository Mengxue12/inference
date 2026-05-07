"""
tflite backend (https://github.com/tensorflow/tensorflow/lite)
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

from threading import Lock

try:
    # try dedicated tflite package first
    import tflite_runtime
    import tflite_runtime.interpreter as tflite

    _version = tflite_runtime.__version__
    _git_version = tflite_runtime.__git_version__
except BaseException:
    # fall back to tflite bundled in tensorflow
    import tensorflow as tf
    from tensorflow.lite.python import interpreter as tflite

    _version = tf.__version__
    _git_version = tf.__git_version__

import numpy as np
import backend


class BackendTflite(backend.Backend):
    def __init__(self):
        super(BackendTflite, self).__init__()
        self.sess = None
        self.lock = Lock()
        self.fixed_batch_size = 1

    def version(self):
        return _version + "/" + _git_version

    def name(self):
        return "tflite"

    def image_format(self):
        # tflite is always NHWC
        return "NHWC"

    def load(self, model_path, inputs=None, outputs=None, use_tpu=False, max_batchsize=1):
        self.use_tpu = use_tpu
        self.fixed_batch_size = max(1, int(max_batchsize))
        if use_tpu:
            from pycoral.utils.edgetpu import make_interpreter

            self.sess = make_interpreter(model_path)
        else:
            self.sess = tflite.Interpreter(model_path=model_path)

        for input_detail in self.sess.get_input_details():
            shape_signature = input_detail["shape_signature"]
            print(f"input_detail: {input_detail}")
            input_shape = list(input_detail["shape"])
            if shape_signature[0] is None or shape_signature[0] == -1: # scenario 1: signature provides dynamic batch size
                input_shape[0] = self.fixed_batch_size
            elif shape_signature[0] != self.fixed_batch_size: # scenario 2: signature provides fixed batch size
                raise ValueError(f"Batch size {self.fixed_batch_size} does not match input shape signature {shape_signature}.")
           
            print(f"resizing tensor {input_detail['name']} to {input_shape}")
            self.sess.resize_tensor_input(input_detail["index"], input_shape)
        self.sess.allocate_tensors()
        # keep input/output name to index mapping
        self.input2index = {
            i["name"]: i["index"] for i in self.sess.get_input_details()
        }
        self.output2index = {
            i["name"]: i["index"] for i in self.sess.get_output_details()
        }
        # keep input/output names
        self.inputs = list(self.input2index.keys())
        self.outputs = list(self.output2index.keys())
        return self

    def predict(self, feed):
        first_input = self.inputs[0]
        actual_batch = int(feed[first_input].shape[0])

        with self.lock:
            # set inputs
            for k, v in self.input2index.items():
                input_data = feed[k]
                if input_data.shape[0] < self.fixed_batch_size:
                    pad_shape = (self.fixed_batch_size - input_data.shape[0],) + input_data.shape[1:]
                    input_data = np.concatenate(
                        [input_data, np.repeat(input_data[-1:], pad_shape[0], axis=0)], axis=0
                    )
                if self.use_tpu and self.sess.get_input_details()[v]["dtype"] == np.uint8:
                    input_scale, input_zero_point = self.sess.get_input_details()[v][
                        "quantization"
                    ]
                    input_data = input_data / input_scale + input_zero_point
                    input_data = input_data.astype(np.uint8)
                self.sess.set_tensor(v, input_data)
            self.sess.invoke()
            # get results
            res = [self.sess.get_tensor(v) for _, v in self.output2index.items()]

        return [r[:actual_batch] if len(r.shape) > 0 else r for r in res]
