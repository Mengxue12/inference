"""
tflite backend (https://github.com/tensorflow/tensorflow/lite)
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

from threading import Lock

try:
    # try dedicated tflite package first
    import ai_edge_litert
    from ai_edge_litert import interpreter as tflite

    _version = ai_edge_litert.__version__
    _git_version = ai_edge_litert.__version__
except BaseException:
    # fall back to tflite bundled in tensorflow
    print("Falling back to tensorflow tflite")
    import tensorflow as tf
    from tensorflow.lite.python import interpreter as tflite

    _version = tf.__version__
    _git_version = tf.__git_version__

import numpy as np
import backend

def _quantize_float_to_integer(input_data, scale, zero_point, dst_dtype):
    """Map real values to quantized tensor values using TFLite per-tensor (scale, zp)."""
    if scale is None or scale == 0:
        raise ValueError(
            "Input expects quantized dtype {} but quantization scale is missing or zero".format(
                dst_dtype
            )
        )
    zp = 0.0 if zero_point is None else float(zero_point)
    scaled = np.rint(input_data.astype(np.float64) / float(scale) + zp)
    dt = np.dtype(dst_dtype)
    if dt == np.uint8:
        return np.clip(scaled, 0, 255).astype(np.uint8)
    if dt == np.int8:
        return np.clip(scaled, -128, 127).astype(np.int8)
    if dt == np.int16:
        return np.clip(scaled, -32768, 32767).astype(np.int16)
    if dt == np.uint16:
        return np.clip(scaled, 0, 65535).astype(np.uint16)
    raise ValueError("Unsupported quantized input dtype: {}".format(dst_dtype))


def _prepare_input_tensor(input_data, input_detail):
    """Cast / quantize feed tensors to match interpreter input dtype."""
    want_dtype = input_detail["dtype"]
    want = np.dtype(want_dtype)
    if np.issubdtype(want, np.floating):
        if np.issubdtype(input_data.dtype, np.floating):
            return np.asarray(input_data, dtype=np.float32)
        return input_data
    if not np.issubdtype(want, np.integer): # in case the shape signature is not an integer, pass through
        return input_data
    # Integer input (e.g. uint8 / int8): pass through if dtype already matches.
    if np.issubdtype(input_data.dtype, want):
        return np.asarray(input_data, dtype=want)
    if not np.issubdtype(input_data.dtype, np.floating):
        return np.asarray(input_data, dtype=want)
    # Integer input: quantize float to integer
    scale, zero_point = input_detail.get("quantization") or (None, None)
    return _quantize_float_to_integer(input_data, scale, zero_point, want)


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

    def load(
        self,
        model_path,
        inputs=None,
        outputs=None,
        use_tpu=False,
        max_batchsize=1,
        image_size=None,
        inference_threads=None,
    ):
        self.use_tpu = use_tpu
        self.fixed_batch_size = max(1, int(max_batchsize))
        if use_tpu:
            from pycoral.utils.edgetpu import make_interpreter

            self.sess = make_interpreter(model_path)
        else:
            if inference_threads is None:
                self.sess = tflite.Interpreter(model_path=model_path)
            else:
                self.sess = tflite.Interpreter(
                    model_path=model_path, num_threads=int(inference_threads)
                )

        # NHWC targets derived from dataset side. Any of these may be None,
        # meaning "fall back to the model's static shape".
        target_h = target_w = target_c = None
        if image_size is not None:
            if len(image_size) < 2:
                raise ValueError(
                    "image_size must be [H, W] or [H, W, C], got {}".format(image_size)
                )
            target_h = int(image_size[0])
            target_w = int(image_size[1])
            if len(image_size) > 2:
                target_c = int(image_size[2])

        print(f"output details: {self.sess.get_output_details()}")

        for input_detail in self.sess.get_input_details():
            input_shape = [int(x) for x in input_detail["shape"]]
            raw_sig = input_detail.get("shape_signature")
            if raw_sig is None:
                shape_signature = list(input_shape)
            else:
                shape_signature = list(raw_sig)

            print(f"input_detail: {input_detail}")

            # batch (NHWC dim 0)
            if len(shape_signature) > 0 and len(input_shape) > 0:
                sig0 = shape_signature[0]
                if sig0 is None or int(sig0) == -1:
                    input_shape[0] = self.fixed_batch_size
                elif int(sig0) != self.fixed_batch_size:
                    raise ValueError(
                        "Batch size {} does not match input shape signature {}.".format(
                            self.fixed_batch_size, shape_signature
                        )
                    )

            # spatial H, W and channel C (NHWC dims 1, 2, 3)
            if len(input_shape) >= 4 and len(shape_signature) >= 4:
                dim_targets = {1: ("H", target_h), 2: ("W", target_w), 3: ("C", target_c)}
                for idx, (dim_name, want) in dim_targets.items():
                    sig_i = shape_signature[idx]
                    if sig_i is None or int(sig_i) == -1:
                        if want is not None:
                            input_shape[idx] = want
                        elif input_shape[idx] <= 0:
                            input_shape[idx] = 224 if idx in (1, 2) else 3
                    else:
                        fixed = int(sig_i)
                        if want is not None and fixed != want:
                            raise ValueError(
                                "Input {}: {} is fixed to {} but image_size requests {}.".format(
                                    input_detail.get("name", "?"), dim_name, fixed, want
                                )
                            )
                        input_shape[idx] = fixed

            print("resizing tensor {} to {}".format(input_detail["name"], input_shape))
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
                input_detail = self.sess.get_input_details()[v]
                input_data = _prepare_input_tensor(input_data, input_detail)
                self.sess.set_tensor(v, input_data)
            self.sess.invoke()
            # get results
            res = [self.sess.get_tensor(v) for _, v in self.output2index.items()]

        return [r[:actual_batch] if len(r.shape) > 0 else r for r in res]
