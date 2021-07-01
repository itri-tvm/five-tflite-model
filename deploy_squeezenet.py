"""
Deploy Pre-Trained TensorFlow Lite squeezenet
=============================================
By Kuen-Wey Lin<kwlin@itri.org.tw>

The following two packages are required
pip3 install tensorflow
pip3 install tflite
Version 2.3.0 is tested
"""

######################################################################
# Set environment variables
# -------------------------

import tvm

target = 'llvm'
target_host = 'llvm'
ctx = tvm.cpu(0)

model_path = './squeezenet.tflite'
input_name = 'Placeholder'
data_type = 'float32' # input's data type
img_path = './image_classification/'

######################################################################
# Set input size
# --------------

batch_size = 1
num_class = 1000
image_dimention = 3
image_shape = (224, 224)
data_shape = (batch_size,) + image_shape + (image_dimention,)
out_shape = (batch_size, num_class)

######################################################################
# Load a TFLite model
# -------------------

import os
tflite_model_file = os.path.join(model_path)
tflite_model_buf = open(tflite_model_file, "rb").read()

# Get TFLite model from buffer
try:
    import tflite
    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

######################################################################
# Convert the TFLite model into Relay IR
# --------------------------------------

import tvm.relay as relay
dtype_dict = {input_name: data_type}
shape_dict = {input_name: data_shape}

mod, params = relay.frontend.from_tflite(tflite_model,
                                         shape_dict=shape_dict,
                                         dtype_dict=dtype_dict)

######################################################################
# Compile the Relay module
# ------------------------

with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize":True}):
    graph, lib, params = relay.build(mod, target=target, target_host=target_host, params=params)

######################################################################
# Get ImageNet lable
# ------------------

import tensorflow as tf
import numpy as np
labels_path = tf.keras.utils.get_file(
    'ImageNetLabels.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

######################################################################
# Create TVM runtime and do inference
# -----------------------------------

from tvm.contrib import graph_runtime
def get_tvm_accuracy(graph, lib, params, ctx, img_name):
    print("\n")
    # create module
    module = graph_runtime.create(graph, lib, ctx)

    from PIL import Image
    print("img_name:", img_name)
    image_data = Image.open(img_path+img_name).resize(image_shape)
    image_data = np.array(image_data).astype(data_type)
    image_data = np.expand_dims(image_data, axis=0)
    image_data[:, :, :, 0] = 2.0 / 255.0 * image_data[:, :, :, 0] - 1
    image_data[:, :, :, 1] = 2.0 / 255.0 * image_data[:, :, :, 1] - 1
    image_data[:, :, :, 2] = 2.0 / 255.0 * image_data[:, :, :, 2] - 1
    if image_data.shape == (1, 224, 224, 4):
        image_data = image_data[:,:,:, 0:3]
    print("img shape:", image_data.shape)

    # set ground truth
    ground_truth = int(os.path.splitext(img_name)[0])
    ground_truth += 1
    print("Ground truth: %s (ID: %d)" % (imagenet_labels[ground_truth], ground_truth))

    # set input and parameters
    module.set_input(input_name, tvm.nd.array(image_data))
    module.set_input(**params)

    # run
    import time
    timeStart = time.time()
    module.run()
    timeEnd = time.time()
    print("Inference time: %f" % (timeEnd - timeStart))

    # get output
    tvm_output = module.get_output(0).asnumpy()

    # print top-1
    #top1 = np.argmax(tvm_output[0])
    #print("Top-1: %s (ID: %d)" % (block.classes[top1], top1))
    # print top-5
    top5 = tvm_output[0].argsort()[-5:][::-1]
    print("TVM's prediction:", top5)
    check_top1 = 0
    check_top5 = 0
    for top_id in range(5):
        if top_id == 0 and top5[top_id] == ground_truth:
            check_top1 = 1
        if top5[top_id] == ground_truth:
            check_top5 = 1
        print("Top-%d: %s (ID: %d) " % (top_id+1, imagenet_labels[top5[top_id]], top5[top_id]))
    return check_top1, check_top5

file_list = os.listdir(img_path)
top1_total = 0
top5_total = 0
for file_name in file_list:
    top1_subtotal, top5_subtotal = get_tvm_accuracy(graph, lib, params, ctx, file_name)
    top1_total += top1_subtotal
    top5_total += top5_subtotal
print("\nNum of tested images:", len(file_list))
print("TVM Top-1 accuracy: {:.2%}; Top-5 accuracy: {:.2%}".format(top1_total/len(file_list), top5_total/len(file_list)))

print("Model:", model_path)

