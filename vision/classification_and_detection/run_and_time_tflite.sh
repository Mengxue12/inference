#!/bin/bash

if [ "x$DATA_DIR" == "x" ]; then
    echo "DATA_DIR not set" && exit 1
fi
if [ "x$MODEL_DIR" == "x" ]; then
    echo "MODEL_DIR not set" && exit 1
fi

model_path="$MODEL_DIR"

# defaults
backend=tflite
model=resnet50
device="cpu"

for i in $* ; do
    case $i in
       tf|onnxruntime|tflite|pytorch|tvm-onnx|tvm-pytorch|tvm-tflite|ncnn) backend=$i; shift;;
       cpu|gpu|tpu|rocm) device=$i; shift;;
       gpu) device=gpu; shift;;
       resnet50|mobilenet|ssd-mobilenet|ssd-resnet34|ssd-resnet34-tf|retinanet|efficientnet) model=$i; shift;;
    esac
done

dockercmd=docker
if [ $device == "gpu" ]; then
    version=$(docker version -f "{{.Server.Version}}")
    major_version=$(echo "$version"| cut -d'.' -f 1)
    minor_version=$(echo "$version"| cut -d'.' -f 2)
    if [ $major_version -gt 19 ]; then
        gpus="--gpus all"
    elif [ $major_version -ge 19 ] && \
        [ $minor_version -ge 03 ]; then
        gpus="--gpus all"
    else
        gpus="--runtime=nvidia"
    fi
fi

OUTPUT_DIR=${OUTPUT_DIR:-`pwd`/output/"$backend-$device/$model"}
_acc_args="$EXTRA_OPS $*"
if [[ "$_acc_args" == *--accuracy* ]]; then
    OUTPUT_DIR="$OUTPUT_DIR/accuracy"
fi
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

image=mlperf-infer-imgclassify-$device
docker build  -t $image:v5.1-tflite -f Dockerfile.tflite .
opts="--model $model_path --model-name $model \
--backend $backend --device $device --cache 1 \
--dataset imagenet_tflite --dataset-path $DATA_DIR \
--output /output $EXTRA_OPS $@"
echo "opts: $opts"

# /mlperf comes from the image (Dockerfile.tflite). To override with a host checkout: add -v "$(pwd)":/mlperf
docker run $gpus -e opts="$opts" \
    -v $DATA_DIR:$DATA_DIR -v $MODEL_DIR:$MODEL_DIR \
    -v $OUTPUT_DIR:/output -v /proc:/host_proc \
    -t $image:v5.1-tflite /mlperf/run_lite.sh 2>&1 | tee $OUTPUT_DIR/output.txt
