#!/bin/bash

source run_common.sh

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

# copy the config to cwd so the docker contrainer has access
cp ../../mlperf.conf .

OUTPUT_DIR=${OUTPUT_DIR:-`pwd`/output/$name}
_acc_args="$extra_args $EXTRA_OPS $*"
if [[ "$_acc_args" == *--accuracy* ]]; then
    OUTPUT_DIR="$OUTPUT_DIR/accuracy"
fi
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

image=mlperf-infer-imgclassify-$device
docker build  -t $image:v5.1-tflite -f Dockerfile.tflite .
opts="--profile $profile --model $model_path \
    --dataset-path $DATA_DIR --output /output $extra_args $@"
echo "opts: $opts"

docker run $gpus -e opts="$opts" \
    -v $DATA_DIR:$DATA_DIR -v $MODEL_DIR:$MODEL_DIR -v `pwd`:/mlperf \
    -v $OUTPUT_DIR:/output -v /proc:/host_proc \
    -t $image:v5.1-tflite /mlperf/run_lite.sh 2>&1 | tee $OUTPUT_DIR/output.txt
