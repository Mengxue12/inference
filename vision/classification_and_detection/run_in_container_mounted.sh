#!/bin/bash

set -euo pipefail

usage() {
    cat <<EOF
Usage:
  $0 --backend <backend> --model_name <model_name_csv> --device <device> \\
     --scenario <scenario> --max-batchsize <n> \\
     --inference_thread <threads_csv> --preprocessed_dir <dir> --cache <0|1> \\
     [--output-dir <dir>] [extra args ...]

Example:
  QUANTIZATION_TYPE="int8,fp16" RESOLUTION="224,192" PLATFORM=cpu NUM_RUN=1 \\
  $0 --backend tflite --model_name "mobilenetv2,resnet50v2" --device cpu \\
     --scenario SingleStream --max-batchsize 1 \\
     --inference_thread "2,4" --preprocessed_dir /data/preprocessed --cache 1

Environment variables:
  QUANTIZATION_TYPE   quantization type CSV string (required unless --quantization_type is given)
  RESOLUTION          resolution CSV string (optional)
  PLATFORM            output dir platform part (default: device)
  NUM_RUN             output dir run number (default: 1)
EOF
}

backend=""
model_name=""
device=""
scenario=""
max_batchsize=""
inference_thread=""
preprocessed_dir=""
cache=""
cache_dir=""
use_preprocessed_dataset=""
custom_output_dir=""
quantization_type="${QUANTIZATION_TYPE:-}"
resolution_env="${RESOLUTION:-}"
platform="${PLATFORM:-}"
num_run="${NUM_RUN:-1}"
extra_cli_args=()

while [ $# -gt 0 ]; do
    case "$1" in
        --backend)
            backend="${2:-}"
            shift 2
            ;;
        --model_name)
            model_name="${2:-}"
            shift 2
            ;;
        --device)
            device="${2:-}"
            shift 2
            ;;
        --resolution)
            resolution_env="${2:-}"
            shift 2
            ;;
        --scenario)
            scenario="${2:-}"
            shift 2
            ;;
        --max-batchsize)
            max_batchsize="${2:-}"
            shift 2
            ;;
        --inference_thread)
            inference_thread="${2:-}"
            shift 2
            ;;
        --cache)
            cache="${2:-}"
            shift 2
            ;;
        --cache_dir)
            cache_dir="${2:-}"
            shift 2
            ;;
        --quantization_type)
            quantization_type="${2:-}"
            shift 2
            ;;
        --platform)
            platform="${2:-}"
            shift 2
            ;;
        --num_run)
            num_run="${2:-}"
            shift 2
            ;;
        --preprocessed_dir)
            preprocessed_dir="${2:-}"
            shift 2
            ;;
        --use_preprocessed_dataset)
            use_preprocessed_dataset="${2:-}"
            shift 2
            ;;
        --output-dir)
            custom_output_dir="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            extra_cli_args+=("$1")
            shift
            ;;
    esac
done

if [ -z "$backend" ] || [ -z "$model_name" ] || [ -z "$device" ] || \
   [ -z "$scenario" ] || [ -z "$max_batchsize" ] || \
   [ -z "$inference_thread" ] || [ -z "$cache" ] || \
   [ -z "$quantization_type" ]; then
    usage
    exit 1
fi

if [ "x${DATA_DIR:-}" = "x" ]; then
    echo "DATA_DIR not set"
    exit 1
fi

if [ "x${MODEL_DIR:-}" = "x" ]; then
    echo "MODEL_DIR not set"
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

sanitize_component() {
    local value="$1"
    echo "$value" | sed -E 's/[[:space:],|;:\/]+/_/g'
}

split_csv_to_lines() {
    local value="$1"
    echo "$value" | tr ',' '\n' | sed -E 's/^[[:space:]]+//;s/[[:space:]]+$//' | sed '/^$/d'
}

declare -A MODEL_CONFIGS=(
    ["mobilenetv2"]="100224:224"
    ["efficientnetb0"]="224:224"
    ["nasnetmobile"]="224:224"
    ["resnet50v2"]="224:224"
)

if [ -z "$platform" ]; then
    platform="$device"
fi

if [ -z "$cache_dir" ]; then
    cache_dir="$DATA_DIR/preprocessed"
fi

if [ -z "$use_preprocessed_dataset" ]; then
    use_preprocessed_dataset="1"
fi

if [ -z "$resolution_env" ]; then
    resolution_values=""
else
    resolution_values="$(split_csv_to_lines "$resolution_env")"
fi
if [ -z "$resolution_values" ]; then
    resolution_values="__USE_DEFAULT__"
fi

model_values="$(split_csv_to_lines "$model_name")"
quant_values="$(split_csv_to_lines "$quantization_type")"
thread_values="$(split_csv_to_lines "$inference_thread")"

run_counter=0
while IFS= read -r model_item; do
    if [ -z "${MODEL_CONFIGS[$model_item]:-}" ]; then
        echo "Unsupported model for MODEL_CONFIGS: $model_item"
        exit 1
    fi

    config="${MODEL_CONFIGS[$model_item]}"
    default_modelversion="${config%%:*}"
    default_resolution="${config##*:}"

    while IFS= read -r quant_item; do
        while IFS= read -r res_item; do
            if [ "$res_item" = "__USE_DEFAULT__" ]; then
                resolution="$default_resolution"
            else
                resolution="$res_item"
            fi

            if [ "$model_item" = "mobilenetv2" ]; then
                modelversion="100${resolution}"
            else
                modelversion="$resolution"
            fi

            model_path="$MODEL_DIR/${model_item}${quant_item}/${modelversion}"

            while IFS= read -r thread_item; do
                run_name="$(sanitize_component "${model_item}_${quant_item}_${resolution}_${thread_item}_${platform}_${scenario}")"
                OUTPUT_DIR="${custom_output_dir:-${OUTPUT_DIR:-$SCRIPT_DIR/output/$run_name/run_${num_run}}}"
                mkdir -p "$OUTPUT_DIR"

                opts="--model $model_path \
                    --cache $cache --cache_dir $cache_dir \
                    --dataset imagenet_tflite --dataset-path $DATA_DIR \
                    --backend $backend --device $device \
                    --resolution $resolution --scenario $scenario \
                    --max-batchsize $max_batchsize --inference_threads $thread_item \
                    --output /output --preprocessed_dir $preprocessed_dir \
                    --model-name $model_item"

                if [ "$use_preprocessed_dataset" = "1" ] || [ "$use_preprocessed_dataset" = "true" ]; then
                    opts="$opts --use_preprocessed_dataset"
                fi

                if [ ${#extra_cli_args[@]} -gt 0 ]; then
                    for arg in "${extra_cli_args[@]}"; do
                        opts="$opts $arg"
                    done
                fi

                echo "Using OUTPUT_DIR=$OUTPUT_DIR"
                echo "Resolved model_path=$model_path"
                echo "Running mounted run_helper.sh in container context..."

                opts="$opts" bash ./run_helper.sh 2>&1 | tee "$OUTPUT_DIR/output.txt"
            done <<< "$thread_values"
        done <<< "$resolution_values"
    done <<< "$quant_values"
done <<< "$model_values"
