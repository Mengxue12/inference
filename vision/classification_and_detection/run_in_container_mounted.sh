#!/bin/bash

set -euo pipefail

usage() {
    cat <<EOF
Usage:
  $0 --backend <backend> --model_base <model_base_csv> --device <device> \\
     --scenario <scenario> \\
     --inference_threads <threads_csv> --max-batchsize <batchsize> \\
     --cache <0|1> \\
     --preprocessed_dir <dir> \\
     [--output-dir <dir>] [extra args ...]

  If extra args include --accuracy, the default output path uses a sibling
  directory \"accuracy\" instead of \"run_<NUM_RUN>\" under each run_name.

Example:
  QUANTIZATION_TYPE="int8,fp16" RESOLUTION="224,192" PLATFORM=cpu NUM_RUN=1 \\
  $0 --backend tflite --model_base "mobilenetv2,resnet50v2" --device cpu \\
     --scenario SingleStream \\
     --inference_threads "2,4" \\
     --max-batchsize 1  --cache 1--preprocessed_dir /data/preprocessed

Environment variables:
  QUANTIZATION_TYPE   quantization type CSV string (required unless --quantization_type is given)
  RESOLUTION          resolution CSV string (optional)
  PLATFORM            output dir platform part (default: device)
  NUM_RUN             output dir run number (default: 1)
EOF
}

backend=""
model_base="${MODEL_BASE:-}"
device=""
scenario="${SCENARIO:-SingleStream}"
preprocessed_dir=""
cache=""
cache_dir=""
use_preprocessed_dataset=""
custom_output_dir=""
quantization_type="${QUANTIZATION_TYPE:-}"
resolution="${RESOLUTION:-}"
inference_threads="${INFERENCE_THREADS:-}"
max_batchsize="${BATCHSIZE:-}"
platform="${PLATFORM:-}"
num_run="${NUM_RUN:-1}"
extra_cli_args=()

while [ $# -gt 0 ]; do
    case "$1" in
        --backend)
            backend="${2:-}"
            shift 2
            ;;
        --model_base)
            model_base="${2:-}"
            shift 2
            ;;
        --device)
            device="${2:-}"
            shift 2
            ;;
        --resolution)
            resolution="${2:-}"
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
        --inference_threads)
            inference_threads="${2:-}"
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
        --output-base-dir)
            output_base_dir="${2:-}"
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

want_accuracy_run=0
for _arg in "${extra_cli_args[@]}"; do
    if [[ "$_arg" == *--accuracy* ]]; then
        want_accuracy_run=1
        break
    fi
done

# check if the required arguments are set
if [ -z "$backend" ] || [ -z "$model_base" ] || [ -z "$device" ] || \
   [ -z "$quantization_type" ]; then
    usage
    exit 1
fi

# check if the data and model directories are set
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

# Write experiment manifest at the run root (OUTPUT_DIR); MLPerf logs live in OUTPUT_DIR/logs.
write_run_manifest() {
    local out_dir="$1" started_at="$2" ended_at="$3" exit_code="$4"
    MANIFEST_OUT_DIR="$(cd "$out_dir" && pwd)" \
    MANIFEST_STARTED_AT="$started_at" \
    MANIFEST_ENDED_AT="$ended_at" \
    MANIFEST_EXIT_CODE="$exit_code" \
    MANIFEST_MODEL_NAME="$model_item" \
    MANIFEST_QUANT_ITEM="$quant_item" \
    MANIFEST_RESOLUTION="$resolution" \
    MANIFEST_INFERENCE_THREADS="$thread_item" \
    MANIFEST_MAX_BATCHSIZE="${max_batchsize:-}" \
    MANIFEST_PLATFORM="$platform" \
    MANIFEST_SCENARIO="$scenario" \
    MANIFEST_BACKEND="$backend" \
    MANIFEST_DEVICE="$device" \
    MANIFEST_ACCURACY="$want_accuracy_run" \
    python3 - <<'PY'
import json
import os

def _int_field(name):
    raw = os.environ.get(name, "") or ""
    raw = raw.strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return raw

out = os.environ["MANIFEST_OUT_DIR"]
exit_code = os.environ.get("MANIFEST_EXIT_CODE", "1")
status = "completed" if exit_code == "0" else "failed"

data = {
    "model_name": os.environ["MANIFEST_MODEL_NAME"],
    "quant_item": os.environ["MANIFEST_QUANT_ITEM"],
    "resolution": _int_field("MANIFEST_RESOLUTION"),
    "inference_threads": _int_field("MANIFEST_INFERENCE_THREADS"),
    "max_batchsize": _int_field("MANIFEST_MAX_BATCHSIZE"),
    "platform": os.environ["MANIFEST_PLATFORM"],
    "scenario": os.environ["MANIFEST_SCENARIO"],
    "backend": os.environ["MANIFEST_BACKEND"],
    "device": os.environ["MANIFEST_DEVICE"],
    "accuracy": os.environ.get("MANIFEST_ACCURACY", "0") == "1",
    "mlperf_logs_dir": "logs",
    "started_at": os.environ["MANIFEST_STARTED_AT"],
    "ended_at": os.environ["MANIFEST_ENDED_AT"],
    "status": status,
    "run_exit_code": _int_field("MANIFEST_EXIT_CODE"),
}

path = os.path.join(out, "manifest.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
PY
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
    cache_dir="$DATA_DIR"
fi


# Resolution sweep list from env RESOLUTION (comma-separated); empty => model default.
if [ -z "$resolution" ]; then
    resolution_values="__USE_DEFAULT__"
else
    resolution_values="$(split_csv_to_lines "$resolution")"
fi

model_values="$(split_csv_to_lines "$model_base")"
quant_values="$(split_csv_to_lines "$quantization_type")"
thread_values="$(split_csv_to_lines "$inference_threads")"

WAIT_BEFORE_SECONDS=${WAIT_BEFORE_SECONDS:-10}
WAIT_AFTER_SECONDS=${WAIT_AFTER_SECONDS:-10}
if [[ "$WAIT_BEFORE_SECONDS" =~ ^[0-9]+$ ]] && (( WAIT_BEFORE_SECONDS > 0 )); then
    echo "Waiting for $WAIT_BEFORE_SECONDS seconds before running experiments"
    sleep "$WAIT_BEFORE_SECONDS"
fi

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

            model_path="$MODEL_DIR/${model_item}${quant_item}/${modelversion}/model.tflite"

            while IFS= read -r thread_item; do
                echo "****************************************************"
                echo "Running model_item=$model_item, quant_item=$quant_item, resolution=$resolution, thread_item=$thread_item, platform=$platform"
                run_name="$(sanitize_component "${model_item}_${quant_item}_${resolution}_${thread_item}_${platform}")"
                if [ -n "$custom_output_dir" ]; then
                    if [ "$want_accuracy_run" -eq 1 ]; then
                        _out_base="$custom_output_dir"
                        _out_leaf="${_out_base##*/}"
                        if [[ "$_out_leaf" == run_* ]]; then
                            OUTPUT_DIR="${_out_base%/*}/accuracy"
                        else
                            OUTPUT_DIR="${_out_base}/accuracy"
                        fi
                    else
                        OUTPUT_DIR="$custom_output_dir"
                    fi
                else
                    _run_root="$output_base_dir/$scenario/$run_name"
                    if [ "$want_accuracy_run" -eq 1 ]; then
                        OUTPUT_DIR="$_run_root/accuracy"
                    else
                        OUTPUT_DIR="$_run_root/run_${num_run}"
                    fi
                fi
                mkdir -p "$OUTPUT_DIR"
                LOGS_DIR="$OUTPUT_DIR/logs"
                mkdir -p "$LOGS_DIR"

                opts="--backend $backend \
--model $model_path \
--device $device \
--scenario $scenario \
--cache $cache \
--cache_dir $cache_dir \
--dataset imagenet_tflite \
--dataset-path $DATA_DIR \
--resolution $resolution \
--inference_threads $thread_item \
--max-batchsize $max_batchsize \
--output $LOGS_DIR \
--model-name $model_item"

                if [ "$use_preprocessed_dataset" = "1" ] || [ "$use_preprocessed_dataset" = "true" ]; then
                    opts="$opts --use_preprocessed_dataset"
                fi

                if [ ${#extra_cli_args[@]} -gt 0 ]; then
                    for arg in "${extra_cli_args[@]}"; do
                        opts="$opts $arg"
                    done
                fi

                echo "Using OUTPUT_DIR=$OUTPUT_DIR (MLPerf logs under $LOGS_DIR)"
                echo "Resolved model_path=$model_path"
                echo "Running mounted run_lite.sh in container context..."
                echo "opts: $opts"

                run_started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
                set +e
                opts="$opts" bash ./run_lite.sh 2>&1 | tee "$LOGS_DIR/output.txt"
                run_exit="${PIPESTATUS[0]}"
                set -euo pipefail
                run_ended_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
                write_run_manifest "$OUTPUT_DIR" "$run_started_at" "$run_ended_at" "$run_exit"
                if [ "$run_exit" -ne 0 ]; then
                    echo "Run failed with exit code $run_exit" >&2
                    exit "$run_exit"
                fi
                echo "sleep 60"
                sleep 60
            done <<< "$thread_values"
        done <<< "$resolution_values"
    done <<< "$quant_values"
done <<< "$model_values"
