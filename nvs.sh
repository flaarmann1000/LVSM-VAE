#! /bin/bash
# 
# Usage
# 
# 2 GPUs Training
# bash ./scripts/nvs.sh --model_space PX --ray_encoding plucker --pos_enc prope --gpus "0,1"
# bash ./scripts/nvs.sh --model_space PX --ray_encoding plucker --pos_enc gta --gpus "0,1"
# 
# 2 GPUs Testing (with zooming in)
# bash ./scripts/nvs.sh --model_space PX --ray_encoding plucker --pos_enc prope --gpus "0,1" --test-zoom-in "1 3 5"
#
# 2 GPUs Testing (with more context views)
# bash ./scripts/nvs.sh --model_space PX --ray_encoding plucker --pos_enc prope --gpus "0,1" --test-context-views "2 4 8 16"
#
# 2 GPUs Testing (with rendering video)
# bash ./scripts/nvs.sh --model_space PX --ray_encoding plucker --pos_enc prope --gpus "0,1" --test-render-video


# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --decode)
      DECODE="$2"
      shift 2
      ;;
    --overfit)
      OVERFIT="$2"
      shift 2
      ;;
    --model_space)
      MODEL_SPACE="$2"
      shift 2
      ;;
    --ray_encoding)
      RAY_ENCODING="$2"   
      shift 2
      ;;
    --pos_enc)
      POS_ENC="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --test-zoom-in)
      TEST_ZOOM_IN="$2"
      shift 2
      ;;
    --test-context-views)
      TEST_CONTEXT_VIEWS="$2"
      shift 2
      ;;
    --test-render-video)
      TEST_RENDER_VIDEO=true
      shift 1          
      ;;
    --upscale)
      UPSCALE="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 --ray_encoding <ray_encoding> --pos_enc <pos_enc> --gpus <gpu_list> [--test-zoom-in <zoom_factors>]"
      echo "  --model_space: PX or VAE"
      echo "  --ray_encoding: PLUCKER, CAMRAY, NONE, or RAYMAP"
      echo "  --pos_enc: PROPE, GTA, or NONE"
      echo "  --gpus: comma-separated GPU list (e.g., '0,1')"
      echo "  --test-zoom-in: space-separated zoom factors for testing (e.g., '3 5')"
      echo "  --test-context-views: space-separated context views for testing (e.g., '2 4 8 16')"
      echo "  --test-render-video: render video for testing"
      exit 0
      ;;
    *)
      echo "Unknown option $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# defaults
OVERFIT="${OVERFIT:-0}"
DECODE="${DECODE:-1}"
UPSCALE="${UPSCALE:-1}"



# Check required arguments
if [ -z "$MODEL_SPACE" ]; then
  echo "Error: --model_space is required"
  exit 1
fi


if [ -z "$RAY_ENCODING" ]; then
  echo "Error: --ray_encoding is required"
  exit 1
fi

if [ -z "$POS_ENC" ]; then
  echo "Error: --pos_enc is required"
  exit 1
fi

if [ -z "$GPUS" ]; then
  echo "Error: --gpus is required"
  exit 1
fi


NGPUS=$(echo $GPUS | tr ',' '\n' | wc -l)

NAME="release-${NGPUS}gpus-b8-s1-80k"
BASE_CMD=(
    "NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc-per-node=$NGPUS"
    "main.py lvsm"
    "--decode ${DECODE}"
    "--model_space ${MODEL_SPACE}"
    "--overfit ${OVERFIT}"
    "--upscale ${UPSCALE}"
    "--amp --amp_dtype fp16"
    "--dataset_batch_scenes 8"
    "--dataset_supervise_views 1"
    "--model_config.encoder.num_layers 4"
    "--model_config.encoder.layer.d_model 768"
    "--model_config.encoder.layer.nhead 16"
    "--model_config.encoder.layer.dim_feedforward 1024"
    "--model_config.encoder.layer.qk_norm"
    # "--max_steps 5800 --test_every 5500"
    "--max_steps 15000 --test_every 1000"  
    "--model_config.ray_encoding ${RAY_ENCODING}"
    "--model_config.pos_enc ${POS_ENC}"
    "--output_dir results/nvs/${MODEL_SPACE}/${NAME}-${RAY_ENCODING}-${POS_ENC}"
)

echo "NAME: ${NAME}"
echo "MODEL_SPACE: ${MODEL_SPACE}"
echo "OVERFIT: ${OVERFIT}"
echo "RAY_ENCODING: ${RAY_ENCODING}"
echo "POS_ENC: ${POS_ENC}"

if [ -n "$TEST_ZOOM_IN" ]; then
    for zoom_factor in $TEST_ZOOM_IN; do
        echo "Starting testing with zoom factor ${zoom_factor}..."
        CMD=(
            "${BASE_CMD[@]}"
            "--test_only --auto_resume"
            "--test_zoom_factor ${zoom_factor}"
            "--test_subdir eval-zoom${zoom_factor}x"
        )
        CUDA_VISIBLE_DEVICES=$GPUS eval "${CMD[@]}"
    done
    exit 0
elif [ -n "$TEST_CONTEXT_VIEWS" ]; then
    for context_views in $TEST_CONTEXT_VIEWS; do
        echo "Starting testing with ${context_views} context views..."
        CMD=(
            "${BASE_CMD[@]}"
            "--test_only --auto_resume"
            "--model_config.ref_views ${context_views}"
            "--test_input_views ${context_views}"
            "--test_index_fp evaluation_index_re10k_context${context_views}.json"
            "--test_subdir eval-context${context_views}"
        )
        CUDA_VISIBLE_DEVICES=$GPUS eval "${CMD[@]}"
    done
    exit 0
elif [ -n "$TEST_RENDER_VIDEO" ]; then
    echo "Starting testing with rendering video for fisrt 10 scenes ..."
    CMD=(
        "${BASE_CMD[@]}"
        "--test_only --auto_resume --render_video --test_n 10"
    )
    CUDA_VISIBLE_DEVICES=$GPUS eval "${CMD[@]}"
    exit 0
else
    echo "Starting training process..."
    CMD=(
        "${BASE_CMD[@]}"
    )
    CUDA_VISIBLE_DEVICES=$GPUS eval "${CMD[@]}"
    exit 0
fi