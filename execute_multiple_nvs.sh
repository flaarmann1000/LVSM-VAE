#! /bin/bash

#######################################################################
### DEFINE YOUR ENCODING COMBINATIONS HERE (RAY_ENCODING POS_ENC) ####
#######################################################################
COMBINATIONS=(
    "PLUCKER GTA"
    "PLUCKER NONE"
    "CAMRAY PROPE"
    # Add more combinations as:
    # "ray_encoding_name pos_encoding_name"
)
#######################################################################


# Parse command line arguments (only GPUs + test options now)
while [[ $# -gt 0 ]]; do
  case $1 in
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
    -h|--help)
      echo "Usage: $0 --gpus <gpu_list> [--test-zoom-in <zoom_factors>] [--test-context-views <values>] [--test-render-video]"
      echo ""
      echo "Define your combinations inside the script in the COMBINATIONS array."
      exit 0
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done


# Check required GPU argument
if [ -z "$GPUS" ]; then
  echo "Error: --gpus is required"
  exit 1
fi


NGPUS=$(echo $GPUS | tr ',' '\n' | wc -l)

BASE_CMD_COMMON=(
    "--amp --amp_dtype fp16"
    "--dataset_batch_scenes 8"
    "--dataset_supervise_views 1"
    "--model_config.encoder.num_layers 4"
    "--model_config.encoder.layer.d_model 768"
    "--model_config.encoder.layer.nhead 16"
    "--model_config.encoder.layer.dim_feedforward 1024"
    "--model_config.encoder.layer.qk_norm"
    "--max_steps 100000 --test_every 80000"
)


for combo in "${COMBINATIONS[@]}"; do

  set -- $combo
  RAY_ENCODING=$1
  POS_ENC=$2

  NAME="release-${NGPUS}gpus-b8-s1-80k"

  echo "=========================================================="
  echo "Running combination:"
  echo "  RAY_ENCODING: $RAY_ENCODING"
  echo "  POS_ENCODING: $POS_ENC"
  echo "=========================================================="

  BASE_CMD=(
      "NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc-per-node=$NGPUS"
      "main.py lvsm"
      "${BASE_CMD_COMMON[@]}"
      "--model_config.ray_encoding ${RAY_ENCODING}"
      "--model_config.pos_enc ${POS_ENC}"
      "--output_dir results/nvs/${NAME}-${RAY_ENCODING}-${POS_ENC}"
  )


    if [ -n "$TEST_ZOOM_IN" ]; then
      for zoom_factor in $TEST_ZOOM_IN; do
          echo "Testing zoom factor $zoom_factor..."
          CMD=(
              "${BASE_CMD[@]}"
              "--test_only --auto_resume"
              "--test_zoom_factor ${zoom_factor}"
              "--test_subdir eval-zoom${zoom_factor}x"
          )
          CUDA_VISIBLE_DEVICES=$GPUS eval "${CMD[@]}"
      done
      continue
  fi

  if [ -n "$TEST_CONTEXT_VIEWS" ]; then
      for context_views in $TEST_CONTEXT_VIEWS; do
          echo "Testing with ${context_views} context views..."
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
      continue
  fi

  if [ -n "$TEST_RENDER_VIDEO" ]; then
      echo "Testing by rendering video for first 10 scenes..."
      CMD=(
          "${BASE_CMD[@]}"
          "--test_only --auto_resume --render_video --test_n 10"
      )
      CUDA_VISIBLE_DEVICES=$GPUS eval "${CMD[@]}"
      continue
  fi

  echo "Starting training..."
  CMD=("${BASE_CMD[@]}")
  CUDA_VISIBLE_DEVICES=$GPUS eval "${CMD[@]}"

done

echo "All combinations finished."
