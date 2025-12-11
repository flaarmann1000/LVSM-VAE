#!/bin/bash

##############################################################################
# Multi-run launcher for LVSM + NVS script
#
# Example:
# bash run_multi.sh \
#   --gpus 0 \
#   --cmd " --overfit 1 --model_space VAE --ray_encoding CAMRAY --pos_enc PROPE --max_steps 20000 " \
#   --cmd " --overfit 1 --model_space VAE --ray_encoding CAMRAY --pos_enc PROPE --norm 1 --max_steps 20000 " \
#   --cmd " --overfit 1 --model_space VAE --ray_encoding CAMRAY --pos_enc PROPE --norm 1 --patch_size 4 "
#
##############################################################################

GPUS=""
CMDS=()

# -------------------- Parse Arguments --------------------
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --cmd)
      CMDS+=("$2")
      shift 2
      ;;
    -h|--help)
      echo "Usage:"
      echo "  bash run_multi.sh --gpus <GPU_LIST> --cmd \"<nvs arguments>\" [--cmd \"<nvs arguments>\"] ..."
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [ -z "$GPUS" ]; then
    echo "ERROR: --gpus is required"
    exit 1
fi

if [ ${#CMDS[@]} -eq 0 ]; then
    echo "ERROR: Provide at least one --cmd \"...\" entry"
    exit 1
fi

echo ""
echo "======================================================="
echo " Running ${#CMDS[@]} LVSM experiments sequentially"
echo " GPUs: $GPUS"
echo "======================================================="
echo ""

# -------------------- Run all experiments --------------------
for i in "${!CMDS[@]}"; do
    echo ""
    echo "-------------------------------------------------------"
    echo "Experiment $((i+1)) / ${#CMDS[@]}"
    echo "Args: ${CMDS[$i]}"
    echo "-------------------------------------------------------"
    echo ""

    CMD="CUDA_VISIBLE_DEVICES=$GPUS bash ./nvs.sh \
          ${CMDS[$i]} \
          --gpus \"$GPUS\""

    echo "Executing:"
    echo "$CMD"
    
    eval $CMD

    echo ""
    echo "Finished experiment $((i+1))"
    echo "-------------------------------------------------------"
done

echo "All experiments completed."
