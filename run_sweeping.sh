#!/bin/bash
#
# Training script for poisson_lin_lin model - sweeping experiment with tmux
# Runs in tmux session for persistent execution
#

SESSION_NAME="sweep_train"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=============================================================================="
echo "POISSON VAE TRAINING - SWEEPING EXPERIMENT (TMUX)"
echo "=============================================================================="
echo "Session name: $SESSION_NAME"
echo "Epochs: 1500"
echo "Checkpoint: ./checkpoints/sweeping/poisson_lin_lin_tmux"
echo "=============================================================================="
echo ""

# Check if session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "⚠ Tmux session '$SESSION_NAME' already exists."
    echo "Options:"
    echo "  1. Attach to existing session: tmux attach -t $SESSION_NAME"
    echo "  2. Kill and restart: tmux kill-session -t $SESSION_NAME && $0"
    exit 1
fi

# Create new tmux session and run training
echo "Creating tmux session '$SESSION_NAME'..."
tmux new-session -d -s $SESSION_NAME -c "$SCRIPT_DIR"

tmux send-keys -t $SESSION_NAME "cd $SCRIPT_DIR" C-m
tmux send-keys -t $SESSION_NAME "python train_sweeping.py 2>&1 | tee checkpoints/sweeping/poisson_lin_lin_tmux/training.log" C-m

echo "✓ Training started in tmux session '$SESSION_NAME'"
echo ""
echo "Useful commands:"
echo "  Attach to session:   tmux attach -t $SESSION_NAME"
echo "  Detach from session: Ctrl+b then d"
echo "  Kill session:        tmux kill-session -t $SESSION_NAME"
echo "  View log:            tail -f checkpoints/sweeping/poisson_lin_lin_tmux/training.log"
echo ""
echo "Device monitoring:"
echo "  watch -n 1 nvidia-smi"
