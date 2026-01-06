#!/bin/bash

# Shell script to run sweeping experiment with tmux
# This replicates the exact training from the notebook on GPU 1

SESSION_NAME="sweeping_tmux"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=================================="
echo "Starting Sweeping Experiment (tmux)"
echo "=================================="
echo "Session: $SESSION_NAME"
echo "GPU: 1"
echo "Script: train_sweeping_tmux.py"
echo "Checkpoint: ./checkpoints/sweeping_tmux"
echo "=================================="

# Check if tmux session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "⚠ Tmux session '$SESSION_NAME' already exists."
    echo "Options:"
    echo "  1. Attach to existing session: tmux attach -t $SESSION_NAME"
    echo "  2. Kill existing session: tmux kill-session -t $SESSION_NAME"
    echo "  3. Use a different session name"
    exit 1
fi

# Create new tmux session and run training
echo "Creating new tmux session..."
tmux new-session -d -s $SESSION_NAME -c $SCRIPT_DIR

# Set environment variable for GPU
tmux send-keys -t $SESSION_NAME "export CUDA_VISIBLE_DEVICES=1" C-m

# Run the training script
tmux send-keys -t $SESSION_NAME "python train_sweeping_tmux.py" C-m

echo "✓ Tmux session created and training started!"
echo ""
echo "To monitor training:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "To detach from session (while inside):"
echo "  Ctrl+b, then d"
echo ""
echo "To kill session:"
echo "  tmux kill-session -t $SESSION_NAME"
echo ""
echo "To check if session is running:"
echo "  tmux ls"
