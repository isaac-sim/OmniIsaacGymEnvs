checkpoint_path="/home/nikepupu/Desktop/OmniIsaacGymEnvs/omniisaacgymenvs/runs/FrankaMobileDrawer/nn/FrankaMobileDrawer.pth"

# Trap the SIGINT signal
trap 'echo "Caught SIGINT signal, exiting..."; exit' SIGINT

# Initialize the failure counter
failure_count=0

# Training loop
until {
    # Check if the checkpoint exists at the start of each loop iteration
    if [ -f "$checkpoint_path" ]; then
        echo "Checkpoint found. Using checkpoint for training."
        ~/.local/share/ov/pkg/isaac_sim-2023.1.0-hotfix.1/python.sh scripts/rlgames_train.py task=FrankaMobileDrawer headless=True checkpoint="$checkpoint_path"
    else
        echo "Checkpoint not found. Continuing without checkpoint."
        ~/.local/share/ov/pkg/isaac_sim-2023.1.0-hotfix.1/python.sh scripts/rlgames_train.py task=FrankaMobileDrawer headless=True
    fi
}; do
    echo "Training attempt failed."

    # Increment the failure counter
    ((failure_count++))

    # Check if the failure count has reached the limit
    if [ $failure_count -ge 20 ]; then
        echo "Failed 20 times. Stopping the training loop."
        exit 1
    fi

    echo "Restarting the training loop... (Attempt: $failure_count)"
    # Additional logic can be added here if necessary
done

