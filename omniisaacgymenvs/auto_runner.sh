#!/bin/bash

export PYTHON_PATH=~/.local/share/ov/pkg/isaac_sim-2023.1.0/python.sh

$PYTHON_PATH  scripts/rlgames_train.py task=Biped experiment=Biped_final_0 max_iterations=2000 headless=True task.env.deathCost=-1.0 task.env.terminationHeight=0.0 task.env.terminationHeading=0.0 task.env.terminationUp=0.0 task.env.progress_reward=1.0 train.params.config.learning_rate=5e-4
$PYTHON_PATH  scripts/rlgames_train.py task=Biped experiment=Biped_final_1 max_iterations=2000 headless=True task.env.deathCost=-1.0 task.env.terminationHeight=0.0 task.env.terminationHeading=0.0 task.env.terminationUp=0.0 task.env.progress_reward=1.0 train.params.config.learning_rate=5e-5
$PYTHON_PATH  scripts/rlgames_train.py task=Biped experiment=Biped_final_2 max_iterations=2000 headless=True task.env.deathCost=-5.0 task.env.terminationHeight=0.0 task.env.terminationHeading=0.0 task.env.terminationUp=0.0 task.env.progress_reward=1.0 train.params.config.learning_rate=5e-5
$PYTHON_PATH  scripts/rlgames_train.py task=Biped experiment=Biped_final_3 max_iterations=2000 headless=True task.env.deathCost=-5.0 task.env.terminationHeight=0.7 task.env.terminationHeading=0.0 task.env.terminationUp=0.0 task.env.progress_reward=1.0 train.params.config.learning_rate=5e-5
$PYTHON_PATH  scripts/rlgames_train.py task=Biped experiment=Biped_final_4 max_iterations=2000 headless=True task.env.deathCost=-5.0 task.env.terminationHeight=0.7 task.env.terminationHeading=0.0 task.env.terminationUp=0.0 task.env.progress_reward=5.0 train.params.config.learning_rate=5e-5
$PYTHON_PATH  scripts/rlgames_train.py task=Biped experiment=Biped_final_5 max_iterations=2000 headless=True task.env.deathCost=-5.0 task.env.terminationHeight=0.7 task.env.terminationHeading=0.0 task.env.terminationUp=0.7 task.env.progress_reward=5.0 train.params.config.learning_rate=5e-5
$PYTHON_PATH  scripts/rlgames_train.py task=Biped experiment=Biped_final_6 max_iterations=2000 headless=True task.env.deathCost=-5.0 task.env.terminationHeight=0.7 task.env.terminationHeading=0.5 task.env.terminationUp=0.7 task.env.progress_reward=5.0 train.params.config.learning_rate=5e-5
$PYTHON_PATH  scripts/rlgames_train.py task=Biped experiment=Biped_final_7 max_iterations=2000 headless=True task.env.deathCost=-5.0 task.env.terminationHeight=0.7 task.env.terminationHeading=0.5 task.env.terminationUp=0.7 task.env.progress_reward=5.0 train.params.config.learning_rate=5e-5 task.env.actionsCost=0.03 task.env.energyCost=0.1

# run without resets
# run with lower lr
# run with higher death cost
# run with height reset
# increase progress reward
# run with up reset
# run with heading reset
# run with higher action and energy cost

# $PYTHON_PATH scripts/rlgames_train.py task=Biped test=True checkpoint=runs/Biped_15_4/nn/Biped_15_4.pth num_envs=2