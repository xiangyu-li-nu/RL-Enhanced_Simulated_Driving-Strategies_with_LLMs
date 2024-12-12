# RL-Enhanced_Simulated_Driving-Strategies_with_LLMs
## Overview

This repository provides an implementation of a decision-making framework for Main Autonomous Vehicle (MAV) and Remote Autonomous Vehicle (RAV) tasks, optimized for LLaMA models. The framework supports the following models:

- LLaMA 3.2-1b-Instruct
- LLaMA 3.2-3b-Instruct
- LLaMA 3.1-8b-Instruct

The project integrates imitation learning, self-play reinforcement learning, and supervised fine-tuning to enable MAV and RAV to make cooperative or competitive decisions based on their states. The codebase is designed for scalability and adaptability, allowing it to be applied to broader autonomous decision-making problems.

## File Descriptions
1. Choice_new_prompt_generate.py
   - Generates decision-making prompts for MAV and RAV.
   - Supports OpenAI GPT and LLaMA models for action prediction.
   - Includes utilities for processing historical states and generating actions.

2. Chose_data.py
   - Filters high-reward data from the training dataset.
   - Saves filtered data to a new JSON file for supervised fine-tuning.

3. Environment_testing.py
   - Simulates the interaction environment for MAV and RAV.
   - Defines state transitions, collision detection, and reward functions.

4. Evaluate_LLM.py
   - Evaluates LLM performance on MAV and RAV decision-making tasks.
   - Prepares test prompts and generates decisions using OpenAI GPT or LLaMA.

5. LLama_train_new.py
   - Fine-tunes the LLaMA model using LoRA for MAV and RAV tasks.
   - Includes data tokenization, model training, and evaluation loops.

6. LLM_GAME.py
   - Implements reinforcement learning using the PPO algorithm.
   - Simulates MAV and RAV gameplay in competitive scenarios.

7. LLM_simulate.py
   - Simulates historical states and current conditions for MAV and RAV.
   - Calls LLMs to predict actions based on simulated states.

8. Openai_generate.py
   - Generates training datasets using GPT models via OpenAI API.
   - Includes tools for constructing and saving large-scale question-answer pairs.

9. Prompt_generate.py
   - Creates input prompts for MAV and RAV decisions.
   - Incorporates detailed scenario descriptions and action options.

10. Use_lora_new_model.py
    - Loads and uses fine-tuned LLaMA models with LoRA for inference.
    - Provides utilities for generating predictions from trained models.

## Authors

Xiangyu Li
- Second-year Ph.D. student in Transportation Engineering at Northwestern University
- Email: xiangyuli2027@u.northwestern.edu
