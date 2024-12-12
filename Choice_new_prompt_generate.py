import random
import numpy as np
from prompt_generate import *
from openai import OpenAI
import re
import torch
from transformers import pipeline

def calculate_time_to_zero(x, v, a):
    """
    Calculate the time for a vehicle to reach the origin (assuming uniform acceleration).
    Solve the equation: 0 = x + v * delta_t + 0.5 * a * delta_t^2
    Return the time delta_t to reach the origin (if a solution exists), otherwise return None.
    """
    if a == 0:
        # If acceleration is 0, it becomes a linear equation v * delta_t = -x
        if v == 0:
            return 10000  # No solution (vehicle cannot move)
        return x / v  # Linear solution

    # Solve the quadratic equation 0 = x + v * delta_t + 0.5 * a * delta_t^2
    # Corresponding equation is 0 = 0.5 * a * delta_t^2 + v * delta_t + x
    A = 0.5 * a
    B = v
    C = x

    # Calculate the discriminant
    discriminant = B**2 + 4 * A * C
    if discriminant < 0:
        return None  # No solution (i.e., the vehicle cannot reach the origin)

    # Solve for the two roots, choose the positive one
    delta_t1 = (-B + np.sqrt(discriminant)) / (2 * A)
    delta_t2 = (-B - np.sqrt(discriminant)) / (2 * A)

    # Return the positive time value
    valid_times = [t for t in [delta_t1, delta_t2] if t > 0]
    if valid_times:
        return min(valid_times)  # Return the smallest positive time

    return 10000  # If no valid solution, return None

def extract_wrapped_content(text): 
    """
    Extract the content wrapped by `--` and `%%`, and remove special characters (such as `*`).
    Ensure that the first element is the content wrapped by `--`, and the second element is the content wrapped by `%%`.
    """
    # Extract the content wrapped by `--`
    first_match = re.search(r'--([^\*]+)--', text)
    decision = first_match.group(1) if first_match else "None"

    # Extract the content wrapped by `%%`
    second_match = re.search(r'%%([^\*]+)%%', text)
    action = second_match.group(1) if second_match else "None"

    # Return the extracted result
    return {"decision": decision, "action": action}

# OpenAI Decision Generation
def generate_decision_with_openai(mav_prompt, rav_prompt):
    client = OpenAI(
        base_url='https://xiaoai.plus/v1',
        api_key=''  # Replace with your actual API key
    )

    # Generate MAV decision
    mav_completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": mav_prompt[0]['content']},
                  {"role": "user", "content": mav_prompt[1]['content']}]
    )
    mav_decision = mav_completion.choices[0].message.content
    mav_decision_action = extract_wrapped_content(mav_decision)
    mav_decision_action["question"] = mav_prompt[0]['content'] + "\n" + mav_prompt[1]['content']
    mav_decision_action["answer"] = mav_decision

    # Generate RAV decision
    rav_completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": rav_prompt[0]['content']},
                  {"role": "user", "content": rav_prompt[1]['content']}]
    )
    rav_decision = rav_completion.choices[0].message.content
    rav_decision_action = extract_wrapped_content(rav_decision)
    rav_decision_action["question"] = rav_prompt[0]['content'] + "\n" + rav_prompt[1]['content']
    rav_decision_action["answer"] = rav_decision

    return mav_decision_action, rav_decision_action

from transformers import AutoModelForCausalLM, AutoTokenizer 
from peft import PeftModel, PeftConfig

# Llama Decision Generation

def generate_decision_with_llama(mav_prompt, rav_prompt):
    # model_id = "D:/pretrained_models/llama3.1-7b"
    # pipe = pipeline(
    #     "text-generation",
    #     model=model_id,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    # )

    # Load Llama 3.1 Pre-trained Model
    # model_id = "D:/pretrained_models/llama3.1-7b"
    #
    # model = AutoModelForCausalLM.from_pretrained(model_id)
    # tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load PEFT configuration and the trained model
    peft_model_id = "peft_model"  # Replace with the path to your trained model
    config = PeftConfig.from_pretrained(peft_model_id)

    # Load the base model (e.g., T5, BART, or other Seq2Seq models)
    model_name = "D:/pretrained_models/Llama-3.2-1B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Load the PEFT model (i.e., the trained LoRA model)
    model = PeftModel.from_pretrained(model, peft_model_id)

    # Load the Tokenizer (ensure compatibility with the model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad_token for the tokenizer if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as the padding token

    # Create the pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Generate MAV decision
    # mav_prompt_text = mav_prompt[1]['content']  # Assuming MAV decision is based on prompt[1] content
    outputs = pipe(mav_prompt, max_new_tokens=4096)
    mav_decision = outputs[0]["generated_text"][2]["content"]
    mav_decision_action = extract_wrapped_content(mav_decision)

    # Generate RAV decision
    # rav_prompt_text = rav_prompt[1]['content']
    outputs = pipe(rav_prompt, max_new_tokens=4096)
    rav_decision = outputs[0]["generated_text"][2]["content"]
    rav_decision_action = extract_wrapped_content(rav_decision)

    # Add the MAV and RAV questions and answers
    mav_decision_action["question"] = mav_prompt[0]['content'] + "\n" + mav_prompt[1]['content']
    mav_decision_action["answer"] = mav_decision

    rav_decision_action["question"] = rav_prompt[0]['content'] + "\n" + rav_prompt[1]['content']
    rav_decision_action["answer"] = rav_decision

    return mav_decision_action, rav_decision_action
    


# Main function to generate decisions based on the selected model
def get_mav_and_rav_decision(mav_history, rav_history, action_options, mav_state, rav_state, model_choice="openai"):

    # Generate prompts for MAV and RAV
    mav_prompt = generate_mav_prompt(mav_history, rav_history, action_options, mav_state, rav_state)
    rav_prompt = generate_rav_prompt(mav_history, rav_history, action_options, mav_state, rav_state)

    # Based on the selected model, generate decisions
    if model_choice == "openai":
        return generate_decision_with_openai(mav_prompt, rav_prompt)
    elif model_choice == "llama":
        return generate_decision_with_llama(mav_prompt, rav_prompt)

# Simulate historical states and vehicle movements
def simulate_history_state(m, t, max_initial_speed=35, max_initial_accel=2, allow_negative_distance=True,
                           same_initial_conditions=False):

    """
    Randomly generate 'm' pairs of historical states, update acceleration and position based on vehicle actions,
    with the goal being 0. The last pair represents the current state.
    """

    # Define action options
    action_options = {
        'MAV': {
            'decisions': ['cooperate', 'compete'],
            'actions': {
                'cooperate': ['decelerate', 'Lane Change', 'Maintain Speed'],
                'compete': ['accelerate', 'Maintain Speed']
            }
        },
        'RAV': {
            'decisions': ['cooperate', 'compete'],
            'actions': {
                'cooperate': ['decelerate', 'Maintain Speed'],
                'compete': ['accelerate', 'Maintain Speed']
            }
        }
    }

    # Initialize environment
    mav_x = random.randint(90, 120)  # Randomly initialize MAV's position
    mav_v = random.randint(20, max_initial_speed)  # Randomly initialize MAV's speed
    mav_a = random.randint(0, max_initial_accel)  # Randomly initialize MAV's acceleration
    mav_lane = 2  # Fixed lane for MAV

    rav_x = random.randint(100, 130)  # Randomly initialize RAV's position
    rav_a = random.randint(0, max_initial_accel)  # Randomly initialize RAV's acceleration

    # Calculate RAV's speed to ensure it meets MAV at x=0
    mav_achieve_time = calculate_time_to_zero(mav_x, mav_v, mav_a)
    rav_avg_v = rav_x / mav_achieve_time
    rav_v = rav_avg_v - rav_a * mav_achieve_time / 2
    rav_lane = 3  # Fixed lane for RAV

    mav_history = []
    rav_history = []
    rewards = []
    reward_details = []

    all_info = []
    for i in range(m):
        # Save the current historical state
        mav_history.append({
            'x_m_t': mav_x,
            'v_m_t': mav_v,
            'a_m_t': mav_a,
            'lane_m_t': mav_lane
        })

        rav_history.append({
            'x_r_t': rav_x,
            'v_r_t': rav_v,
            'a_r_t': rav_a,
            'lane_r_t': rav_lane
        })

        # Randomly select MAV and RAV decisions and actions
        mav_decision_action, rav_decision_action = get_mav_and_rav_decision(mav_history[:-1], rav_history[:-1], action_options,
                                                                            mav_history[-1], rav_history[-1])

        mav_decision = mav_decision_action["decision"].strip().lower()
        mav_action = mav_decision_action["action"].strip().lower()

        if mav_action == None:
            mav_action = "accelerate"
            print("MAV has no action, adding default action: accelerate")

        rav_decision = rav_decision_action["decision"].strip().lower()
        rav_action = rav_decision_action["action"].strip().lower()

        if rav_action == None:
            rav_action = "accelerate"
            print("RAV has no action, adding default action: accelerate")

        # Calculate time delay before and after the initial acceleration
        time_before_mav = calculate_time_to_zero(mav_x, mav_v, mav_a)
        time_before_rav = calculate_time_to_zero(rav_x, rav_v, rav_a)

        # Update acceleration based on selected actions
        if mav_action == 'accelerate':
            mav_a += 2 * t  # Increase acceleration
        elif mav_action == 'decelerate':
            mav_a -= 2 * t  # Decrease acceleration

        if rav_action == 'accelerate':
            rav_a += 2 * t  # Increase acceleration
        elif rav_action == 'decelerate':
            rav_a -= 2 * t  # Decrease acceleration

        # Calculate the time delay after updating acceleration
        time_after_mav = calculate_time_to_zero(mav_x, mav_v, mav_a)
        time_after_rav = calculate_time_to_zero(rav_x, rav_v, rav_a)

        # Calculate time delays
        time_delay_mav = time_before_mav - time_after_mav
        time_delay_rav = time_before_rav - time_after_rav

        # Determine if MAV wins (reaches the goal first)
        mav_achieve_goal = True if time_after_mav > time_after_rav else False  # MAV reaches goal first?
        mav_win_score = 10 if mav_achieve_goal else -10  # MAV win score
        RAV_win_score = -10 if mav_achieve_goal else 10  # RAV win score

        # Calculate collision penalty based on time difference
        time_difference = abs(time_after_mav - time_after_rav) if time_after_mav != float('inf') and time_after_rav != float('inf') else float('inf')
        collision_penalty = -5 / (time_difference + 1e-5)  # Smaller time difference, higher penalty

        # Calculate individual rewards for MAV and RAV
        mav_reward, mav_reward_detail = mav_reward_function(mav_action, time_delay_mav, mav_achieve_goal, collision_penalty, mav_win_score)
        rav_reward, rav_reward_detail = rav_reward_function(rav_action, time_delay_rav, collision_penalty, RAV_win_score)

        # Save all information
        all_info.append({
            "question": mav_decision_action["question"],
            "answer": mav_decision_action["answer"],
            # Add other information as needed
        })
        

# Reward function for MAV
def mav_reward_function(mav_action, time_delay_mav, mav_achieve_goal, collision_penalty, win_score):

    """
    Calculate the reward function for MAV.
    """

    switch_lane_penalty = -5  # Penalty for lane change
    achieve_goal_reward = 10  # Reward for reaching the goal
    time_reward = 2  # Time reward coefficient

    mav_reward = 0
    mav_reward_detail = {
        'switch_lane_penalty': 0,
        'achieve_goal_reward': 0,
        'collision_penalty': 0,
        'time_reward': 0
    }

    if mav_action == 'Lane Change':
        mav_reward += switch_lane_penalty
        mav_reward_detail['switch_lane_penalty'] = switch_lane_penalty

    if mav_achieve_goal:
        mav_reward += achieve_goal_reward
        mav_reward_detail['achieve_goal_reward'] = achieve_goal_reward

    # Time reward
    mav_time_reward = time_delay_mav * time_reward
    mav_reward += mav_time_reward
    mav_reward_detail['time_reward'] = mav_time_reward

    # Collision penalty
    mav_reward += collision_penalty
    mav_reward_detail['collision_penalty'] = collision_penalty

    # Win score
    mav_reward += win_score
    mav_reward_detail['win_score'] = win_score

    return mav_reward, mav_reward_detail

# Reward function for RAV
def rav_reward_function(rav_action, time_delay_rav, collision_penalty, win_score):

    """
    Calculate the reward function for RAV.
    """

    switch_lane_penalty = -5  # Penalty for lane change
    time_reward = 2  # Time reward coefficient

    rav_reward = 0
    rav_reward_detail = {
        'switch_lane_penalty': 0,
        'collision_penalty': 0,
        'time_reward': 0
    }

    if rav_action == 'Lane Change':
        rav_reward += switch_lane_penalty
        rav_reward_detail['switch_lane_penalty'] = switch_lane_penalty

    # Time reward
    rav_time_reward = time_delay_rav * time_reward
    rav_reward += rav_time_reward
    rav_reward_detail['time_reward'] = rav_time_reward

    # Collision penalty
    rav_reward += collision_penalty
    rav_reward_detail['collision_penalty'] = collision_penalty

    # Win score
    rav_reward += win_score
    rav_reward_detail['win_score'] = win_score

    return rav_reward, rav_reward_detail

# Test function
import json

def test_simulate_history_state():

    m = 1
    t = 1.0
    max_initial_speed = 35
    max_initial_accel = 2
    allow_negative_distance = True
    same_initial_conditions = False
    data = []

    for i in range(1):
        all_info = simulate_history_state(
            m, t, max_initial_speed, max_initial_accel, allow_negative_distance, same_initial_conditions
        )
        data = data + all_info
        for info in all_info:
            print(info)

    try:
        with open('train_new_test.json', 'r') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        # If file does not exist, initialize as empty list
        existing_data = []

    # Append new questions and answers to existing data
    existing_data.extend(data)

    # Write back to the file in append mode
    with open('train_new_test.json', 'w') as f:
        json.dump(existing_data, f, indent=4)

    print(f"Results for {data} generations appended to train.json")

# Run test function
if __name__ == "__main__":
    test_simulate_history_state()
    