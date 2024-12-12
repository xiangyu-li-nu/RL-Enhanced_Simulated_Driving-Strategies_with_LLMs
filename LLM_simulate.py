import torch
from transformers import pipeline
from prompt import *

# Load the Llama 3.1 pre-trained model
model_id = r"D:\LLM_for_AV\Llama-3.2-1B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Simulate MAV and RAV history state data
def simulate_history_state():
    # Simulate historical state data for MAV and RAV (positions, lanes, speed, acceleration in the past)
    mav_history = [
        {'x_m_t': 100, 'lane_m_t': 1, 'v_m_t': 30, 'a_m_t': 2, 'decision_m_t': 'cooperate', 'action_m_t': 'decelerate'},
        # MAV past state 1
        {'x_m_t': 105, 'lane_m_t': 1, 'v_m_t': 32, 'a_m_t': 2.1, 'decision_m_t': 'compete', 'action_m_t': 'accelerate'},
        # MAV past state 2
        {'x_m_t': 110, 'lane_m_t': 1, 'v_m_t': 33, 'a_m_t': 1.9, 'decision_m_t': 'cooperate',
         'action_m_t': 'Lane Change'}  # MAV past state 3
    ]

    # RAV's historical state, including position, speed, acceleration, and decision (cooperate or compete)
    rav_history = [
        {'x_r_t': 110, 'lane_r_t': 1, 'v_r_t': 28, 'a_r_t': 1.5, 'decision_r_t': 'cooperate', 'action_r_t': 'Decelerate'},
        # RAV past state 1
        {'x_r_t': 112, 'lane_r_t': 1, 'v_r_t': 30, 'a_r_t': 1.6, 'decision_r_t': 'compete', 'action_r_t': 'accelerate'},
        # RAV past state 2
        {'x_r_t': 115, 'lane_r_t': 1, 'v_r_t': 31, 'a_r_t': 1.7, 'decision_r_t': 'compete', 'action_r_t': 'Maintain Speed'}
        # RAV past state 3
    ]

    return mav_history, rav_history


# Simulate MAV and RAV current state
def simulate_current_state():
    # Simulate current state (real-time position, speed, acceleration, etc.)
    mav_state = {
        'x_m_t': 120,  # MAV current position
        'lane_m_t': 1,  # MAV current lane
        'v_m_t': 35,  # MAV current speed
        'a_m_t': 2.5,  # MAV current acceleration
        'P_cooperate_m': 0.8  # MAV predicted probability of RAV cooperating
    }

    rav_state = {
        'x_r_t': 125,  # RAV current position
        'lane_r_t': 1,  # RAV current lane
        'v_r_t': 33,  # RAV current speed
        'a_r_t': 1.8,  # RAV current acceleration
        'decision_r_t': 'cooperate',  # RAV current decision (cooperate)
        'action_r_t': 'Lane Change'  # RAV current action (changing lane)
    }

    return mav_state, rav_state


# Construct MAV system input prompts

# Inference function
def infer_mav_decision(mav_history, rav_history, mav_state, rav_state):
    # Generate MAV input prompt
    messages = generate_mav_prompt(mav_history, rav_history, mav_state, rav_state)

    # Call the model for inference
    outputs = pipe(messages, max_new_tokens=512)

    decision_explanation = outputs[0]["generated_text"][-1]["content"]

    decision, action = extract_wrapped_content(decision_explanation)
    # Return model output
    return {"decision_explanation": decision_explanation, "decision": decision, "action": action}

def infer_rav_decision(mav_history, rav_history, mav_state, rav_state):
    # Generate MAV input prompt
    messages = generate_mav_prompt(mav_history, rav_history, mav_state, rav_state)

    # Call the model for inference
    outputs = pipe(messages, max_new_tokens=512)

    decision_explanation = outputs[0]["generated_text"][-1]["content"]

    decision, action = extract_wrapped_content(decision_explanation)
    # Return model output
    return {"decision_explanation": decision_explanation, "decision": decision, "action": action}

# Main function
if __name__ == "__main__":
    # Simulate MAV and RAV history state data
    mav_history, rav_history = simulate_history_state()

    # Simulate MAV and RAV current state
    mav_state, rav_state = simulate_current_state()

    # Call inference function
    decision = infer_mav_decision(mav_history, rav_history, mav_state, rav_state)

    # Print model-generated decision and suggested actions
    print("MAV's Decision and Suggested Action: ")
    print(decision)
