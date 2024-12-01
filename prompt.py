import re

def generate_mav_prompt(mav_history, rav_history, mav_state, rav_state,ttc = 0.5):
    # Build system tips
    system_message = """
    You are driving an autonomous vehicle (MAV) and must decide how to react to a Remote Autonomous Vehicle (RAV) attempting to merge into your lane from an entrance ramp.
    Your goal is to either cooperate by decelerating or changing lanes, or to compete by maintaining your position or accelerating to block the RAV.
    The decision you make will affect the safety and efficiency of the traffic flow.
    """

    # Historical status data for MAV and RAV
    mav_history_str = "\n".join([
        f"MAV history - Position: {mav_state['x_m_t']}, Lane: {mav_state['lane_m_t']}, Speed: {mav_state['v_m_t']}, Acceleration: {mav_state['a_m_t']}\n" +
        f"RAV history - Position: {rav_state['x_r_t']}, Lane: {rav_state['lane_r_t']}, Speed: {rav_state['v_r_t']}, Acceleration: {rav_state['a_r_t']}, Decision: {rav_state['decision_r_t']}, Action: {rav_state['action_r_t']}"
        for mav_state, rav_state in zip(mav_history, rav_history)
    ])
    # Current status description
    current_state_message = f"""### **Input State Information (Step 1):**
**Historical State:**
{mav_history_str}

---

### **State Variables (Step 2):**
    MAV current state:
    - Position: {mav_state['x_m_t']}, Lane: {mav_state['lane_m_t']}, Speed: {mav_state['v_m_t']}, Acceleration: {mav_state['a_m_t']}
    
    RAV current state:
    - Position: {rav_state['x_r_t']}, Lane: {rav_state['lane_r_t']}, Speed: {rav_state['v_r_t']}, Acceleration: {rav_state['a_r_t']}
    
    Time-to-Collision (TTC) metric: {ttc}
---

### **RAV Behavior Prediction (Step 3):**
Predict RAV’s cooperation probability (P_cooperate,m) based on its historical states:
- P_cooperate,m =
    - 0 if MAV predicts RAV will choose to compete (based on RAV’s historical states).
    - 1 if MAV predicts RAV will choose to cooperate (based on RAV’s historical states).

---

### **Action Output (Step 4):**
Based on RAV’s historical state and current state, choose your plan:
- **Action:** Choose **Cooperate** or **Compete**, and explain your decision. Provide recommended actions:

    - **If you choose to cooperate:**
        - **Decelerate:** Slow down to allow the RAV to merge or avoid a collision.
        - **Lane Change:** Change lanes to allow the RAV to merge safely.

    - **If you choose to compete:**
        - **Accelerate:** Increase speed to maintain your position or pass the RAV.
        - **Maintain Speed:** Maintain your speed and position to block the RAV from merging smoothly.

---

### **Decision Explanation:**
Provide a detailed explanation of your decision based on the input state variables and predicted behavior. Your explanation should address:
-Why you chose --Cooperation-- or --Competition--.
-The rationale for your recommended action.
- For the decision, use -- to wrap the choice (e.g., --Cooperation--, --Competition--). For the recommended action, use %% to wrap the action. If the decision is --Cooperation--, the actions may include %%Decelerate%% or %%Lane Change%%. If the decision is --Competition--, the actions may include %%Maintain Speed%% or %%Accelerate%%.

    """

    # Combining into model input
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": current_state_message},
    ]

    return messages


def generate_rav_prompt(mav_history, rav_history, mav_state, rav_state, ttc=0.5):
    # Construct system message
    system_message = """
    You are driving a Remote Autonomous Vehicle (RAV) and must decide how to react to a Main Autonomous Vehicle (MAV) already driving on the main road.
    Your goal is to cooperate by decelerating, or to compete by maintaining your position or accelerating to block the MAV.
    The decision you make will impact the safety and efficiency of the traffic flow.
    """

    # RAV and MAV historical state data
    rav_history_str = "\n".join([
        f"RAV history - Position: {rav_state['x_r_t']}, Lane: {rav_state['lane_r_t']}, Speed: {rav_state['v_r_t']}, Acceleration: {rav_state['a_r_t']}, Decision: {mav_state['decision_r_t']}, Action: {mav_state['action_r_t']}\n" +
        f"MAV history - Position: {mav_state['x_m_t']}, Lane: {mav_state['lane_m_t']}, Speed: {mav_state['v_m_t']}, Acceleration: {mav_state['a_m_t']}"
        for rav_state, mav_state in zip(rav_history, mav_history)
    ])

    # Current state description
    current_state_message = f"""### **Input State Information (Step 1):**
**Historical State:**
{rav_history_str}

---

### **State Variables (Step 2):**
    RAV current state:
    - Position: {rav_state['x_r_t']}, Lane: {rav_state['lane_r_t']}, Speed: {rav_state['v_r_t']}, Acceleration: {rav_state['a_r_t']}

    MAV current state:
    - Position: {mav_state['x_m_t']}, Lane: {mav_state['lane_m_t']}, Speed: {mav_state['v_m_t']}, Acceleration: {mav_state['a_m_t']}

    Time-to-Collision (TTC) metric: {ttc}
---

### **MAV Behavior Prediction (Step 3):**
Predict MAV’s cooperation probability (P_cooperate,r) based on its historical states:
- P_cooperate,r =
    - 0 if RAV predicts MAV will choose to compete (based on MAV’s historical states).
    - 1 if RAV predicts MAV will choose to cooperate (based on MAV’s historical states).

---

### **Action Output (Step 4):**
Based on MAV’s historical state and current state, choose your plan:
- **Action:** Choose **Cooperate** or **Compete**, and explain your decision. Provide recommended actions:

    - **If you choose to cooperate:**
        - **Decelerate:** Slow down to allow the RAV to merge or avoid a collision.

    - **If you choose to compete:**
        - **Accelerate:** Increase speed to maintain your position or pass the RAV.
        - **Maintain Speed:** Maintain your speed and position to block the RAV from merging smoothly.

---

### **Decision Explanation:**
Provide a detailed explanation of your decision based on the input state variables and predicted behavior. Your explanation should address:
- Why you chose --Cooperation-- or --Competition--.
- The rationale for your recommended action.
- For the decision, use -- to wrap the choice (e.g., --Cooperation--, --Competition--). For the recommended action, use %% to wrap the action. If the decision is --Cooperation--, the actions only include %%Decelerate%%. If the decision is --Competition--, the actions may include %%Maintain Speed%% or %%Accelerate%%.
"""

    # Combining into model input
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": current_state_message},
    ]

    return messages


def generate_extract_mav_key_prompt(llm_output):
    # Build system tips
    system_message = """
    Given the input data, extract the decision and the recommended actions as follows:

    1. **Decision Extraction**: Identify whether the decision is related to cooperation or competition. 
       - If the decision is "cooperate," output "--Cooperation--".
       - If the decision is "compete," output "--Compete--".

    2. **Action Extraction**: Based on the decision, identify the corresponding actions.
       - For "cooperate" decisions, the actions could be: "Accelerate", "Decelerate", "Lane Change".
         - If the decision is to cooperate, wrap the action in "%%" (e.g., %%Accelerate%%, %%Decelerate%%, %%Lane Change%%).
       - For "compete" decisions, the actions could be: "No Change", "Accelerate".
         - If the decision is to compete, wrap the action in "%%" (e.g., %%Maintain Speed%%, %%Accelerate%%).
    """

    # Construct the current status message (mock content)
    current_state_message = f"""{llm_output}"""

    # Combining into model input
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": current_state_message},
    ]

    return messages


def extract_wrapped_content(text):
    """
    Extract the first content wrapped by `--` and the content wrapped by `%%`, and remove special symbols (such as `*`).
    Make sure the first element is the content wrapped by `--`, and the second element is the content wrapped by `%%`.
    """
    # Extract the content wrapped by `--` for the first time
    first_match = re.search(r'--([^\*]+)--', text)
    decision = first_match.group(1) if first_match else None

    # Extract the content wrapped by `%%` for the first time
    second_match = re.search(r'%%([^\*]+)%%', text)
    action = second_match.group(1) if second_match else None

    # Returns the extracted results
    return decision, action