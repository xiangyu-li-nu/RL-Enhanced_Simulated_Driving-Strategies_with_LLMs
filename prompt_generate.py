import random

def simulate_history_state(m, t, max_initial_speed=35, max_initial_accel=2, allow_negative_distance=True,
                           same_initial_conditions=False):
    """
    Randomly generate m pairs of historical states, and update their acceleration and position based on vehicle actions, 
    with the endpoint at 0. The last pair is the current state.
    """
    m = random.randint(max_initial_accel, m)
    # Define a dictionary to store MAV and RAV decision and action options
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

    mav_history = []
    rav_history = []

    # If same_initial_conditions is True, randomly generate a common initial state
    if same_initial_conditions:
        initial_x = random.randint(90, 120)
        initial_v = random.randint(10, max_initial_speed)
        initial_a = random.randint(-max_initial_accel, max_initial_accel)  # Initial acceleration as an integer
        initial_lane = random.randint(1, 2)
        initial_decision = random.choice(action_options['MAV']['decisions'])
        initial_action = random.choice(action_options['MAV']['actions'][initial_decision])

        # Use the same initial conditions for both MAV and RAV
        mav_x = rav_x = initial_x
        mav_v = rav_v = initial_v
        mav_a = rav_a = initial_a
        mav_lane = rav_lane = initial_lane
        mav_decision = rav_decision = initial_decision
        mav_action = rav_action = initial_action
    else:
        # Randomly initialize MAV and RAV states
        mav_x = random.randint(90, 120)
        mav_v = random.randint(10, max_initial_speed)
        mav_a = random.randint(-max_initial_accel, max_initial_accel)  # Initial acceleration as an integer
        mav_lane = random.randint(1, 2)
        mav_decision = random.choice(action_options['MAV']['decisions'])
        mav_action = random.choice(action_options['MAV']['actions'][mav_decision])

        rav_x = random.randint(100, 130)
        rav_v = random.randint(10, max_initial_speed)
        rav_a = random.randint(-max_initial_accel, max_initial_accel)  # Initial acceleration as an integer
        rav_lane = random.randint(1, 2)
        rav_decision = random.choice(action_options['RAV']['decisions'])
        rav_action = random.choice(action_options['RAV']['actions'][rav_decision])

    for i in range(m):
        # Record the current historical state
        mav_history.append({
            'x_m_t': mav_x,
            'v_m_t': mav_v,
            'a_m_t': mav_a,
            'lane_m_t': mav_lane,
            'decision_m_t': mav_decision,
            'action_m_t': mav_action
        })

        rav_history.append({
            'x_r_t': rav_x,
            'v_r_t': rav_v,
            'a_r_t': rav_a,
            'lane_r_t': rav_lane,
            'decision_r_t': rav_decision,
            'action_r_t': rav_action
        })

        # Update position and acceleration every t seconds
        # Update MAV
        if allow_negative_distance or mav_x > 0:
            if mav_x - mav_v * t <= 0 and not allow_negative_distance:  # If the next position is negative, stop
                mav_x = 0
                mav_v = 0  # Stop the vehicle
            else:
                mav_x -= mav_v * t + (mav_a * t * t)/2  # Update position based on velocity and time
                if mav_action == 'accelerate':
                    mav_a += 2 * t  # Accelerate, maintain integer acceleration
                elif mav_action == 'decelerate':
                    mav_a -= 2 * t  # Decelerate, maintain integer acceleration
                # 'Lane Change' and 'Maintain Speed' do not change acceleration

        # Update RAV
        if allow_negative_distance or rav_x > 0:
            if rav_x - rav_v * t <= 0 and not allow_negative_distance:  # If the next position is negative, stop
                rav_x = 0
                rav_v = 0  # Stop the vehicle
            else:
                rav_x -= rav_v * t + (rav_v * t * t)/2  # Update position based on velocity and time
                if rav_action == 'accelerate':
                    rav_a += 2 * t  # Accelerate, maintain integer acceleration
                elif rav_action == 'decelerate':
                    rav_a -= 2 * t  # Decelerate, maintain integer acceleration
                # 'Lane Change' and 'Maintain Speed' do not change acceleration

        # If negative position is not allowed, stop before reaching 0
        if not allow_negative_distance:
            if mav_x <= 0:
                return mav_history, rav_history, action_options
            if rav_x <= 0:
                return mav_history, rav_history, action_options

        # To ensure reaching the endpoint (x=0), set position to 0 at the last state
        if i == m - 1:
            mav_x = 0
            rav_x = 0

        # Randomly change decision and action
        mav_decision = random.choice(action_options['MAV']['decisions'])
        mav_action = random.choice(action_options['MAV']['actions'][mav_decision])

        rav_decision = random.choice(action_options['RAV']['decisions'])
        rav_action = random.choice(action_options['RAV']['actions'][rav_decision])

    return mav_history, rav_history, action_options



def generate_rav_prompt(mav_history, rav_history, action_options, mav_state, rav_state):
    """
    Generate a prompt describing how the RAV should choose to cooperate or compete based on MAV's historical state and current state.
    """

    # Construct the system message
    system_message = """
In this scenario, as the driver of a **Remote Autonomous Vehicle (RAV)** attempting to merge into the main road where a **Main Autonomous Vehicle (MAV)** is already driving, you must choose between cooperating or competing for the available lane space. Here's a breakdown of the decision-making process:

### 1. **Cooperation (Decelerating or Maintaining Speed):**
   - **Decelerating:** Reducing your acceleration by 2 units to create more space for the MAV to continue at its current speed is a cooperative strategy. This allows the MAV to maintain its momentum and smooth traffic flow. However, decelerating may result in a loss of time and efficiency for your vehicle.
   - **Maintaining Speed:** Keeping your current speed while waiting for the MAV to pass could be a reasonable approach if you're not in a hurry and there’s enough space for the MAV to continue without issue. This decision ensures that the MAV can pass safely while minimizing unnecessary changes in your acceleration.
   
   **Pros:** Promotes safety by reducing the risk of a collision or conflict. Maintains smoother traffic flow.
   **Cons:** Could reduce your travel speed, leading to a delay in reaching your destination.

### 2. **Competition (Accelerating or Blocking the MAV):**
   - **Accelerating:** Increasing your speed by 2 units will allow you to reach a higher position quickly and potentially block the MAV from continuing. This aggressive strategy might help you maintain your position in the lane but could force the MAV to slow down or adjust its position. This increases the risk of creating a more unsafe and competitive situation.
   - **Maintaining Position:** If you choose to maintain your position and not yield to the MAV, you are essentially competing for space in the same lane. If there’s not enough room, this could lead to a potential conflict where neither vehicle is able to move freely. Maintaining your position could prevent the MAV from continuing, but it risks creating a bottleneck or collision.
   
   **Pros:** Maintains or increases your speed, potentially reducing travel time.
   **Cons:** Risk of collision or tension with the MAV, as it may need to adjust its position or slow down to avoid conflict. This could lead to inefficiency and unsafe traffic behavior.

### Decision Factors:
- **Safety:** Cooperation tends to create a safer scenario by allowing both vehicles to adjust their speeds or trajectories to avoid conflict. If you accelerate or maintain your position, you might block the MAV's progress, leading to a potential unsafe scenario.
- **Efficiency:** Competing by accelerating could help you maintain a higher speed, but this comes at the risk of making traffic flow more chaotic. Cooperation, on the other hand, might decrease your speed, but it supports smoother traffic.
   eg:RAV current state:\n    - Position: 116, Lane: 3, Speed: 24.776699029126213, Acceleration: 0\n\n    MAV current state:\n    - Position: 103, Lane: 2, Speed: 22, Acceleration: 0
   output: To determine the MAV's cooperation probability, we need to analyze the MAV's current state. It is positioned in lane 2 with a speed of 22 and has no acceleration, which suggests a steady driving pattern. Historical state analysis is not provided, but from its current behavior, it does not exhibit signs of aggressive maneuvers like acceleration or lane changing.\n\n2. **Your Decision:**\n   - Based on MA V's steady state and the assumption of its cooperative nature (P_cooperate,r = 1), it's likely that the MAV will maintain or adapt its speed to allow smooth merging. Therefore, a cooperative strategy from the RAV, aiming to ensure safety and a smoother traffic flow, will be appropriate.\n   \n   - **Cooperative Approach:**\n      - Suggested action: Maintain Speed. This will allow the MAV to adjust its behavior once it perceives the RAV's consistent and non-aggressive move.\n   \n   Therefore, maintaining your speed is a reasonable action to, directly aligning with the expected cooperative tendency of the MAV, allowing enough time and space for both vehicles to maneuver safely.\n\n### **Action Output:**\n\n- **Decision:** --Cooperation--\n- **Action:** %%Maintain Speed%%\n\nThe decision to cooperate by maintaining speed minimizes the risk of conflict and contributes to smoother traffic integration without compromising safety.
   """

    # RAV and MAV historical state data
    rav_history_str = "\n".join([ 
        f"RAV history - Position: {rav_state['x_r_t']}, Lane: {rav_state['lane_r_t']}, Speed: {rav_state['v_r_t']}, Acceleration: {rav_state['a_r_t']}\n" +
        f"MAV history - Position: {mav_state['x_m_t']}, Lane: {mav_state['lane_m_t']}, Speed: {mav_state['v_m_t']}, Acceleration: {mav_state['a_m_t']}"
        for rav_state, mav_state in zip(rav_history[:-1], mav_history[:-1])  # Exclude the last element as the current state
    ])

    # Modify the recommended actions based on action_options
    cooperate_actions = action_options["RAV"]["actions"]['cooperate']
    compete_actions = action_options["RAV"]["actions"]['compete']

    cooperate_message = f"Choose an action from the available cooperation actions: {', '.join(cooperate_actions)}"
    compete_message = f"Choose an action from the available competition actions: {', '.join(compete_actions)}"

    # Description of the current state
    current_state_message = f"""### **Input State Information (Step 1):**
**Historical State:**
{rav_history_str}

---

### **State Variables (Step 2):**
    RAV current state:
    - Position: {rav_state['x_r_t']}, Lane: {rav_state['lane_r_t']}, Speed: {rav_state['v_r_t']}, Acceleration: {rav_state['a_r_t']}

    MAV current state:
    - Position: {mav_state['x_m_t']}, Lane: {mav_state['lane_m_t']}, Speed: {mav_state['v_m_t']}, Acceleration: {mav_state['a_m_t']}

### **MAV's Behavior Prediction (Step 3):**
- Predict MAV's cooperation probability (P_cooperate,r) based on its historical states:
    - P_cooperate,r = 0 if MAV is likely to choose to compete.
    - P_cooperate,r = 1 if MAV is likely to choose to cooperate.

### **Decision Explanation (Step 4):**
1. **MAV's Decision Prediction:**
    - Based on MAV's historical behavior, predict if MAV will cooperate or compete. If MAV tends to cooperate, the probability of cooperation is high (P_cooperate = 1). If MAV tends to compete, the probability is low (P_cooperate = 0).

2. **Your Decision:**
    - Choose whether to cooperate or compete based on MAV's likely behavior. Explain why you made the choice.

    - **If you decide to cooperate:**
        - Suggested action: {cooperate_actions[0]} (Choose one action from the cooperation options: {', '.join(cooperate_actions)}).

    - **If you decide to compete:**
        - Suggested action: {compete_actions[0]} (Choose one action from the competition options: {', '.join(compete_actions)}).

### **Action Output (Step 5):**
Based on MAV’s  Behavior Prediction, historical state and current state, choose your plan:
- **Action:** Choose **Cooperate** or **Compete**, and explain your decision. Provide recommended actions:

    - **If you choose to cooperate:**
        {cooperate_message}

    - **If you choose to compete:**
        {compete_message}
    
    Decision (For the decision, use -- to wrap the choice (e.g., --Cooperation--, --Competition--).): --?--
    Action (For the recommended action, use %% to wrap the action): %%?%%

    """

    # Combine into model input
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": current_state_message},
    ]

    return messages

def generate_mav_prompt(mav_history, rav_history, action_options, mav_state, rav_state):
    """
    Generate MAV prompt, describing how MAV decides whether to cooperate or compete based on RAV's historical and current states.
    """
    # Build system message
    system_message = """
To make a decision in this scenario, we need to consider both safety and efficiency, taking into account the impact of each action on the system. Here's a breakdown of the factors:

### 1. **Cooperation (Decelerating or Changing Lanes):**
   - **Decelerating:** Reducing speed by decreasing your acceleration could create more space for the RAV to merge into your lane, but it will result in a decrease in your momentum, potentially causing a loss in overall efficiency.
   - **Changing Lanes:** Changing lanes would allow the RAV to merge without any risk of collision, but this maneuver could introduce time overheads. The change in position may require the MAV to adjust its trajectory, consuming more time and potentially causing a slight decrease in overall efficiency.

   **Pros:** Enhanced safety for both vehicles, smoother traffic flow.
   **Cons:** Potential for more time consumption or loss of speed, resulting in a delay in reaching the destination.

### 2. **Competition (Maintaining Position or Accelerating):**
   - **Maintaining Position:** If you maintain your position, the RAV might be forced to slow down or stop, but there is a risk of a collision if the RAV does not yield. This decision maintains your speed but could result in a tense situation with potential safety concerns.
   - **Accelerating:** By accelerating, you increase the gap between your MAV and the RAV, potentially blocking the RAV from merging into the lane. While this might seem like a more aggressive stance, it might create a safer scenario in which the RAV has to wait until it is safe to merge. However, this could lead to more congestion and inefficient traffic flow if multiple vehicles follow this strategy.

   **Pros:** Faster travel time for the MAV.
   **Cons:** Possible conflict with the RAV, creating an unsafe situation or aggressive driving.

### Decision Factors:
- **Safety:** Cooperation generally leads to safer outcomes, as it allows for smoother integration of the RAV into the flow of traffic. Acceleration or maintaining position may lead to an unsafe scenario.
- **Efficiency:** Maintaining speed or accelerating is more efficient in terms of travel time, but might come at the cost of safety and potentially block the RAV from merging.
- **Time Cost of Lane Change:** While changing lanes gives more space for the RAV, it introduces a time overhead due to the maneuver. However, if the lane change ensures smoother traffic flow, the cost may be worth it.

### Strategy:
- If the RAV is approaching at a high speed and the merge seems difficult, **decelerating** or **changing lanes** would be the safest bet, especially if the time cost of changing lanes is manageable.
- If the RAV is not approaching quickly and can merge smoothly, maintaining position or even **accelerating** might be a good way to maintain momentum, but it comes with a higher risk of conflict.
- If you're unsure about the behavior of the RAV, a combination of cooperation and slight deceleration or a cautious lane change might strike the right balance between safety and efficiency.

Ultimately, the choice depends on how much risk you're willing to take versus the time efficiency you're aiming for. In most cases, **cooperation through deceleration or lane change** should be prioritized for safety, but **competition (maintaining or increasing speed)** might be more suitable if time is a critical factor and the RAV is not a significant threat.

    eg: input: MAV current state:
    - Position: 103, Lane: 2, Speed: 22, Acceleration: 0

    RAV current state:
    - Position: 116, Lane: 3, Speed: 24.776699029126213, Acceleration: 0
    output:**RAV's Decision Prediction:**
    - Given the RAV's current state with a speed of 24.776699029126213 in lane 3 and no current acceleration, it appears that the RAV is maintaining a slightly higher speed than the MAV. If the RAV's historical state indicates a tendency to merge assertively or maintain its speed rather than slow down or yield, then P_cooperate,r = 0, suggesting it is likely to compete. Without specific historical data here, let's assume a cautious approach that it might compete due to its current speed status.

2. **Your Decision:**
   - Since the RAV might compete, we need to choose between cooperation and competition. The MAV is currently in lane 2 at a slightly lower speed (22) compared to the RAV. The relative positions indicate that if no action is taken, the MAV might cause inconvenience or a possible unsafe scenario for the RAV attempting to merge. 

   - Given safety considerations and the current state where the RAV is slightly ahead in speed, choosing to **cooperate** seems safer. This can be achieved by **decelerating** slightly to create more space for the RAV to merge, reducing potential conflict while maintaining a reasonable flow of traffic.

### **Action Output (Step 5):**

- **Decision:** --Cooperation--
- **Action:** %%decelerate%%

This decision and action should enhance safety by allowing the RAV to merge smoothly, reducing the likelihood of collision or aggressive maneuvers from either vehicle. It aligns with a strategy that prioritizes safety and system efficiency.
    """

    # MAV and RAV historical state data
    mav_history_str = "\n".join([
        f"MAV history - Position: {mav_state['x_m_t']}, Lane: {mav_state['lane_m_t']}, Speed: {mav_state['v_m_t']}, Acceleration: {mav_state['a_m_t']}\n" +
        f"RAV history - Position: {rav_state['x_r_t']}, Lane: {rav_state['lane_r_t']}, Speed: {rav_state['v_r_t']}, Acceleration: {rav_state['a_r_t']}"
        for mav_state, rav_state in zip(mav_history, rav_history)
    ])

    # Modify recommended actions based on action_options
    cooperate_actions = action_options["MAV"]["actions"]['cooperate']
    compete_actions = action_options["MAV"]["actions"]['compete']

    cooperate_message = f"Choose an action from the available cooperation actions: {', '.join(cooperate_actions)}"
    compete_message = f"Choose an action from the available competition actions: {', '.join(compete_actions)}"

    # Description of the current state
    current_state_message = f"""### **Input State Information (Step 1):**
**Historical State:**
{mav_history_str}

---

### **State Variables (Step 2):**
    MAV current state:
    - Position: {mav_state['x_m_t']}, Lane: {mav_state['lane_m_t']}, Speed: {mav_state['v_m_t']}, Acceleration: {mav_state['a_m_t']}

    RAV current state:
    - Position: {rav_state['x_r_t']}, Lane: {rav_state['lane_r_t']}, Speed: {rav_state['v_r_t']}, Acceleration: {rav_state['a_r_t']}

### **RAV's Behavior Prediction (Step 3):**
- Predict RAV's cooperation probability (P_cooperate,r) based on its historical states:
    - P_cooperate,r = 0 if RAV is likely to choose to compete.
    - P_cooperate,r = 1 if RAV is likely to choose to cooperate.

### **Decision Explanation (Step 4):**
1. **RAV's Decision Prediction:**
    - Based on RAV's historical behavior, predict if RAV will cooperate or compete. If RAV tends to cooperate, the probability of cooperation is high (P_cooperate = 1). If RAV tends to compete, the probability is low (P_cooperate = 0).

2. **Your Decision:**
    - Choose whether to cooperate or compete based on RAV's likely behavior. Explain why you made the choice.

    - **If you decide to cooperate:**
        - Suggested action: {cooperate_actions[0]} (Choose one action from the cooperation options: {', '.join(cooperate_actions)}).

    - **If you decide to compete:**
        - Suggested action: {compete_actions[0]} (Choose one action from the competition options: {', '.join(compete_actions)}).

### **Action Output (Step 5):**
Based on RAV’s Behavior Prediction, historical state and current state, choose your plan:
- **Action:** Choose **Cooperate** or **Compete**, and explain your decision. Provide recommended actions:
    
    - **If you choose to cooperate :**
        {cooperate_message}

    - **If you choose to compete:**
        {compete_message}
    
    Decision(For the decision, use -- to wrap the choice (e.g., --Cooperation--, --Competition--).): --?--
    Action(For the recommended action, use %% to wrap the action): %%?%%
    """

    # Combine into model input
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": current_state_message},
    ]

    return messages


def generate_prompts_for_mav_rav(m, t, max_initial_speed=35, max_initial_accel=2, allow_negative_distance=True,
                                 same_initial_conditions=False):
    """
    Generate prompts for MAV and RAV, including historical states, current states, and behavioral decision options.
    """

    # Randomly generate MAV and RAV historical states
    mav_history, rav_history, action_options = simulate_history_state(
        m, t, max_initial_speed, max_initial_accel, allow_negative_distance, same_initial_conditions
    )

    # Get the final MAV and RAV current states (i.e., the last pair of historical states)
    mav_state = mav_history[-1]
    rav_state = rav_history[-1]

    # Generate MAV prompt
    mav_prompt = generate_mav_prompt(mav_history, rav_history, action_options, mav_state, rav_state)

    # Generate RAV prompt
    rav_prompt = generate_rav_prompt(mav_history, rav_history, action_options, mav_state, rav_state)

    return mav_prompt, rav_prompt
