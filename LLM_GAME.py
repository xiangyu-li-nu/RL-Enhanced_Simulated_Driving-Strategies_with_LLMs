import gym
import numpy as np
import torch
from transformers import pipeline
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import trange


# Step 1: Load LLM model pipeline
def load_llm_pipeline(model_id, device_map):
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    return pipe


# GPU allocation
device_map_a = "cuda:0"  # Player A uses GPU 0
device_map_b = "cuda:1"  # Player B uses GPU 1

# Model paths for Player A and B
model_id_a = r"D:\LLM_for_AV\llama3.1-7b"
model_id_b = r"D:\LLM_for_AV\Llama-3.2-1B-Instruct"

pipe_a = load_llm_pipeline(model_id_a, device_map_a)
pipe_b = load_llm_pipeline(model_id_b, device_map_b)


# Step 2: Define the environment (with history tracking)
class PPOGameEnv(gym.Env):
    def __init__(self, opponent_policy=None, agent_player="A", max_rounds=10):
        super(PPOGameEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(3)  # Action space with 3 possible actions
        self.observation_space = gym.spaces.Box(
            low=np.array([10, 10, 10, 10]),  # Lower bound of the observation space
            high=np.array([10, 100, 10, 100]),  # Upper bound of the observation space
            dtype=np.float32
        )
        self.max_position = 100  # Maximum track length
        self.max_rounds = max_rounds  # Maximum number of rounds
        self.round_count = 0  # Initial round count
        self.opponent_policy = opponent_policy  # The opponent's strategy
        self.agent_player = agent_player  # The current player's identity (A or B)
        self.history = []  # Initialize action and result history
        self.reset()

    def reset(self):
        self.state_a = np.array([0, 0], dtype=np.float32)  # Initial state for Player A (speed, position)
        self.state_b = np.array([0, 0], dtype=np.float32)  # Initial state for Player B (speed, position)
        self.round_count = 0  # Reset round counter
        self.history = []  # Clear history on reset
        return np.concatenate([self.state_a, self.state_b])  # Return the concatenated state of both players

    def step(self, action):
        # Get opponent's action
        action_b = self.opponent_policy.get_action(self.state_b, self.history, player="B")  # Get action for Player B
        action_a = self.opponent_policy.get_action(self.state_a, self.history, player="A")  # Get action for Player A
        
        # Update states based on actions
        self.state_a = self.update_state(self.state_a, action_a, action_b)
        self.state_b = self.update_state(self.state_b, action_b, action_a)

        self.round_count += 1  # Increment the round counter
        done = (
                self.state_a[1] >= self.max_position or  # Player A reaches the end
                self.state_b[1] >= self.max_position or  # Player B reaches the end
                self.round_count >= self.max_rounds  # Maximum rounds reached
        )

        reward = self.compute_reward(done)  # Compute the reward based on the game state
        obs = np.concatenate([self.state_a, self.state_b])  # Combine the states of both players
        self.history.append((action_a, action_b, reward))  # Append the actions and reward to the history
        info = {}  # Additional info (empty for now)
        return obs, reward, done, info

    def update_state(self, state, action, opponent_action):
        speed, position = state
        # Update speed and position based on actions
        if action == 0 and opponent_action == 1:
            speed += 1
        elif action == 1 and opponent_action == 2:
            speed = max(speed - 1, 0)
        elif action == 2 and opponent_action == 0:
            speed += 1
        else:
            speed = max(speed - 1, 0)

        position += speed
        position = min(position, self.max_position)  # Ensure position doesn't exceed max position
        return np.array([speed, position], dtype=np.float32)  # Return the updated state

    def compute_reward(self, done):
        if done:
            if self.agent_player == "A":
                if self.state_a[1] >= self.max_position:  # Player A wins
                    reward = 1
                elif self.state_b[1] >= self.max_position:  # Player B wins
                    reward = -1
                else:
                    reward = 0  # Draw
            else:
                if self.state_b[1] >= self.max_position:  # Player B wins
                    reward = 1
                elif self.state_a[1] >= self.max_position:  # Player A wins
                    reward = -1
                else:
                    reward = 0  # Draw
        else:
            reward = 0  # No reward if the game isn't over
        return reward


def extract_wrapped_content(text):
    import re
    """
    Extract the content wrapped first by `--` and then by `%%`, and remove special symbols (like `*`).
    Ensure that the first element is the content wrapped by `--` and the second element is the content wrapped by `%%`.
    """
    # Extract the content wrapped first by `--`
    first_match = re.search(r'--([^\*]+)--', text)
    decision = first_match.group(1) if first_match else None

    # Return the extracted result
    return decision

# Step 3: LLM action generation logic using pipeline
class LLMPolicy:
    def __init__(self, pipe):
        self.pipe = pipe  # Initialize with the language model pipeline

    def get_action(self, state, history, player="A"):
        history_text = self.format_history(history)  # Format the history for the current player
        system_message = (
            "You are a strategic game player in a competitive environment.\n\n"
            "Game Rules:\n"
            "1. There are two players: Player A and Player B. Each player controls a vehicle. The goal is to reach the finish line first.\n"
            "2. In each turn, you have the following three possible actions:\n"
            "   -- 0 -- Accelerate: Increase your speed by 1.\n"
            "   -- 1 -- Decelerate: Decrease your speed by 1.\n"
            "   -- 2 -- Maintain speed: Keep your current speed.\n"
            "3. The actions are competitive, meaning:\n"
            "   - If both players accelerate, Player A’s speed increases while Player B’s speed stays the same.\n"
            "   - If one player accelerates and the other decelerates, the accelerating player gets ahead, while the decelerating player loses speed.\n"
            "   - If one player accelerates and the other maintains speed, the accelerating player moves ahead.\n"
            "   - If both players decelerate, their speeds decrease, but the one decelerating less maintains the advantage.\n"
            "   - If both players maintain speed, their speeds stay the same.\n"
            "4. The winner is the first player to reach the finish line. If both players reach the finish line at the same time, the player with the highest speed wins.\n"
            "5. Each player's actions and progress are recorded in the game history.\n\n"
            "Player actions must be chosen using the following format:\n"
            "--0-- for Accelerate, --1-- for Decelerate, --2-- for Maintain speed.\n"
            "Your response must be exactly one of the following: --0--, --1--, or --2--.\n"
            "You must respond based on your current state and history."
        )
        current_state_message = (
            f"Player {player}, your current state is:\n"
            f"Speed: {state[0]}, Position: {state[1]}.\n"
            f"History of actions:\n{history_text}\n"
            "Choose your action as a number, wrapped in double asterisks (--):\n"
            "--0-- for Accelerate, --1-- for Decelerate, --2-- for Maintain speed.\n"
            "Your response must be exactly one of the following: --0--, --1--, or --2--."
        )
        messages = [
            {"role": "system", "content": system_message},  # System message with the game rules
            {"role": "user", "content": current_state_message},  # Player's current state and history
        ]

        # Generate response
        outputs = self.pipe(messages, max_new_tokens=512)
        decision_explanation = outputs[0]["generated_text"][-1]["content"]  # Extract the explanation from LLM output
        print(f"LLM Output for Player {player}: {decision_explanation}")
        action = extract_wrapped_content(decision_explanation)  # Extract the action (e.g., '--0--', '--1--', '--2--')
        return action

    @staticmethod
    def format_history(history):
        if not history:
            return "No previous rounds."  # Return a message if there is no previous history
        formatted = []
        for i, (a_action, b_action, reward) in enumerate(history):
            formatted.append(f"Round {i + 1}: Player A: {a_action}, Player B: {b_action}, Reward: {reward}")
        return "\n".join(formatted)  # Format the history into a string

# Create LLM policies for both players
policy_a = LLMPolicy(pipe_a)  # Player A's strategy using LLM
policy_b = LLMPolicy(pipe_b)  # Player B's strategy using LLM

# Create environments
env_a = DummyVecEnv([lambda: PPOGameEnv(opponent_policy=policy_b, agent_player="A")])  # Environment for Player A
env_b = DummyVecEnv([lambda: PPOGameEnv(opponent_policy=policy_a, agent_player="B")])  # Environment for Player B

# Create PPO models for both agents
ppo_a = PPO("MlpPolicy", env_a, verbose=1, device="cuda:0", n_steps=128, batch_size=64)  # PPO model for Player A
ppo_b = PPO("MlpPolicy", env_b, verbose=1, device="cuda:1", n_steps=128, batch_size=64)  # PPO model for Player B

# Step 4: Adversarial training
num_rounds = 1000  # Number of training rounds

for round in trange(num_rounds, desc="Training"):
    ppo_a.learn(total_timesteps=500, reset_num_timesteps=False)  # Train Player A
    ppo_b.learn(total_timesteps=500, reset_num_timesteps=False)  # Train Player B

# Step 5: Test results
envs = [env_a, env_b]  # Environments for both players
models = [ppo_a, ppo_b]  # PPO models for both players
players = ["A", "B"]  # Player identifiers

for player_idx, (env, model, player) in enumerate(zip(envs, models, players)):
    obs = env.reset()  # Reset environment for testing
    done = False
    print(f"\nTesting Player {player}:")
    while not done:
        action, _ = model.predict(obs, deterministic=True)  # Predict action based on current state
        obs, reward, done, info = env.step(action)  # Take step in the environment
        print(f"Player {player}: State: {obs}, Reward: {reward}")

