import highway_env
from highway_env.envs import HighwayEnv
import numpy as np



def calculate_time_to_zero(x, v, a):
    """
    Compute the time it takes the vehicle to reach the origin (assuming uniform acceleration).
    Solve the equation 0 = x + v * delta_t + 0.5 * a * delta_t^2
    Returns the time it takes to reach the origin, delta_t, if a solution exists, otherwise None.
    """
    if a == 0:
        # If the acceleration is 0, it becomes a linear equation v * delta_t = -x
        if v == 0:
            return None  # No solution (the vehicle cannot move)
        return x / v  # Linear solution

    # Solving quadratic equations 0 = x + v * delta_t + 0.5 * a * delta_t^2
    # The corresponding equation is 0 = 0.5 * a * delta_t^2 + v * delta_t + x
    A = 0.5 * a
    B = v
    C = x

    # Calculate the discriminant
    discriminant = B**2 + 4 * A * C
    if discriminant < 0:
        return None  # No solution (i.e. the vehicle cannot reach the origin)

    # Solve for two roots and choose the positive root
    delta_t1 = (-B + np.sqrt(discriminant)) / (2 * A)
    delta_t2 = (-B - np.sqrt(discriminant)) / (2 * A)

    # Only positive time values are returned
    valid_times = [t for t in [delta_t1, delta_t2] if t > 0]
    if valid_times:
        return min(valid_times)  # Returns the smallest positive time
    return None  # If there is no valid solution, return None

def calculate_average_speed(v, a, delta_t):
    """
    Use the formula v + 0.5 * a * delta_t to calculate the average velocity under acceleration.
    v: initial velocity of the vehicle
    a: acceleration of the vehicle
    delta_t: time step
    """
    v_avg = v + 0.5 * a * delta_t  # Average speed formula
    return v_avg
def calculate_ttc(x1, x2, v_avg1, v_avg2, epsilon=1e-6):
    """
    Calculate Time-to-Collision (TTC) using average speed.
    x1, x2: Position of vehicle 1 and vehicle 2
    v1, v2: Speed ​​of vehicle 1 and vehicle 2
    v_avg1, v_avg2: Average speed of vehicle 1 and vehicle 2
    epsilon: Threshold to avoid division by zero error
    """
    if abs(v_avg1 - v_avg2) > epsilon:
        # Calculate TTC using average speed
        ttc = abs(x2 - x1) / abs(v_avg1 - v_avg2)
        return ttc
    else:
        # If the relative velocity of the two vehicles is zero (or close to zero), then infinity is returned, meaning a collision is impossible
        return np.inf
class CustomMergeEnv(HighwayEnv):
    def __init__(self, mav_state=None, rav_state=None, **kwargs):
        super().__init__(**kwargs)
        self.mav_state = mav_state if mav_state else self.create_vehicle_state(agent_id=0, lane=2, position=10, vehicle_type='MAV')
        self.rav_state = rav_state if rav_state else self.create_vehicle_state(agent_id=1, lane=3, position=10, vehicle_type='RAV')
        self.num_lanes = 3
        self.lane_lengths = [200, 200, 100]
        self.vehicle_lengths = {'MAV': 5, 'RAV': 4.5}
        self.agents = [self.mav_state, self.rav_state]

    def create_vehicle_state(self, agent_id, lane, position, vehicle_type):
        state = {
            'id': agent_id,
            'lane': lane,
            'position': position,
            'speed': 5,  # Initial speed set to 5 m/s
            'acceleration': 0,  # Initial acceleration is 0
            'cooperate_prob': np.random.rand(),
            'length': self.vehicle_lengths[vehicle_type]
        }
        return state

    def reset(self):
        # Reset positions to 10m, and other states remain the same
        pass

    def step(self, actions):
        rewards = []
        done = False

        for agent, action in zip(self.agents, actions):
            agent['acceleration'] = action
            agent['speed'] += agent['acceleration']
            agent['position'] += agent['speed']

            # Check if vehicles have passed the origin (x < 0)
            if agent['position'] < 0:
                done = True

            reward = self.compute_reward(agent)
            agent['reward'] = reward
            rewards.append(reward)

        return [self.mav_state, self.rav_state], rewards, done, {}

    def compute_reward(self, agent):
        return agent['speed'] * 0.1

    def check_collision(self, vehicle1, vehicle2, epsilon=1e-6, tau_TTC=2.0):
        """
        Check collision risk based on the vehicle's position, velocity, and acceleration.
        If the TTC between the vehicles is less than tau_TTC, return a negative collision penalty.
        """
        # Get the status of vehicle 1 and vehicle 2 (position, lane, speed, acceleration)
        x1, lane1, v1, a1 = vehicle1['position'], vehicle1['lane'], vehicle1['speed'], vehicle1['acceleration']
        x2, lane2, v2, a2 = vehicle2['position'], vehicle2['lane'], vehicle2['speed'], vehicle2['acceleration']

        # Calculate the time it takes for the two vehicles to reach the origin (delta_t)
        delta_t1 = calculate_time_to_zero(x1, v1, a1)
        delta_t2 = calculate_time_to_zero(x2, v2, a2)

        # If the time cannot be calculated (i.e. the origin cannot be reached), return 0 (no collision)
        if delta_t1 is None or delta_t2 is None:
            return 0  # No risk of collision

        # Select the fastest time to reach the origin as delta_t
        delta_t = min(delta_t1, delta_t2)

        # Calculate the average speed of two cars
        v_avg1 = calculate_average_speed(v1, a1, delta_t)
        v_avg2 = calculate_average_speed(v2, a2, delta_t)

        # Calculating TTC（Time-to-Collision）
        ttc = calculate_ttc(x1, x2, v_avg1, v_avg2, epsilon)

        # If TTC is less than the threshold tau_TTC, then calculate the collision penalty
        if ttc < tau_TTC:
            penalty = - (tau_TTC - ttc) / tau_TTC  # The closer the collision is, the greater the penalty is, and a negative number is returned.
        else:
            penalty = 0  # No risk of collision

        return penalty


# Example Usage
config = {"duration": 200}
mav_initial_state = {
    'id': 0,
    'lane': 2,
    'position': 10,  # Initially 10 meters ahead of the origin
    'speed': 7,  # Initial speed 5 m/s
    'acceleration': 1,  # Initial acceleration 0
    'cooperate_prob': np.random.rand(),
    'length': 5
}
rav_initial_state = {
    'id': 1,
    'lane': 3,
    'position': 10,  # Initially 10 meters ahead of the origin
    'speed': 5,  # Initial speed 5 m/s
    'acceleration': 0,  # Initial acceleration 0
    'cooperate_prob': np.random.rand(),
    'length': 4.5
}

env = CustomMergeEnv(mav_state=mav_initial_state, rav_state=rav_initial_state, config=config)
initial_state = env.reset()
print("Initial state:", initial_state)

# Simulate a step with both vehicles accelerating
actions = [1, 1]  # Both MAV and RAV accelerate at 1 unit
next_state, rewards, done, info = env.step(actions)

print("Next state:", next_state)
print("Rewards:", rewards)
print("Done:", done)

# Check collision penalty
collision_penalty = env.check_collision(env.mav_state, env.rav_state)
print("Collision penalty:", collision_penalty)
