import gym
import numpy as np
import random
import json
import matplotlib.pyplot as plt
from gym import spaces
from stable_baselines3 import PPO

class MedicalCaseAssignmentEnv(gym.Env):
    def __init__(self):
        super(MedicalCaseAssignmentEnv, self).__init__()
        
        self.num_doctors = 5
        self.max_cases_per_doctor = 10  # Constraint on max cases per doctor per day
        self.sla_urgent_time = 10  # Urgent cases must be assigned within 10 minutes
        
        # Action space: choosing a doctor for the case
        self.action_space = spaces.Discrete(self.num_doctors)
        
        # Observation space: [patient profile, doctor availability, SLA compliance, dynamic prioritization]
        self.observation_space = spaces.Box(low=0, high=1, shape=(9,), dtype=np.float32)
        
        self.misassignments = np.zeros(self.num_doctors)  # Track misassignments for learning
        self.assignment_queue = []  # Store case assignments
        self.training_rewards = []  # Store rewards for training loss plot
        self.test_rewards = []  # Store rewards for test loss plot
        self.patient_wait_times = []  # Store patient wait times
        self.sla_compliance_rates = []  # Track SLA compliance
        self.reset()

    def reset(self):
        # Reset patient attributes
        self.patient_age = random.uniform(0, 1)  # Normalized age
        self.patient_gender = random.choice([0, 1])  # 0: Male, 1: Female
        self.medical_history = random.uniform(0, 1)  # Severity of past medical history
        self.case_urgency = random.uniform(0, 1)  # 1: Emergency, 0: Routine
        self.preferred_doctor = random.choice(range(self.num_doctors))
        self.dynamic_priority = self.case_urgency * 0.7 + self.medical_history * 0.3  # Weighted priority score
        
        # Reset doctor attributes
        self.doctor_specialty = np.random.choice([0, 1], self.num_doctors)  # 1: Specialist, 0: Generalist
        self.doctor_availability = np.random.choice([0, 1], self.num_doctors)  # 1: Available, 0: Unavailable
        self.doctor_experience = np.random.uniform(0, 1, self.num_doctors)  # Higher values = more experience
        self.doctor_ratings = np.random.uniform(0, 1, self.num_doctors)  # Patient ratings
        self.doctor_workload = np.zeros(self.num_doctors)  # Cases assigned per doctor today
        
        self.sla_compliance = random.uniform(0, 1)  # Compliance with hospital SLAs
        self.patient_wait_time = random.uniform(0, 20)  # Simulated patient wait time
        
        return self._get_observation()
    
    def _get_observation(self):
        return np.array([
            self.patient_age, self.patient_gender, self.medical_history,
            self.case_urgency, np.mean(self.doctor_availability),
            self.sla_compliance, np.mean(self.doctor_experience),
            np.mean(self.doctor_ratings), self.dynamic_priority
        ], dtype=np.float32)
    
    def step(self, action):
        if self.doctor_availability[action] == 0:
            reward = -10  # Heavy penalty for selecting unavailable doctor
            self.misassignments[action] += 1
        elif self.doctor_workload[action] >= self.max_cases_per_doctor:
            reward = -5  # Penalty for exceeding max cases per doctor per day
            self.misassignments[action] += 1
        else:
            reward = 10 * (1 - abs(self.medical_history - self.doctor_experience[action]))  # Matching expertise
            reward += 5 * self.case_urgency  # Reward urgency handling
            reward += 10 * self.sla_compliance  # Reward SLA compliance
            reward += 5 * self.doctor_ratings[action]  # Reward for selecting well-rated doctors
            reward -= 2 * self.misassignments[action]  # Penalize repeated misassignments
            
            self.doctor_workload[action] += 1  # Increase doctor workload
            self.misassignments[action] = max(0, self.misassignments[action] - 0.5)  # Gradual learning from misassignments
            
            # Save assignment to queue
            assignment = {
                "doctor": action,
                "patient": {
                    "age": self.patient_age,
                    "gender": self.patient_gender,
                    "medical_history": self.medical_history,
                    "urgency": self.case_urgency,
                    "preferred_doctor": self.preferred_doctor
                },
                "wait_time": self.patient_wait_time,
                "justification": f"Assigned based on experience ({self.doctor_experience[action]:.2f}), availability, ratings ({self.doctor_ratings[action]:.2f}), and urgency ({self.case_urgency:.2f})"
            }
            self.assignment_queue.append(assignment)
            self.patient_wait_times.append(self.patient_wait_time)
        
        # Dynamic prioritization update
        self.dynamic_priority = self.case_urgency * 0.7 + self.medical_history * 0.3
        
        # Track SLA compliance
        self.sla_compliance_rates.append(self.sla_compliance)
        
        next_state = self._get_observation()
        done = False
        
        return next_state, reward, done, {}

def evaluate_agent(env):
    avg_wait_time = np.mean(env.patient_wait_times)
    doctor_utilization = np.mean(env.doctor_workload) / env.max_cases_per_doctor
    misassignment_rate = np.mean(env.misassignments)
    sla_compliance_rate = np.mean(env.sla_compliance_rates)
    
    print(f"Average Patient Wait Time: {avg_wait_time:.2f} minutes")
    print(f"Doctor Utilization Rate: {doctor_utilization:.2f}")
    print(f"Misassignment Rate: {misassignment_rate:.2f}")
    print(f"SLA Compliance Rate: {sla_compliance_rate:.2f}")


def test_agent(env, model, num_episodes=10):
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action, _ = model.predict(state)
            state, reward, done, _ = env.step(action)


# Train the RL Agent
env = MedicalCaseAssignmentEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10)

test_agent(env, model)
evaluate_agent(env)
plot_training_rewards(env.training_rewards, "train_loss.png")
