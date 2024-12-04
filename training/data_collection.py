import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def collect_data(env_org, env, agent, episodes=1000, num_workers=4):
    state_data = []
    
    for _ in tqdm(range(episodes)):
        state, _ = env.reset()
        for _ in range(env_org.simulation_length):
            action = agent.get_action(env_org)
            next_state, _, done, _, _ = env.step(action)
            state_data.append(state)
            state = next_state
            if done:
                break

    state_data = np.array(state_data)
    return state_data


def preprocess_data(state_data):
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(state_data)
    train_data, test_data = train_test_split(normalized_data, test_size=0.2, random_state=42)
    joblib.dump(scaler, "state_scaler.pkl")
    return train_data, test_data



def load_and_preprocess_data(state_data_path="autoencoder_data.npz", scaler_path="state_scaler.pkl"):
    data = np.load(state_data_path)
    train_data = data["train"] 
    test_data = data["test"] 

    scaler = joblib.load(scaler_path)

    return train_data, test_data, scaler
