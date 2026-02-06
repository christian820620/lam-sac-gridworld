"""
TRAINING WORKER CLIENT
This script connects to the Commander Terminal and executes training
tasks on demand. It cannot run standalone.
"""
import socket
import json
import os
import numpy as np
import torch

# --- GRAPHICS BACKEND FIX ---
import matplotlib
# Force the native Mac driver. 
# This avoids the "Qt platform plugin 'cocoa'" crash entirely.
try:
    matplotlib.use('MacOSX') 
except:
    print("[Warning] MacOSX backend not found. Graphics might be disabled.")

import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

# --- CRITICAL IMPORT ---
# This was missing in the previous error. It pulls your custom AI from the other file.
from lam_sac_env import LargeActionModel, TwoWallsGap10x10LAMEnv

# Configuration
HOST = 'localhost'
PORT = 65432

def evaluate_and_visualize(model, start, goal, lam_model, lam_vector, n_episodes=3):
    """
    Visualizes the result of the training for the user to see.
    """
    print(f"   -> Visualizing performance...")
    env = TwoWallsGap10x10LAMEnv(goal=tuple(goal), start=tuple(start), lam_model=lam_model)
    
    # Inject the specific LAM context (Urgency/Caution)
    tensor_ctx = torch.tensor(lam_vector, dtype=torch.float32)
    env.update_lam_context(tensor_ctx)

    plt.ion()
    # We use a try/except block for plotting to ensure training doesn't crash if graphics fail
    try:
        fig = plt.gcf()
        if not plt.fignum_exists(fig.number):
             fig, ax = plt.subplots(figsize=(5, 5))
        else:
             ax = plt.gca()
             ax.clear()
    except:
        fig, ax = plt.subplots(figsize=(5, 5))

    img = None
    successes = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Render
            try:
                frame = env.render()
                if img is None:
                    img = ax.imshow(frame)
                    ax.axis('off')
                else:
                    img.set_data(frame)
                plt.pause(0.01)
            except:
                pass # Skip render if window closed

            if done and info.get("is_success"):
                successes += 1
    
    env.close()
    return successes / n_episodes

def main():
    os.makedirs("models", exist_ok=True)
    
    # Initialize the custom Neural Network Class
    lam_model = LargeActionModel()
    
    print(f"--- ROBOT TRAINING CLIENT ---")
    print(f"Attempting to connect to Commander at {HOST}:{PORT}...")
    
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((HOST, PORT))
        print("SUCCESS: Connected to Commander Terminal.")
    except ConnectionRefusedError:
        print("\n[CRITICAL ERROR] Connection Refused.")
        print("The Commander Terminal is NOT running.")
        print("You must start 'commander_terminal.py' first.")
        return

    # Load existing model or create new one
    model_path = "models/sac_lam_trials_final.zip"
    if os.path.exists(model_path):
        print("Loaded existing model.")
        model = SAC.load(model_path)
    else:
        print("Created new SAC model.")
        # Create a dummy env just to initialize the model shape
        dummy_env = DummyVecEnv([lambda: TwoWallsGap10x10LAMEnv(goal=(9,9))])
        model = SAC("MlpPolicy", dummy_env, verbose=1)

    # --- MAIN LISTENING LOOP ---
    try:
        while True:
            print("\n[Standby] Waiting for orders...")
            data = client_socket.recv(4096)
            if not data:
                break
            
            command = json.loads(data.decode())
            
            if command.get("action") == "shutdown":
                print("Shutdown signal received.")
                break
                
            if command.get("action") == "train":
                start = command['start']
                goal = command['goal']
                lam_vec = command['lam_vector']
                steps = command.get('steps', 5000)
                
                print(f"[Mission Start] Training: Start={start} -> Goal={goal}")
                print(f"               Constraints: {lam_vec[:2]}...")

                # 1. Setup Environment with new parameters
                env = TwoWallsGap10x10LAMEnv(goal=tuple(goal), start=tuple(start), lam_model=lam_model)
                
                # Apply LAM Context (Convert list back to Tensor)
                ctx_tensor = torch.tensor(lam_vec, dtype=torch.float32)
                env.update_lam_context(ctx_tensor)
                
                # 2. Train
                venv = DummyVecEnv([lambda: env])
                model.set_env(venv)
                model.learn(total_timesteps=steps)
                
                # 3. Save
                model.save(model_path)
                
                # 4. Evaluate (Visual)
                success_rate = evaluate_and_visualize(model, start, goal, lam_model, lam_vec)
                
                # 5. Report back
                msg = f"Training Complete. Success Rate: {success_rate*100:.1f}%"
                client_socket.sendall(msg.encode())

    except KeyboardInterrupt:
        print("Stopping manually.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client_socket.close()
        print("Disconnected.")

if __name__ == '__main__':
    main()