'''This file manages connection. Doesnt load PyTorch or the environment directly, only processes text & sends the JSON commands to train_trials.py'''
import socket
import json
import time

# ==========================================
# 1. Configuration
# ==========================================
HOST = 'localhost'
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)

def text_to_lam_values(prompt):
    """
    Converts text to a list of floats (to be sent over JSON).
    """
    # [Urgency, Caution, ... others]
    cmd_vector = [0.0] * 10 
    prompt = prompt.lower()
    
    if "hurry" in prompt or "fast" in prompt:
        cmd_vector[0] = 1.0
    elif "slow" in prompt:
        cmd_vector[0] = -1.0
        
    if "careful" in prompt or "avoid" in prompt:
        cmd_vector[1] = 1.0
    elif "reckless" in prompt:
        cmd_vector[1] = -1.0
        
    return cmd_vector

def main():
    print(f"--- COMMANDER TERMINAL SERVER ---")
    print(f"Status: ONLINE. Listening on {HOST}:{PORT}")
    print(f"Action: Please run 'train_trials.py' in a separate window now.")
    print("-----------------------------------")

    # 1. Setup Server Socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        
        # This line BLOCKS until train_trials.py connects
        conn, addr = s.accept()
        
        with conn:
            print(f"\n[System] Robot connected from {addr}!")
            print("[System] Ready for commands.")
            print("Type 'Run Trial 1', 'Go to 5,5 careful', or 'exit'.\n")

            while True:
                try:
                    user_input = input("COMMANDER >> ")
                    
                    if not user_input:
                        continue
                        
                    if user_input.lower() in ['exit', 'quit']:
                        print("[System] Sending shutdown signal...")
                        conn.sendall(json.dumps({"action": "shutdown"}).encode())
                        break

                    # --- Parse Command ---
                    # Default: Run a custom training session
                    mission_data = {
                        "action": "train",
                        "start": [0, 0],
                        "goal": [9, 9],
                        "lam_vector": text_to_lam_values(user_input),
                        "steps": 10000  # How long to train per command
                    }

                    # Basic parsing logic
                    if "trial 1" in user_input.lower():
                        mission_data["start"] = [0, 0]
                        mission_data["goal"] = [9, 9]
                    elif "trial 2" in user_input.lower():
                        mission_data["start"] = [9, 9]
                        mission_data["goal"] = [0, 0]
                    elif "go to" in user_input.lower():
                        try:
                            # Parse "Go to x,y"
                            parts = user_input.lower().split("go to")[1].strip().split(",")
                            mission_data["goal"] = [int(parts[0]), int(parts[1])]
                        except:
                            print("[Error] Could not parse coordinates. Using default.")

                    # --- Send to Robot ---
                    print(f"[System] Uploading mission parameters to robot...")
                    conn.sendall(json.dumps(mission_data).encode())

                    # --- Wait for Completion ---
                    print("[System] Robot is training... (Waiting for report)")
                    response = conn.recv(1024).decode()
                    print(f"[Robot Report] {response}")

                except BrokenPipeError:
                    print("\n[Error] Robot disconnected unexpectedly.")
                    break
                except KeyboardInterrupt:
                    break

if __name__ == "__main__":
    main()