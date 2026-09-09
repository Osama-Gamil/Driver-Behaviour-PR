import subprocess
from pathlib import Path
import time

'''Python OBD Client
       │
       │ MQTT Publish
       │
       ▼
┌──────────────┐
│     EMQX     │
│   :1883      │
└──────┬───────┘
       │
       │ Kafka Producer
       │ kafka:9092
       ▼
┌──────────────┐
│    Kafka     │
│   :9092      │
└──────┬───────┘
       │
       ▼
   obd-data
   topic
   '''

script_dir = Path(__file__).resolve().parent

def manage_infrastructure():
    print("Starting Docker containers (EMQX & Kafka)...")
    subprocess.run(["docker", "compose", "up", "-d"],cwd=script_dir ,check=True)
    print("Infrastructure is running. Press Ctrl+C to stop and tear down.")
    
#    try:
        # Keep the script alive to catch the termination signal
#        while True:
#            time.sleep(1)
#    except KeyboardInterrupt:
#        print("\nTearing down Docker infrastructure...")
#        subprocess.run(["docker", "compose", "down"])

def stop_infrastructure(): 
    print("Stoping Docker containers (EMQX & Kafka)...")
    subprocess.run(
        ["docker", "compose", "down"],
        cwd=script_dir,
        check=True
    )


def run_client_code():
    print("Starting OBD Client device")
    subprocess.run(
        ["python", "./python-client/single-client.py"],
        cwd=script_dir,
        check=True
    )


if __name__ == "__main__":
    manage_infrastructure()
    run_client_code()