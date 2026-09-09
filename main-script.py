
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
"""
Starts the whole system:
  1. Docker containers (EMQX, Kafka, Kafka UI, TimescaleDB)
  2. Waits until EMQX and Kafka actually accept connections
  3. Starts the MQTT -> Kafka bridge
  4. Starts the vehicle simulator
  5. Tears everything down on exit, no matter how we exit
"""

import socket
import subprocess
import sys
import time
from pathlib import Path

script_dir = Path(__file__).resolve().parent

# Use the same Python that is running this file, not whatever "python" means
# on PATH. On Windows that difference bites often.
PYTHON = sys.executable

BRIDGE_SCRIPT = script_dir / "python-client" / "mqtt_kafka_bridge.py"
CLIENT_SCRIPT = script_dir / "python-client" / "single-client.py"


def wait_for_port(host, port, name, timeout=90):
    """Keep trying to open a TCP socket until something answers."""
    print(f"Waiting for {name} on {host}:{port} ...", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                print(" ready.")
                return True
        except OSError:
            print(".", end="", flush=True)
            time.sleep(2)
    print(f"\n{name} did not become ready within {timeout}s.")
    return False


def start_infrastructure():
    print("Starting Docker containers...")
    subprocess.run(["docker", "compose", "up", "-d"], cwd=script_dir, check=True)


def stop_infrastructure():
    print("\nStopping Docker containers...")
    subprocess.run(["docker", "compose", "down"], cwd=script_dir, check=False)


def main():
    bridge = None
    try:
        start_infrastructure()

        # "Started" from Docker only means the process launched. EMQX needs
        # another 15-30s before its MQTT listener is open. This is what was
        # causing the rc=7 disconnects.
        if not wait_for_port("localhost", 1883, "EMQX"):
            return
        if not wait_for_port("localhost", 29092, "Kafka"):
            return

        # Bridge starts BEFORE the vehicles. MQTT does not replay history,
        # so anything published before the bridge subscribes is lost.
        print("Starting MQTT -> Kafka bridge...")
        bridge = subprocess.Popen([PYTHON, str(BRIDGE_SCRIPT)], cwd=script_dir)
        time.sleep(3)

        if bridge.poll() is not None:
            print("Bridge exited immediately. Check its output above.")
            return

        print("Starting OBD client...\n")
        subprocess.run([PYTHON, str(CLIENT_SCRIPT)], cwd=script_dir, check=False)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except subprocess.CalledProcessError as exc:
        print(f"\nA command failed: {exc}")
    finally:
        # 'finally' means this runs on Ctrl+C, on crash, and on normal exit.
        # Without it, containers keep running after the script dies.
        if bridge is not None and bridge.poll() is None:
            print("Stopping bridge...")
            bridge.terminate()
            try:
                bridge.wait(timeout=5)
            except subprocess.TimeoutExpired:
                bridge.kill()
        stop_infrastructure()
        print("Done.")


if __name__ == "__main__":
    main()