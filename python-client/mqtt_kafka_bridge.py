import json
import paho.mqtt.client as mqtt
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers="localhost:29092",
    value_serializer=lambda v: json.dumps(v).encode(),
)

def on_connect(client, userdata, flags, rc):
    print("Bridge connected to EMQX")
    client.subscribe("telemetry/#")

def on_message(client, userdata, msg):
    data = json.loads(msg.payload)
    # Key by VIN so all messages for one vehicle land in the same partition
    producer.send("vehicle-telemetry", key=data["vin"].encode(), value=data)
    print(f"{msg.topic} -> kafka")

client = mqtt.Client(client_id="mqtt-kafka-bridge")
client.on_connect = on_connect
client.on_message = on_message
client.connect("localhost", 1883, 60)
client.loop_forever()