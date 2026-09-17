import time
import zmq

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5556")

time.sleep(10)   # 等订阅者连接（解决 slow joiner 问题）

for i in range(10):
    topic = "TEMP" if i % 2 == 0 else "HUMI"
    msg = f"{topic} {20 + i}"
    socket.send_string(msg)
    print("[Pub] 发布:", msg)
    time.sleep(0.5)

socket.close()
context.term()