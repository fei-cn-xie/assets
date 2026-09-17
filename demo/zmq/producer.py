import time
import zmq

context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.bind("tcp://*:5557")

for i in range(10):
    socket.send_string(f"任务-{i}")
    print("[Push] 发送任务:", i)
    time.sleep(0.2)

socket.close()
context.term()