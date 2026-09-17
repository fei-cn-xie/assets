import time
import zmq

context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.connect("tcp://localhost:5557")

poller = zmq.Poller()
poller.register(socket, zmq.POLLIN)

while True:
    events = dict(poller.poll(1000))
    if socket in events:
        task = socket.recv_string()
        print("[Pull] 处理:", task)
        time.sleep(0.3)