import zmq

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5556")

poller = zmq.Poller()
poller.register(socket, zmq.POLLIN)

# 订阅前缀："" 表示订阅全部，b"TEMP" 表示只收 TEMP 开头
socket.setsockopt_string(zmq.SUBSCRIBE, "TEMP")

print("[Sub] 开始接收...")
try: 
    while True:
        events = dict(poller.poll(1000))
        if socket in events:
            msg = socket.recv_string()
            print("[Sub] 收到:", msg)
except KeyboardInterrupt:
    print("\n[Sub] 收到 Ctrl+C，退出")