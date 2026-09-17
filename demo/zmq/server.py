import time
import zmq

context = zmq.Context()
socket = context.socket(zmq.REP)          # REP = Reply
socket.bind("tcp://*:5555")               # 服务端 bind


poller = zmq.Poller()
poller.register(socket, zmq.POLLIN)

print("[Server] 启动，等待请求...")
try:
    while True:
        # poll 带 1000ms 超时，到点一定返回
        events = dict(poller.poll(1000))
        if socket in events:
            msg = socket.recv_string()
            print(f"[Server] 收到: {msg}")
            time.sleep(0.5)                   # 模拟业务处理
            socket.send_string(f"已处理 -> {msg}")
except KeyboardInterrupt:
    print("\n[Server] 收到 Ctrl+C，退出")
finally:
    socket.close()
    context.term()