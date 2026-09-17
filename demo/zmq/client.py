import zmq

context = zmq.Context()
socket = context.socket(zmq.REQ)          # REQ = Request
socket.connect("tcp://localhost:5555")    # 客户端 connect

for i in range(5):
    msg = f"请求 #{i}"
    print(f"[Client] 发送: {msg}")
    socket.send_string(msg)

    reply = socket.recv_string()          # REQ 必须 send/recv 严格交替
    print(f"[Client] 回复: {reply}")

socket.close()
context.term()