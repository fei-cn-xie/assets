这是可直接运行的 Python ZeroMQ 通信演示，用三种经典模式展示了消息收发的基本用法。
```python
# 安装依赖
# pip install pyzmq
```

## 1. REQ/REP 模式（请求-应答，最常用）

**server.py**
```python
import time
import zmq

context = zmq.Context()
socket = context.socket(zmq.REP)          # REP = Reply
socket.bind("tcp://*:5555")               # 服务端 bind

print("[Server] 启动，等待请求...")
try:
    while True:
        msg = socket.recv_string()
        print(f"[Server] 收到: {msg}")
        time.sleep(0.5)                   # 模拟业务处理
        socket.send_string(f"已处理 -> {msg}")
except KeyboardInterrupt:
    pass
finally:
    socket.close()
    context.term()
```

**client.py**
```python
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
```

运行：
```bash
python server.py     # 先启动
python client.py     # 再启动
```

---

## 2. PUB/SUB 模式（发布-订阅，一对多广播）

**publisher.py**
```python
import time
import zmq

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5556")

time.sleep(1)   # 等订阅者连接（解决 slow joiner 问题）

for i in range(10):
    topic = "TEMP" if i % 2 == 0 else "HUMI"
    msg = f"{topic} {20 + i}"
    socket.send_string(msg)
    print("[Pub] 发布:", msg)
    time.sleep(0.5)

socket.close()
context.term()
```

**subscriber.py**
```python
import zmq

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5556")

# 订阅前缀："" 表示订阅全部，b"TEMP" 表示只收 TEMP 开头
socket.setsockopt_string(zmq.SUBSCRIBE, "TEMP")

print("[Sub] 开始接收...")
while True:
    msg = socket.recv_string()
    print("[Sub] 收到:", msg)
```

运行：
```bash
python subscriber.py    # 先启动订阅者
python publisher.py     # 再启动发布者
```

---

## 3. PUSH/PULL 模式（流水线，任务分发）

**producer.py**
```python
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
```

**worker.py**
```python
import time
import zmq

context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.connect("tcp://localhost:5557")

while True:
    task = socket.recv_string()
    print("[Pull] 处理:", task)
    time.sleep(0.3)
```

多个 worker 同时启动即可自动负载均衡。

---

## 4. 单文件版（用 multiprocessing 一次跑起来）

如果你不想开两个终端，可以用这个：

```python
import multiprocessing as mp
import time
import zmq


def server():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind("tcp://127.0.0.1:5555")
    while True:
        msg = sock.recv_string()
        print(f"[Server] 收到: {msg}")
        sock.send_string(f"echo: {msg}")


def client():
    time.sleep(0.5)          # 等服务端 bind 完成
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect("tcp://127.0.0.1:5555")
    for i in range(5):
        sock.send_string(f"hello {i}")
        print(f"[Client] 回复: {sock.recv_string()}")


if __name__ == "__main__":
    p = mp.Process(target=server, daemon=True)
    p.start()

    client()

    p.terminate()
```

---

## 几个容易踩的坑

| 问题 | 说明 |
|---|---|
| **REQ/REP 必须交替** | REQ 端 `send` 之后必须 `recv`，否则再次 `send` 会报错。需要异步请用 DEALER/ROUTER |
| **PUB/SUB 慢连接** | SUB 连接后收不到之前发的消息，所以 PUB 端要 `sleep` 一会儿再发 |
| **SUB 必须设置订阅** | 不调用 `setsockopt(zmq.SUBSCRIBE, ...)` 则收不到任何消息 |
| **bind vs connect** | 谁先启动谁 bind，一般服务端 bind、客户端 connect；两边都 connect 也可以（如 broker 场景） |
| **不要在 recv 里做重活** | 会阻塞整个 socket，重活交给 worker 线程/进程 |
| **关闭顺序** | 先 `socket.close()` 再 `context.term()`，否则可能挂住 |

需要我再补充 ROUTER/DEALER 异步模式、或加上多线程 / asyncio 版本吗？