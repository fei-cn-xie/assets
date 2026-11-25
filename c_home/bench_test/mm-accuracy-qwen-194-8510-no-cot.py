"""TextVQA 多模态准确率测试（不开 CoT / thinking）。

参考脚本：private_home/vllm_script/bench_test/mm-accuracy-qwen-194-8510.py
唯一的行为差异：**每个请求都要求服务端跳过思考阶段**。

请求侧开关（vLLM 语义）：
    chat_template_kwargs = {"enable_thinking": false}
渲染 Qwen 系 chat template 时把思考段落关掉，模型直接给出答案。
它等价于请求体的 `reasoning_effort: "none"`（vLLM 会把 reasoning_effort != "none"
映射成 enable_thinking=True，显式传 enable_thinking 时以显式值为准）。

为什么要先跑探针：
    "不思考" 是否生效取决于三件事——模型自带的 chat template 是否识别
    enable_thinking、服务端是否吞掉了这个字段、服务端是否配了 reasoning parser。
    任一处不对，ais_bench 仍会静默跑到有 CoT 的结果，浪费一轮压测。
    因此 do_task() 之前先直连打一条请求做断言；不通过就报错退出。
    确认无误后可以 --skip-probe 跳过。

用法：
    python mm-accuracy-qwen-194-8510-no-cot.py                 # 先探针，再跑 ais_bench
    python mm-accuracy-qwen-194-8510-no-cot.py --probe-wait 600  # 边端还在加载时多等一会
    python mm-accuracy-qwen-194-8510-no-cot.py --skip-probe    # 跳过探针

探针的三类失败是分开报的，避免把"端口没起"误判成"CoT 没关"：
    net    端口/网络不通（服务没起、没就绪、或本机到目标不通）
    http   服务可达但 HTTP 报错（模型名不匹配、鉴权、引擎错）
    param  有回复但仍在思考（模板不认 enable_thinking 等）
探针还会**显式绕过 http_proxy/HTTPS_PROXY**：urllib 默认会把请求转给代理，
在带代理的机器上表现为 Connection refused（连的是代理而不是边端）。
"""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

# 请求侧的"关思考"开关：与 temperature/top_p 一样，属于 OpenAI 请求体字段，
# ais_bench 的 VLLMCustomAPIChatStream 会把 generation_kwargs 合并进请求体。
# 注意这里生成的是 **Python 配置**，所以布尔值必须写成 False（不是 JSON 的 false）。
THINKING_OFF = '{"enable_thinking": False}'


def config_dataset_jsonl(dataset_path, num_prompts):
    jsonl_name1 = f'{dataset_path}/textvqa_val_{num_prompts}.jsonl'
    jsonl_name2 = f'{dataset_path}/textvqa_val_{num_prompts}_annotations.json'
    if not os.path.exists(jsonl_name1):
        config1 = ''
        config2 = ''
        for i in range(num_prompts):
            config1 += (
                f'{{"image": "{dataset_path}/train_images/1080p.jpg", "question": "Explain the contents of the picture with more than 500 words and do not ", '
                f'"question_id": {34602 + i}, '
                '"answer": "woman and dog"}\n')
            config2 += ('{"image_id": 0,"answer_type": "other","question_type": "other",'
                       f'"question_id": {34602 + i},'
                       '"answers": [{"answer_id": 1023,"answer": "woman and dog","answer_confidence": "yes"}]},')
        with open(jsonl_name1, "w+", encoding="utf-8") as w:
            w.write(config1)
        with open(jsonl_name2, "w+", encoding="utf-8") as w:
            w.write('{"annotations": [' + config2[:-1] + ']}')
    return jsonl_name1


def get_generation_gkwargs_block():
    """generation_kwargs 的文本片段，额外带上 chat_template_kwargs。

    单独成串是为了不在 get_config_str 的 f-string 里写双大括号，
    生成的配置里是标准的 Python dict 字面量。
    """
    return (
        "        generation_kwargs = dict(\n"
        "            temperature = 0.0,\n"
        "            top_k = -1,\n"
        "            top_p = 1,\n"
        "            seed = None,\n"
        "            repetition_penalty = 1,\n"
        "            ignore_eos = False,\n"
        f"            chat_template_kwargs = {THINKING_OFF},\n"
        "        ),\n"
    )


def get_config_str(model_path, model_name, host_ip, request_rate, host_port, batch_size):
    return f"""
from ais_bench.benchmark.models import VLLMCustomAPIChatStream
from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="{model_path}",
        model="{model_name}",
        request_rate = {request_rate},
        retry = 2,
        host_ip = "{host_ip}",
        host_port = {host_port},
        max_out_len = 2048,
        batch_size={batch_size},
        trust_remote_code=False,
{get_generation_gkwargs_block()}        # 关思考后理论上不会再有 reasoning 段落；
        # 保留该后处理器只是兜底，正常应为空操作。
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]

    """


def get_editable_location():
    #import subprocess

    result = subprocess.run(
        "pip show ais_bench_benchmark",
        shell=True,
        capture_output=True,
        text=True
    )

    output = result.stdout

    for line in output.splitlines():
        if line.startswith("Editable project location:"):
            return line.split(":", 1)[1].strip()

    return None


def wait_for_port(host_ip, host_port, wait_s=60, interval=5):
    """TCP 轮询，等边端把端口绑上。返回 (ok, last_error)。

    边端 vLLM 是先等引擎就绪（含 PD 链路预热，最长 605s/条）再 uvicorn.run，
    所以在"正在加载"期间端口本来就是 refused。这里把它和真正的故障区分开。
    """
    deadline = time.monotonic() + max(0, wait_s)
    attempt = 0
    last_error = ""
    while True:
        attempt += 1
        try:
            with socket.create_connection((host_ip, host_port), timeout=3):
                return True, ""
        except OSError as exc:
            last_error = repr(exc)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False, last_error
            print(
                f"[probe] {host_ip}:{host_port} 暂不可连"
                f"（第 {attempt} 次：{exc}）—— 假设边端仍在加载，"
                f"最多再等 {remaining:.0f}s"
            )
            time.sleep(min(interval, remaining))


def probe_no_cot(model_name, host_ip, host_port, timeout=120, wait_s=60):
    """直连服务端，确认 enable_thinking=false 真的生效。

    返回 (ok, kind)，kind 用于区分责任方：
      ok     生效
      net    端口/网络不通（服务没起、没就绪、或本机到目标不通）——与 CoT 无关
      http   HTTP 层报错（404 模型名不对、401 鉴权、500 引擎错）
      param  有回复但仍在思考 —— 请求开关没生效
    """
    proxies = {
        key: value
        for key, value in os.environ.items()
        if key.lower() in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}
    }
    if proxies:
        print(f"[probe] 检测到代理环境变量 {proxies}；探针已显式绕过代理直连")

    # 关键：不用 urllib 的默认 opener —— 它会读取 http_proxy/HTTPS_PROXY，
    # 把请求转到代理上。在带代理的机器上，这会表现为 Connection refused
    # （连的是代理地址，而不是边端），极易误判成服务没起。
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    reachable, last_error = wait_for_port(host_ip, host_port, wait_s=wait_s)
    if not reachable:
        print(f"[probe] FAIL(net): {host_ip}:{host_port} 始终连不上 —— {last_error}")
        return False, "net"

    url = f"http://{host_ip}:{host_port}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "1+1 等于几？只回答数字。"}],
        "max_tokens": 32,
        "temperature": 0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )

    print(f"[probe] POST {url}  payload.chat_template_kwargs={payload['chat_template_kwargs']}")
    try:
        with opener.open(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        print(f"[probe] FAIL(http): HTTP {exc.code} — {detail}")
        return False, "http"
    except Exception as exc:  # noqa: BLE001 - 探针要给出可读结论，不抛栈
        print(f"[probe] FAIL(net): 请求失败 {exc!r}")
        return False, "net"

    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""

    think_markers = ("<" + "think" + ">", "<" + "/think" + ">")
    leaked = [marker for marker in think_markers if marker in content]

    print(f"[probe] content({len(content)} 字符): {content.strip()[:120]!r}")
    if reasoning:
        print(f"[probe] FAIL(param): message.reasoning_content 非空（{len(reasoning)} 字符）"
              " → 服务端仍在思考，enable_thinking=false 没生效")
        return False, "param"
    if leaked:
        print(f"[probe] FAIL(param): content 里出现思考标记 {leaked} → 模板没吃掉 enable_thinking")
        return False, "param"

    print("[probe] OK: 无 reasoning_content、无思考标记，服务端已按不思考处理")
    return True, "ok"


def do_task(skip_probe=False, probe_wait=60):
    global model_path, model_name, host_ip, host_port, dataset_path, test_parameter

    if not skip_probe:
        ok, kind = probe_no_cot(model_name, host_ip, host_port, wait_s=probe_wait)
        if not ok:
            if kind == "net":
                print("\n[probe] 已中止：边端端口不可达 —— 与 CoT 参数无关，先修连通性。")
                print("        在【执行本脚本的机器】上依次确认：")
                print(f"        1) 服务是否监听   ss -lntp | grep {host_port}")
                print(f"        2) 直连是否通     curl -sS -m 3 -o /dev/null -w '%{{http_code}}\\n' "
                      f"http://{host_ip}:{host_port}/v1/models")
                print("        3) 是否有代理     env | grep -i proxy   （探针已自动绕过代理，手工 curl 不会）")
                print("        4) 边端是否就绪   加载/预热最长可达 605s×IO 数，就绪前端口本来就是 refused")
                print(f"        就绪前想让它自己等：--probe-wait {max(probe_wait * 10, 600)}")
            elif kind == "http":
                print("\n[probe] 已中止：服务可达，但 HTTP 层返回错误。")
                print(f"        先核对 --served-model-name 是否等于 model_name={model_name!r}，再看鉴权/引擎状态。")
            else:
                print("\n[probe] 已中止：请求开关没生效 —— 服务端仍在思考。")
                print("        常见原因：模型模板不认 enable_thinking；服务端版本吞掉了该字段；")
                print("        或边端配了 reasoning parser，而请求没把 chat_template_kwargs 带过去。")
                print("        备选方案：边端加 --default-chat-template-kwargs '{\"enable_thinking\": false}'。")
            print("        （连通性/引擎确认无误后，可加 --skip-probe 跳过探针）")
            return 1
    else:
        print("[probe] 已按 --skip-probe 跳过")

    task_num = len(test_parameter)
    idx = 0
    while idx < task_num:
        msg = f">>> begin to do task: {idx + 1}/{task_num}  <<<"
        print(msg)
        batch_size = test_parameter[idx]['batch_size']
        num_prompts = test_parameter[idx]['num_prompts']
        request_rate = test_parameter[idx]['request_rate']
        #读取aisbench路径
        aisbench_location = get_editable_location()
        #构造数据集
        test_data_path = config_dataset_jsonl(dataset_path, num_prompts)
        dataset_script_name = f'{aisbench_location}/ais_bench/benchmark/configs/datasets/textvqa/textvqa_gen_base64.py'
        os.system(f'''sed -i 's|path=.*|path="{test_data_path}",|g' {dataset_script_name}''')
        api_file_name = f'{aisbench_location}/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py'
        api_file_bak_name = f'{aisbench_location}/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat_bk.py'
        if not os.path.exists(api_file_bak_name):
            os.system(
                f'mv {api_file_name} {api_file_bak_name}')
        with open(api_file_name, "w+", encoding="utf-8") as w:
            w.write(get_config_str(model_path=model_path, model_name=model_name, request_rate=request_rate, host_ip=host_ip, host_port=host_port, batch_size=batch_size))
        print(
            f"=== begin to run task {idx + 1}: batch_size={batch_size}, "
            f" request_rate={request_rate} ===")
        os.system(f"ais_bench --models vllm_api_stream_chat --datasets textvqa_gen_base64 --debug --num-prompts {num_prompts}")
        print(
            f">>> finish to do task {idx + 1}/{task_num}: batch_size={batch_size},"
            f" request_rate={request_rate} <<<\n")
        print('wait 3 minutes')
        for t in range(0):
            time.sleep(60)
            print(f'{t+1} minutes passed')
        idx += 1
    return 0


if __name__ == '__main__':
    # 模型权重路径
    model_path = '/weight/Qwen3.6-27B'
    # 模型权重名称
    model_name = "qwen3.6"
    host_ip = "76.76.26.194"
    host_port = 8510
    # 数据集的路径,准确到textvqa_val_100.jsonl文件和val_images文件夹所在目录
    dataset_path = "/data/multimodal_inputs/textvqa"
    # 测试参数，默认输入长度2731、输出长度256
    test_parameter = [
        {'batch_size': 100, 'num_prompts': 100, 'request_rate': 1},
                     ]
    raise SystemExit(
        do_task(
            skip_probe="--skip-probe" in sys.argv,
            probe_wait=int(sys.argv[sys.argv.index("--probe-wait") + 1])
            if "--probe-wait" in sys.argv
            else 60,
        )
    )
