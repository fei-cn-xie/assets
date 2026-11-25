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
    python mm-accuracy-qwen-194-8510-no-cot.py            # 先探针，再跑 ais_bench
    python mm-accuracy-qwen-194-8510-no-cot.py --skip-probe
"""

import json
import os
import subprocess
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


def probe_no_cot(model_name, host_ip, host_port, timeout=120):
    """直连服务端，确认 enable_thinking=false 真的生效。

    判据（任一命中即视为"仍在思考"）：
      * 返回体 message.reasoning_content 非空；
      * content 里出现思考起止标记（由调用方用 markers 传入）。
    """
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
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        print(f"[probe] FAIL: HTTP {exc.code} — {detail}")
        return False
    except Exception as exc:  # noqa: BLE001 - 探针要给出可读结论，不抛栈
        print(f"[probe] FAIL: 请求失败 {exc!r}")
        return False

    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""

    think_markers = ("<" + "think" + ">", "<" + "/think" + ">")
    leaked = [marker for marker in think_markers if marker in content]

    print(f"[probe] content({len(content)} 字符): {content.strip()[:120]!r}")
    if reasoning:
        print(f"[probe] FAIL: message.reasoning_content 非空（{len(reasoning)} 字符）"
              " → 服务端仍在思考，enable_thinking=false 没生效")
        return False
    if leaked:
        print(f"[probe] FAIL: content 里出现思考标记 {leaked} → 模板没吃掉 enable_thinking")
        return False

    print("[probe] OK: 无 reasoning_content、无思考标记，服务端已按不思考处理")
    return True


def do_task(skip_probe=False):
    global model_path, model_name, host_ip, host_port, dataset_path, test_parameter

    if not skip_probe:
        if not probe_no_cot(model_name, host_ip, host_port):
            print("\n[probe] 已中止：请先解决上面的问题，或用 --skip-probe 强制跳过探针。")
            print("        常见原因：模型模板不认 enable_thinking；服务端版本吞掉了该字段；")
            print("        服务端配了 reasoning parser 但请求没带 chat_template_kwargs。")
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
    raise SystemExit(do_task(skip_probe="--skip-probe" in os.sys.argv))
