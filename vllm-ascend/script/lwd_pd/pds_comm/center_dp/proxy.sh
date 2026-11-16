python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
    --host 0.0.0.0 \
    --port 9060 \
    --prefiller-hosts 76.76.26.189 \
    --prefiller-ports 8060 \
    --decoder-hosts 76.76.26.189 \
    --decoder-ports 8061
