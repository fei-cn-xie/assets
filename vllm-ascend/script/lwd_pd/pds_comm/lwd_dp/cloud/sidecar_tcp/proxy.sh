python /vllm-workspace/vllm-ascend/examples/edge_cloud_pd_sidecar/proxy.py \
  --host 0.0.0.0 \
  --port 8600 \
  --prefill-url http://76.76.26.189:8160/v1/chat/completions \
  --decode-url http://76.76.26.189:8161/v1/chat/completions
