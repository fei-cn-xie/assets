mkdir -p /tmp/lwd-d

python /vllm-workspace/vllm/tools/lwd_sidecar.py \
  --directory /bash_dir/pds_comm/sidecar_tcp/lwd_sidecar_directory.json \
  --instance cloud-d \
  --log-level INFO \
  --stats-interval 10
