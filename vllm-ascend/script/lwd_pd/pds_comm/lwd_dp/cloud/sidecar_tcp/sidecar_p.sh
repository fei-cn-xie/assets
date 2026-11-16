mkdir -p /tmp/lwd-p

python /vllm-workspace/vllm/tools/lwd_sidecar.py \
  --directory /bash_dir/pds_comm/sidecar_tcp/lwd_sidecar_directory.json \
  --instance cloud-p \
  --log-level INFO \
  --stats-interval 10
