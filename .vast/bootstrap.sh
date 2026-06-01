# set environment variables
export EDITOR=vim
env >> /etc/environment

# configure tmux
cat >> ~/.tmux.conf << 'EOF'
set-option -g history-limit 10000
EOF

# configure the shell
cat >> ~/.bashrc << 'EOF'
conda deactivate
cd /workspace/bartz
EOF

# install R
apt update -qq
apt install -y --no-install-recommends software-properties-common dirmngr
wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
add-apt-repository -y "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
sudo apt install -y --no-install-recommends r-base r-base-dev

# install R and python envs
make setup

# multi-GPU interconnect smoke test
# A broken inter-GPU link (e.g. SYS topology, no working P2P) makes NCCL
# collectives hang or crawl, silently wrecking data-sharded MCMC. Catch it here:
# time a cross-GPU all-reduce and check it is fast and numerically correct.
if [ "$(nvidia-smi -L 2>/dev/null | wc -l)" -gt 1 ]; then
    echo "multiple GPUs detected; running interconnect smoke test..."
    timeout 120 uv run python - <<'EOF'
import sys
import time

import jax
from jax import numpy as jnp
from jax import pmap

n = jax.device_count()
allreduce = pmap(lambda a: jax.lax.psum(a, "i"), axis_name="i")
x = jnp.ones((n, 2_000_000))  # 8 MB per shard
r = jax.block_until_ready(allreduce(x))  # warm up / NCCL init
times = []
for _ in range(5):
    t0 = time.perf_counter()
    r = jax.block_until_ready(allreduce(x))
    times.append(time.perf_counter() - t0)
ms = min(times) * 1e3
correct = abs(float(r[0, 0]) - n) < 1e-3
print(f"all-reduce: {ms:.2f} ms for 8 MB, result={float(r[0, 0])} (expect {n})")
if not correct or ms > 20:
    sys.exit(
        "BROKEN/SLOW multi-GPU interconnect: data sharding will hang or crawl. "
        "Rent offers with bw_nvlink>0."
    )
print("multi-GPU interconnect OK")
EOF
    rc=$?
    if [ "$rc" -eq 124 ]; then
        echo "ERROR: interconnect smoke test timed out (collective hung) -- broken" \
             "inter-GPU link; rent offers with bw_nvlink>0." >&2
    fi
    [ "$rc" -ne 0 ] && exit "$rc"
fi
