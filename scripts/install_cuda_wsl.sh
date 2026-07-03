#!/usr/bin/env bash
# Install the CUDA Toolkit (no driver) inside WSL2.
# Usage: bash scripts/install_cuda_wsl.sh
#
# Hard rule: never install a Linux display driver inside WSL2, and never install
# the 'cuda' or 'cuda-drivers' meta packages. They pull in a driver that shadows
# the stubs passed through from the Windows host in /usr/lib/wsl/lib, and
# nvidia-smi stops working. Install only cuda-toolkit-XX-Y.
set -euo pipefail

command -v systemd-detect-virt >/dev/null 2>&1 || true
grep -qi microsoft /proc/version || { echo "This does not look like WSL2, aborting."; exit 1; }

echo "==> [0/5] Removing any stale apt key"
sudo apt-key del 7fa2af80 2>/dev/null || true

echo "==> [1/5] Adding the NVIDIA wsl-ubuntu repository"
REPO="https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64"
KEYRING=$(curl -fsSL "$REPO/" | grep -o 'cuda-keyring_[0-9.-]*_all\.deb' | sort -V | tail -1)
[ -n "$KEYRING" ] || { echo "Could not resolve the cuda-keyring package name; check the network."; exit 1; }
echo "    using $KEYRING"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
curl -fsSL -o "$TMP/$KEYRING" "$REPO/$KEYRING"
sudo dpkg -i "$TMP/$KEYRING"
sudo apt-get update -qq

echo "==> [2/5] Selecting the newest cuda-toolkit package"
PKG=$(apt-cache search --names-only '^cuda-toolkit-[0-9]+-[0-9]+$' \
      | awk '{print $1}' | sort -V | tail -1)
[ -n "$PKG" ] || { echo "No cuda-toolkit-* found in the repository, aborting."; exit 1; }
echo "    will install: $PKG"
echo "    (note: not 'cuda' and not 'cuda-drivers' - those would install a driver)"

echo "==> [3/5] Installing (several GB, slow)"
sudo apt-get install -y "$PKG"

echo "==> [4/5] Writing PATH / LD_LIBRARY_PATH into ~/.bashrc (idempotent)"
CUDA_HOME=$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1)
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
if ! grep -q 'fused-llm-kernels: CUDA PATH' ~/.bashrc 2>/dev/null; then
  {
    echo ''
    echo '# fused-llm-kernels: CUDA PATH'
    echo "export CUDA_HOME=$CUDA_HOME"
    echo 'export PATH=$CUDA_HOME/bin:$PATH'
    echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}'
  } >> ~/.bashrc
  echo "    appended to ~/.bashrc"
else
  echo "    ~/.bashrc already configured, skipping"
fi
export PATH="$CUDA_HOME/bin:$PATH"

echo "==> [5/5] Verifying"
"$CUDA_HOME/bin/nvcc" --version | tail -2
ls "$CUDA_HOME/bin/ncu" >/dev/null 2>&1 && echo "    ncu in place: $CUDA_HOME/bin/ncu" \
  || echo "    !! ncu missing from $CUDA_HOME/bin - the nsight-compute package may need installing separately"

cat <<'MSG'

------------------------------------------------------------------
Two things still have to be done:
  1) Open a new WSL terminal (so ~/.bashrc applies), or run: source ~/.bashrc
  2) On the WINDOWS side: NVIDIA Control Panel -> Desktop menu ->
     "Enable Developer Settings" -> Developer -> Manage GPU Performance
     Counters -> "Allow access to the GPU performance counters to all users".
     Without this, ncu cannot read hardware counters (ERR_NVGPUCTRPERM).
     Restart WSL afterwards: run 'wsl --shutdown' in PowerShell.
Then rerun: bash scripts/env_check.sh
------------------------------------------------------------------
MSG
