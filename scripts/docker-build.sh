#!/usr/bin/env bash
# Build the kb-core image locally and save it to a tarball ready for transfer.
#
# Usage:
#     ./scripts/docker-build.sh              # tags as kb-core:latest, saves to ./kb-core.tar
#     ./scripts/docker-build.sh v1.0.0       # tags as kb-core:v1.0.0, saves to ./kb-core-v1.0.0.tar
#
# Output: a tarball you scp/rsync to the server, then `docker load -i` there.

set -euo pipefail

TAG="${1:-latest}"
IMAGE="kb-core:${TAG}"
OUT="kb-core-${TAG}.tar"

cd "$(dirname "$0")/.."

# Force linux/amd64 because:
# 1. Most Aliyun ECS / general cloud Linux is x86_64; arm64 images won't run.
# 2. The PyTorch CPU wheel index only ships x86_64 wheels; on arm64 the slim
#    swap silently falls back to the CUDA-bundled wheel.
# Builds via QEMU emulation when host is Apple Silicon — slower (~30-45 min)
# but produces a deployable image.
echo "🔨 Building ${IMAGE} (linux/amd64) ..."
docker build --platform=linux/amd64 -t "${IMAGE}" .

echo "📦 Saving to ${OUT} ..."
docker save "${IMAGE}" -o "${OUT}"

SIZE=$(du -sh "${OUT}" | awk '{print $1}')
echo
echo "✓ Done."
echo "  Image: ${IMAGE}"
echo "  Tarball: ${OUT} (${SIZE})"
echo
echo "Transfer to your server:"
echo "  rsync -avz --progress ${OUT} root@<server>:/root/"
echo
echo "On the server:"
echo "  docker load -i /root/${OUT}"
echo "  docker run --rm \\"
echo "    --name kb-core \\"
echo "    -p 127.0.0.1:8501:8501 \\"
echo "    --env-file /root/pichia-kb/.env \\"
echo "    -v /root/pichia-kb/data:/app/data \\"
echo "    ${IMAGE}"
