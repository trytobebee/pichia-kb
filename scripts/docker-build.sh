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

echo "🔨 Building ${IMAGE} ..."
docker build -t "${IMAGE}" .

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
