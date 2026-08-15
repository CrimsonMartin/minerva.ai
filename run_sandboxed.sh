#!/usr/bin/env bash
# Run minerva inside a bubblewrap sandbox.
#
# Filesystem: read-only system dirs, writable ONLY this project directory.
# Network: shared with the host (bwrap network isolation is all-or-nothing,
# and the agent needs localhost for LM Studio plus HTTPS for PubMed).
#
# Usage: ./run_sandboxed.sh research "ferroptosis in cancer therapy" --mode depth
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

exec bwrap \
  --unshare-all \
  --share-net \
  --die-with-parent \
  --ro-bind /usr /usr \
  --ro-bind /etc/resolv.conf /etc/resolv.conf \
  --ro-bind /etc/ssl /etc/ssl \
  --ro-bind-try /etc/ca-certificates /etc/ca-certificates \
  --ro-bind-try /lib /lib \
  --ro-bind-try /lib64 /lib64 \
  --ro-bind-try /bin /bin \
  --ro-bind-try /sbin /sbin \
  --proc /proc \
  --dev /dev \
  --tmpfs /tmp \
  --bind "$PROJECT_DIR" "$PROJECT_DIR" \
  --chdir "$PROJECT_DIR" \
  --setenv HOME "$PROJECT_DIR" \
  --setenv PYTHONPATH "$PROJECT_DIR" \
  python3 -m minerva "$@"
