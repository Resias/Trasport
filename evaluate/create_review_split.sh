#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="${1:-/home/data/od_minute}"
TARGET_ROOT="${2:-/root/tmp/Trasport/data_splits/od_minute_review_3way}"

mkdir -p "${TARGET_ROOT}/train" "${TARGET_ROOT}/val" "${TARGET_ROOT}/test"

link_day() {
  local src_dir="$1"
  local dst_dir="$2"
  local day="$3"
  local src="${SOURCE_ROOT}/${src_dir}/OD_minute_202205${day}.npy"
  local dst="${TARGET_ROOT}/${dst_dir}/OD_minute_202205${day}.npy"

  if [[ ! -f "${src}" ]]; then
    echo "Missing source file: ${src}" >&2
    exit 1
  fi

  ln -sfn "${src}" "${dst}"
}

for day in $(seq -w 1 19); do
  link_day train train "${day}"
done

for day in $(seq -w 20 24); do
  link_day train val "${day}"
done

for day in $(seq -w 25 31); do
  link_day test test "${day}"
done

echo "Created review split at ${TARGET_ROOT}"
find "${TARGET_ROOT}" -maxdepth 2 -type l | sort
