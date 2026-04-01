#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SEARCH_ROOT="${1:-${REPO_ROOT}/logs/hydra/multirun}"
POLL_SECONDS="${POLL_SECONDS:-60}"
STATE_DIR="${STATE_DIR:-${REPO_ROOT}/.wandb-sync-state}"
STATE_FILE="${STATE_DIR}/last_synced_mtimes.tsv"

mkdir -p "${STATE_DIR}"
touch "${STATE_FILE}"

find_wandb_cli() {
  if command -v wandb >/dev/null 2>&1; then
    command -v wandb
    return 0
  fi

  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/wandb" ]]; then
    printf '%s\n' "${VIRTUAL_ENV}/bin/wandb"
    return 0
  fi

  if [[ -x "${REPO_ROOT}/.venv/bin/wandb" ]]; then
    printf '%s\n' "${REPO_ROOT}/.venv/bin/wandb"
    return 0
  fi

  return 1
}

WANDB_BIN="$(find_wandb_cli || true)"
if [[ -z "${WANDB_BIN}" ]]; then
  echo "Could not find the wandb CLI. Activate your venv or install wandb first." >&2
  exit 1
fi

declare -A LAST_SYNCED_MTIME=()
while IFS=$'\t' read -r run_dir last_mtime; do
  [[ -n "${run_dir}" ]] || continue
  LAST_SYNCED_MTIME["${run_dir}"]="${last_mtime}"
done < "${STATE_FILE}"

save_state() {
  : > "${STATE_FILE}"
  for run_dir in "${!LAST_SYNCED_MTIME[@]}"; do
    printf '%s\t%s\n' "${run_dir}" "${LAST_SYNCED_MTIME[${run_dir}]}" >> "${STATE_FILE}"
  done
}

latest_mtime() {
  local run_dir="$1"
  find "${run_dir}" -type f -print0 2>/dev/null \
    | xargs -0 stat -f '%m' 2>/dev/null \
    | sort -n \
    | tail -n 1
}

sync_run() {
  local run_dir="$1"
  echo "[$(date '+%F %T')] syncing ${run_dir}"
  "${WANDB_BIN}" sync "${run_dir}"
}

trap 'echo; echo "Stopping wandb sync loop."; exit 0' INT TERM

echo "Watching offline wandb runs under: ${SEARCH_ROOT}"
echo "Using wandb CLI: ${WANDB_BIN}"
echo "Polling every ${POLL_SECONDS}s"

while true; do
  while IFS= read -r run_dir; do
    [[ -d "${run_dir}" ]] || continue

    current_mtime="$(latest_mtime "${run_dir}")"
    [[ -n "${current_mtime}" ]] || continue

    previous_mtime="${LAST_SYNCED_MTIME[${run_dir}]:-}"
    if [[ "${current_mtime}" != "${previous_mtime}" ]]; then
      if sync_run "${run_dir}"; then
        LAST_SYNCED_MTIME["${run_dir}"]="${current_mtime}"
        save_state
      else
        echo "[$(date '+%F %T')] sync failed for ${run_dir}" >&2
      fi
    fi
  done < <(find "${SEARCH_ROOT}" -type d -path '*/wandb/offline-run-*' | sort)

  sleep "${POLL_SECONDS}"
done
