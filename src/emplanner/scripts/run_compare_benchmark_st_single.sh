#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_COMPARE_SCRIPT="${SCRIPT_DIR}/run_compare_benchmark.sh"

if [[ ! -x "${RUN_COMPARE_SCRIPT}" ]]; then
  echo "run_compare_benchmark.sh not found or not executable: ${RUN_COMPARE_SCRIPT}" >&2
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "usage:" >&2
  echo "  ${0} /abs/path/to/obstacles.csv [scenario_name:=my_case] [other ros args...]" >&2
  exit 1
fi

SCENARIO_SET=0
for arg in "$@"; do
  case "${arg}" in
    scenario_name:=*)
      SCENARIO_SET=1
      ;;
  esac
done

ARGS=(
  launch_file:=compare_benchmark_st_single.launch
  --right-panel-st
)

if [[ "${SCENARIO_SET}" -eq 0 ]]; then
  ARGS+=(scenario_name:=st_single_case)
fi

ARGS+=("$@")

exec "${RUN_COMPARE_SCRIPT}" "${ARGS[@]}"
