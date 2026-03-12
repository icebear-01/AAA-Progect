#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PKG_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
RUN_BENCHMARK="${SCRIPT_DIR}/run_compare_benchmark.sh"

if [[ ! -x "${RUN_BENCHMARK}" ]]; then
  echo "run_compare_benchmark.sh not found or not executable: ${RUN_BENCHMARK}" >&2
  exit 1
fi

TARGET="${1:-all}"
if [[ $# -gt 0 ]]; then
  shift
fi

declare -a ANGLES=()
case "${TARGET}" in
  all)
    ANGLES=(0 30 60 90)
    ;;
  0|30|60|90)
    ANGLES=("${TARGET}")
    ;;
  *)
    echo "usage: $(basename "$0") [0|30|60|90|all] [extra roslaunch args...]" >&2
    exit 2
    ;;
esac

RESULT_ROOT="${PKG_DIR}/benchmark_results/turn_angle_experiments"
mkdir -p "${RESULT_ROOT}"
AGGREGATE_CSV="${RESULT_ROOT}/angle_experiments.csv"
OBSTACLE_AGGREGATE_CSV="${RESULT_ROOT}/angle_obstacles.csv"

echo "scenario_name,turn_angle_case,turn_angle_deg,straight_turn_x,straight_turn_arc_length,qp_running_normally,planner_total_ms,dp_sampling_ms,qp_optimization_ms,speed_planning_ms,obstacle_count,summary_path,obstacles_csv" > "${AGGREGATE_CSV}"
echo "scenario_name,turn_angle_deg,id,center_x,center_y,length,width,yaw,x_vel,y_vel" > "${OBSTACLE_AGGREGATE_CSV}"

get_summary_value() {
  local file="$1"
  local key="$2"
  awk -F': ' -v k="${key}" '$1 == k {print $2; exit}' "${file}"
}

for angle in "${ANGLES[@]}"; do
  scenario_name=$(printf "turn_%02d_deg_arc" "${angle}")
  "${RUN_BENCHMARK}" \
    "scenario_name:=${scenario_name}" \
    "turn_angle_case:=${angle}" \
    "$@"

  summary_file="${PKG_DIR}/benchmark_results/${scenario_name}/summary.txt"
  obstacles_file="${PKG_DIR}/benchmark_results/${scenario_name}/obstacles.csv"

  if [[ ! -f "${summary_file}" ]]; then
    echo "summary file not found after run: ${summary_file}" >&2
    exit 3
  fi

  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "${scenario_name}" \
    "$(get_summary_value "${summary_file}" "turn_angle_case")" \
    "${angle}" \
    "$(get_summary_value "${summary_file}" "straight_turn_x")" \
    "$(get_summary_value "${summary_file}" "straight_turn_arc_length")" \
    "$(get_summary_value "${summary_file}" "qp_running_normally")" \
    "$(get_summary_value "${summary_file}" "planner_total_ms")" \
    "$(get_summary_value "${summary_file}" "dp_sampling_ms")" \
    "$(get_summary_value "${summary_file}" "qp_optimization_ms")" \
    "$(get_summary_value "${summary_file}" "speed_planning_ms")" \
    "$(get_summary_value "${summary_file}" "obstacle_count")" \
    "${summary_file}" \
    "${obstacles_file}" >> "${AGGREGATE_CSV}"

  if [[ -f "${obstacles_file}" ]]; then
    awk -F',' -v scenario="${scenario_name}" -v angle="${angle}" 'NR > 1 {
      printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
             scenario, angle, $1, $2, $3, $4, $5, $6, $7, $8
    }' "${obstacles_file}" >> "${OBSTACLE_AGGREGATE_CSV}"
  fi
done

echo "aggregate_csv=${AGGREGATE_CSV}"
echo "aggregate_obstacles_csv=${OBSTACLE_AGGREGATE_CSV}"
