#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PKG_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
WORKSPACE_DIR=$(cd "${PKG_DIR}/../../../.." && pwd)

if [[ -f /opt/ros/noetic/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/noetic/setup.bash
fi

if [[ -f "${WORKSPACE_DIR}/devel/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "${WORKSPACE_DIR}/devel/setup.bash"
fi

SCENARIO_NAME="curve_turn30_feasible"
PLOT_MODE="--paper"
EXPORT_PDF="--export-pdf"
LAUNCH_FILE="compare_benchmark_paper.launch"
SHOW_GRID="--show-grid"
RIGHT_PANEL="curvature"
LAUNCH_ARGS=()
SCENARIO_ARG_SET=0
TURN_ANGLE_CASE=""
STRAIGHT_TURN_ANGLE_ARG_SET=0
OBSTACLE_CSV_ARG_SET=0
LAUNCH_PID=""
BENCHMARK_BIN="${WORKSPACE_DIR}/devel/lib/emplanner/emplanner_compare_benchmark"
BENCHMARK_NODE="/emplanner_compare_benchmark"
RUN_MARKER=""

for arg in "$@"; do
  case "${arg}" in
    scenario_name:=*)
      SCENARIO_NAME="${arg#scenario_name:=}"
      LAUNCH_ARGS+=("${arg}")
      SCENARIO_ARG_SET=1
      ;;
    turn_angle_case:=*)
      TURN_ANGLE_CASE="${arg#turn_angle_case:=}"
      LAUNCH_ARGS+=("${arg}")
      ;;
    straight_turn_angle_deg:=*)
      STRAIGHT_TURN_ANGLE_ARG_SET=1
      LAUNCH_ARGS+=("${arg}")
      ;;
    obstacle_csv:=*)
      OBSTACLE_CSV_ARG_SET=1
      LAUNCH_ARGS+=("${arg}")
      ;;
    launch_file:=*)
      LAUNCH_FILE="${arg#launch_file:=}"
      ;;
    --right-panel-st)
      RIGHT_PANEL="st"
      ;;
    --right-panel-curvature)
      RIGHT_PANEL="curvature"
      ;;
    --no-paper)
      PLOT_MODE=""
      EXPORT_PDF=""
      ;;
    --show-grid)
      SHOW_GRID="--show-grid"
      ;;
    --hide-grid)
      SHOW_GRID=""
      ;;
    *.csv)
      if [[ "${OBSTACLE_CSV_ARG_SET}" -eq 0 && -f "${arg}" ]]; then
        LAUNCH_ARGS+=("obstacle_csv:=${arg}")
        OBSTACLE_CSV_ARG_SET=1
      else
        LAUNCH_ARGS+=("${arg}")
      fi
      ;;
    *)
      LAUNCH_ARGS+=("${arg}")
      ;;
  esac
done

cleanup_roslaunch() {
  if [[ -n "${LAUNCH_PID}" ]] && kill -0 "${LAUNCH_PID}" 2>/dev/null; then
    kill -INT "${LAUNCH_PID}" 2>/dev/null || true
    wait "${LAUNCH_PID}" 2>/dev/null || true
  fi
}

cleanup_marker() {
  if [[ -n "${RUN_MARKER}" && -e "${RUN_MARKER}" ]]; then
    rm -f "${RUN_MARKER}"
  fi
}

cleanup_all() {
  cleanup_roslaunch
  cleanup_marker
}

trap cleanup_all EXIT INT TERM

if [[ -n "${TURN_ANGLE_CASE}" && "${STRAIGHT_TURN_ANGLE_ARG_SET}" -eq 0 ]]; then
  case "${TURN_ANGLE_CASE}" in
    0|00|straight)
      LAUNCH_ARGS+=("straight_turn_angle_deg:=0")
      ;;
    30|60|90)
      LAUNCH_ARGS+=("straight_turn_angle_deg:=${TURN_ANGLE_CASE}")
      ;;
  esac
fi

if [[ "${SCENARIO_ARG_SET}" -eq 0 && -n "${TURN_ANGLE_CASE}" ]]; then
  case "${TURN_ANGLE_CASE}" in
    0|00|straight)
      SCENARIO_NAME="turn_00_deg_arc"
      ;;
    30|60|90)
      SCENARIO_NAME=$(printf "turn_%02d_deg_arc" "${TURN_ANGLE_CASE}")
      ;;
  esac
fi

if [[ "${SCENARIO_ARG_SET}" -eq 0 ]]; then
  LAUNCH_ARGS+=("scenario_name:=${SCENARIO_NAME}")
fi

if [[ "${LAUNCH_FILE}" == "compare_benchmark_st_crossing.launch" && "${RIGHT_PANEL}" == "curvature" ]]; then
  RIGHT_PANEL="st"
fi

RESULT_DIR="${PKG_DIR}/benchmark_results/${SCENARIO_NAME}"
SUMMARY_PATH="${RESULT_DIR}/summary.txt"
RUN_MARKER=$(mktemp)

pushd "${WORKSPACE_DIR}" >/dev/null
roslaunch emplanner "${LAUNCH_FILE}" "${LAUNCH_ARGS[@]}" &
LAUNCH_PID=$!
popd >/dev/null

for _ in $(seq 1 600); do
  if [[ -s "${SUMMARY_PATH}" && "${SUMMARY_PATH}" -nt "${RUN_MARKER}" ]]; then
    break
  fi
  if ! kill -0 "${LAUNCH_PID}" 2>/dev/null; then
    wait "${LAUNCH_PID}" || true
    LAUNCH_PID=""
    break
  fi
  sleep 0.2
done

cleanup_roslaunch
LAUNCH_PID=""

if [[ ! -d "${RESULT_DIR}" ]]; then
  echo "result directory not found: ${RESULT_DIR}" >&2
  exit 1
fi

if [[ ! -s "${SUMMARY_PATH}" || ! "${SUMMARY_PATH}" -nt "${RUN_MARKER}" ]]; then
  echo "benchmark summary was not updated for this run: ${SUMMARY_PATH}" >&2
  exit 1
fi

echo "benchmark finished, generating plots..."

PLOT_CMD=(python3 "${PKG_DIR}/scripts/plot_compare.py" --input-dir "${RESULT_DIR}")
if [[ -n "${PLOT_MODE}" ]]; then
  PLOT_CMD+=("${PLOT_MODE}")
fi
if [[ -n "${EXPORT_PDF}" ]]; then
  PLOT_CMD+=("${EXPORT_PDF}")
fi
if [[ -n "${SHOW_GRID}" ]]; then
  PLOT_CMD+=("${SHOW_GRID}")
fi
PLOT_CMD+=(--right-panel "${RIGHT_PANEL}")
"${PLOT_CMD[@]}"

if [[ -f "${RESULT_DIR}/speed_profile.csv" ]]; then
  echo "generating ST plot..."
  ST_PLOT_CMD=(python3 "${PKG_DIR}/scripts/plot_st_graph.py" --input-dir "${RESULT_DIR}")
  if [[ -n "${PLOT_MODE}" ]]; then
    ST_PLOT_CMD+=("${PLOT_MODE}")
  fi
  if [[ -n "${EXPORT_PDF}" ]]; then
    ST_PLOT_CMD+=("${EXPORT_PDF}")
  fi
  "${ST_PLOT_CMD[@]}"
fi

echo "benchmark_result_dir=${RESULT_DIR}"
