#!/bin/bash
# Copyright (c) 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

PYTHON_VERSION=${PYTHON_VERSION:-$(python3 -V 2>&1 | awk '{print $2}')}
WITH_CINN=${WITH_CINN:-OFF}
WITH_TESTING=${WITH_TESTING:-OFF}
WITH_MKL=${WITH_MKL:-ON}
WITH_MCCL=${WITH_MCCL:-ON}
ON_INFER=${ON_INFER:-ON}
PADDLE_PYTHON_PACKAGE_NAME=${PADDLE_PYTHON_PACKAGE_NAME:-paddlepaddle-musa}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
MUSA_SOURCE_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PADDLE_SOURCE_DIR=$(cd "${MUSA_SOURCE_DIR}/../../Paddle" && pwd)
PADDLE_BUILD_DIR="${PADDLE_SOURCE_DIR}/build"
PLATFORM_ID=$(uname -i)

apply_hack_rules() {
  python3 "${MUSA_SOURCE_DIR}/tools/apply_hack_rules.py" \
    --repo-root "${MUSA_SOURCE_DIR}" \
    --source-root "${PADDLE_SOURCE_DIR}"
}

if [[ "${1:-}" == "--apply-hack-rules" ]]; then
  apply_hack_rules
  exit 0
fi

mkdir -p "${PADDLE_BUILD_DIR}"

PADDLE_CMAKE_ARGS=(
  "-DPY_VERSION=${PYTHON_VERSION}"
  "-DWITH_GPU=OFF"
  "-DWITH_MKL=${WITH_MKL}"
  "-DWITH_DISTRIBUTE=ON"
  "-DWITH_CUSTOM_DEVICE=ON"
  "-DWITH_CUSTOM_DEVICE_SUB_BUILD=ON"
  "-DCUSTOM_DEVICE_ENABLE_CUDA_LANGUAGE=OFF"
  "-DCUSTOM_DEVICE_SOURCE_DIR=${MUSA_SOURCE_DIR}"
  "-DWITH_CINN=${WITH_CINN}"
  "-DWITH_SLEEF=OFF"
  "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
  "-DPADDLE_PYTHON_PACKAGE_NAME=${PADDLE_PYTHON_PACKAGE_NAME}"
)

CUSTOM_DEVICE_CMAKE_ARGS=(
  "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}"
  "-DPADDLE_SOURCE_DIR=${PADDLE_SOURCE_DIR}"
  "-DWITH_MUSA=ON"
  "-DWITH_MCCL=${WITH_MCCL}"
  "-DWITH_TESTING=${WITH_TESTING}"
  "-DWITH_MKL=${WITH_MKL}"
  "-DON_INFER=${ON_INFER}"
  "-DWITH_CINN=${WITH_CINN}"
  "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
  "-DPY_VERSION=${PYTHON_VERSION}"
)

if [[ "${PLATFORM_ID}" == "aarch64" ]]; then
  CUSTOM_DEVICE_CMAKE_ARGS+=("-DWITH_ARM=ON")
else
  CUSTOM_DEVICE_CMAKE_ARGS+=("-DWITH_ARM=OFF")
fi

CUSTOM_DEVICE_CMAKE_ARGS_STR=$(IFS=';'; echo "${CUSTOM_DEVICE_CMAKE_ARGS[*]}")
PADDLE_CMAKE_ARGS+=("-DCUSTOM_DEVICE_CMAKE_ARGS=${CUSTOM_DEVICE_CMAKE_ARGS_STR}")

pushd "${PADDLE_BUILD_DIR}"
cmake -G Ninja "${PADDLE_CMAKE_ARGS[@]}" "${PADDLE_SOURCE_DIR}" 2>&1 | tee compile.log
[[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "Error: CMake configuration failed!"; exit 1; }

if [[ "${PLATFORM_ID}" == "aarch64" ]]; then
  env TARGET=ARMV8 ninja -j$(nproc) 2>&1 | tee -a compile.log
else
  ninja -j$(nproc) 2>&1 | tee -a compile.log
fi
[[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "Error: Paddle MUSA union build failed!"; exit 1; }
popd

PKG_DIR="${PADDLE_BUILD_DIR}/python/dist"
latest_pkg=$(ls -t "${PKG_DIR}" 2>/dev/null | grep "${PADDLE_PYTHON_PACKAGE_NAME}" | head -1)
if [[ -z "${latest_pkg}" ]]; then
  latest_pkg=$(ls -t "${PKG_DIR}" 2>/dev/null | grep "paddlepaddle" | head -1)
fi
if [[ -z "${latest_pkg}" ]]; then
  echo "ERROR: No Paddle package found in ${PKG_DIR}"
  exit 1
fi

echo "Union build completed: ${PKG_DIR}/${latest_pkg}"
