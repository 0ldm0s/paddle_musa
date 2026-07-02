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

init() {
    RED='\033[0;31m'
    BLUE='\033[0;34m'
    BOLD='\033[1m'
    NONE='\033[0m'
    SCRIPT_NAME='paddle_musa build script'
    SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    CUR_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
    PADDLE_PATH=$(cd "${CUR_DIR}/../../Paddle" && pwd)
    PADDLE_PATCHES_DIR=${CUR_DIR}/patches/paddle
}

print_usage() {
  echo -e "\n${RED}Options${NONE}:
  ${BLUE}-a/--all${NONE}: build paddlepaddle and paddle_musa
  ${BLUE}-p/--paddle${NONE}: build paddlepaddle only and install
  ${BLUE}-m/--paddle_musa${NONE}: build paddle_musa only and install
  ${BLUE}-u/--union${NONE}: build Paddle and paddle_musa in union-build single-wheel mode
  ${BLUE}-ap/--apply_paddle_hack${NONE}: apply paddle hack replacement rules only
  ${BLUE}-t/--test${NONE}: run all unit test
  ${BLUE}-s/--single_test${NONE}: run single unit test
  ${BLUE}-c/--clean${NONE}: clean paddle_musa
  ${BLUE}-h/--help${NONE}: show usage
  "
}

function copy_impl() {
  file=$1
  paddle_dst_path=$2
  echo -e "${BLUE}copy ${file} to ${PADDLE_PATH} ...${NONE}"
  cp ${CUR_DIR}/${file} ${PADDLE_PATH}/${paddle_dst_path}
  echo -e "${BLUE}copy done ...${NONE}"
}

function copy_hack_dir_except_rule_files() {
  src_dir=$1
  dst_dir=$2
  if [ ! -d "${CUR_DIR}/${src_dir}" ]; then
    return
  fi

  mkdir -p ${PADDLE_PATH}/${dst_dir}
  for src_file in ${CUR_DIR}/${src_dir}/*; do
    if [ ! -f "${src_file}" ]; then
      continue
    fi

    rel_path=${src_file#${CUR_DIR}/hack/}
    if python3 - "${CUR_DIR}/hack/hack_file_rules.json" "${rel_path}" <<'PY'
import json
import sys

mapping_file, rel_path = sys.argv[1:]
with open(mapping_file, "r", encoding="utf-8") as f:
    mapping = json.load(f)
paths = {item["path"] for item in mapping.get("files", [])}
sys.exit(0 if rel_path in paths else 1)
PY
    then
      echo -e "${BLUE}skip rule-managed hack/${rel_path}${NONE}"
      continue
    fi

    copy_impl "hack/${rel_path}" "${dst_dir}/"
  done
}

apply_hack_rules() {
  echo -e "${BLUE}Applying hack replacement rules to ${PADDLE_PATH} ...${NONE}"
  python3 ${CUR_DIR}/tools/apply_hack_rules.py --repo-root ${CUR_DIR} --source-root ${PADDLE_PATH}
  echo -e "${BLUE}Applying hack replacement rules done ...${NONE}"
}

restore_paddle_repository() {
  if ! git -C "${PADDLE_PATH}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo -e "${RED}${PADDLE_PATH} is not a git repository, cannot safely restore before applying hack rules. ${NONE}"
    exit 15
  fi

  echo -e "${BLUE}Reset Paddle repository before applying hack rules. ${NONE}"
  pushd "${PADDLE_PATH}"

  echo -e "${BLUE}Paddle status before reset:${NONE}"
  git status --short

  git reset --hard HEAD
  if [ "$?" != 0 ]; then
    echo "reset Paddle repository failed!!!!"
    exit 13
  fi

  git clean -fd
  if [ "$?" != 0 ]; then
    echo "clean Paddle repository failed!!!!"
    exit 13
  fi

  git submodule update --init --recursive
  if [ "$?" != 0 ]; then
    echo "update Paddle submodules failed!!!!"
    exit 14
  fi

  git submodule foreach --recursive 'git reset --hard HEAD'
  if [ "$?" != 0 ]; then
    echo "reset Paddle submodules failed!!!!"
    exit 14
  fi

  git submodule foreach --recursive 'git clean -fd'
  if [ "$?" != 0 ]; then
    echo "clean Paddle submodules failed!!!!"
    exit 14
  fi

  echo -e "${BLUE}Paddle status after reset:${NONE}"
  git status --short

  popd
}

apply_hack_paddle_rules() {
  echo -e "${BLUE}Applying hack rules to ${PADDLE_PATH} ...${NONE}"
  restore_paddle_repository

  apply_hack_rules
}

build_paddlepaddle() {
  pushd $PADDLE_PATH
  if [ ! -f "CMakeLists.txt" ];then
    git submodule update --init --recursive --jobs 1;get_pd_ret=$?
    if [ "$get_pd_ret" != 0 ];then
        echo "get paddlepaddle failed!!!!"
        exit 11
    fi
    apply_hack_paddle_rules
  fi 
  
  mkdir -p build
  pushd build
  
  if [ ! -d "CMakeFiles" ];then 
    cmake .. -DWITH_MKL=ON \
            -DWITH_GPU=OFF \
            -DPY_VERSION=3.10 \
            -DWITH_CINN=OFF \
            -DWITH_DISTRIBUTE=ON \
	    -DWITH_SLEEF=OFF \
            -DCMAKE_EXPORT_COMPILE_COMMANDS=on \
            -DCMAKE_CXX_FLAGS="-I/usr/local/musa/include";cmake_ret=$? 
    if [ "$cmake_ret" != 0 ];then
        echo "cmake error"
        exit 9
    fi
  fi
  
  make -j 128;make_ret=$?
  if [ "$cmake_ret" != 0 ];then
      echo "paddle make have error, ret=${make_ret}"
  fi

  if [ ! -f "python/dist/$(ls python/dist)" ]; then
      echo "paddle build failed!!!!!!!"
      exit 12
  fi
  pip uninstall paddlepaddle
  pip install python/dist/paddlepaddle*.whl --force-reinstall
  
  #post_copy_some_hack_files #TODO(moore threads): replace by implemention in cmake
  
  popd
  popd  
}

build_paddle_musa() {

  bash tools/build_standalone.sh;build_ret=$?
  if [ "$build_ret" != 0 ];then
      echo "CMake Error Found !!!"
      exit 8;
  fi

  pip uninstall -y paddle_musa
  pip install --force-reinstall build/dist/paddle_musa-*.whl

  PADDLE_MUSA_ROOT_PATH="$(cd ../../ && pwd)" python setup_ops.py install
}

build_union() {
  bash tools/build_union.sh "$@"

  local pkg_dir="${PADDLE_PATH}/build/python/dist"
  local pkg_name="${PADDLE_PYTHON_PACKAGE_NAME:-paddlepaddle-musa}"
  local pkg_prefix="${pkg_name//-/_}"
  local latest_pkg

  latest_pkg=$(ls -t "${pkg_dir}/${pkg_prefix}"*.whl 2>/dev/null | head -1)
  if [ -z "${latest_pkg}" ]; then
    latest_pkg=$(ls -t "${pkg_dir}"/paddlepaddle*.whl 2>/dev/null | head -1)
  fi
  if [ -z "${latest_pkg}" ]; then
    echo "ERROR: No Paddle wheel package found in ${pkg_dir}"
    exit 1
  fi

  echo -e "${BLUE}Installing union build wheel: ${latest_pkg}${NONE}"
  pip install "${latest_pkg}" --force-reinstall
}

run_all_ut() {
  export PYTHONPATH="$(pwd)/tests/unittests:${PYTHONPATH}"
  for file in $(ls tests/unittests/test_*.py | grep -v "test_reduce_op.py" | grep -v "test_reshape_op.py"); do
    echo "Running $file"
    python3 -m unittest "$file"
  done

  bash tools/run_ut.sh
}

run_single_ut() {
  echo "-s unimplement"
}

clean() {

  # clean paddlepaddle
  echo -e "${BLUE}begin clean paddlepaddle. ${NONE}"
  rm -rf "${PADDLE_PATH}/build"
  echo -e "${BLUE}clean paddlepaddle finished. ${NONE}"

  # clean paddle_musa
  echo -e "${BLUE}begin clean paddle_musa. ${NONE}"
  rm -rf "${CUR_DIR}/build"
  rm -rf "${CUR_DIR}/dist"
  rm -rf "${CUR_DIR}/custom_setup_ops.egg-info"
  rm -rf "${CUR_DIR}"/*.egg-info
  find "${CUR_DIR}" -type d -name "__pycache__" -prune -exec rm -rf {} +
  echo -e "${BLUE}clean paddle_musa finished. ${NONE}"

}

main() {
  init
  while [ $# -gt 0 ]; do
    case "$1" in
    -a | --all)
      build_paddlepaddle
      build_paddle_musa 
      shift
      ;;
    -ap | --apply_paddle_hack)
      apply_hack_paddle_rules
      shift
      ;;
    -p | --paddle)
      build_paddlepaddle
      shift
      ;;
    -m | --paddle_musa)
      build_paddle_musa
      shift
      ;;
    -u | --union)
      shift
      build_union "$@"
      break
      ;;
    -t | --test)
      run_all_ut
      shift
      ;;
    -s | --single_test)
      run_single_ut
      shift
      ;;
    -c | --clean)
      clean
      shift
      ;;
    -h | --help)
      print_usage
      exit
      ;;
    --)
      shift
      break
      ;;
    *)
      print_usage
      exit 0
      ;;
    esac
  done
}


main $@
