# Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

include(ExternalProject)

# update eigen to the commit id f612df27 on 03/16/2021
set(EIGEN_PREFIX_DIR ${THIRD_PARTY_PATH}/eigen3)
set(EIGEN_SOURCE_DIR ${THIRD_PARTY_PATH}/eigen3/src/extern_eigen3)
set(EIGEN_TAG f612df273689a19d25b45ca4f8269463207c4fee)
set(SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/eigen3)

if(CMAKE_COMPILER_IS_GNUCC)
  # See: [Why calling some `git` commands before `patch`?]
  set(EIGEN_PATCH_COMMAND
      git checkout -- . && git checkout ${EIGEN_TAG} && ${Python_EXECUTABLE}
      ${CMAKE_SOURCE_DIR}/tools/apply_hack_rules.py --source-root
      ${PADDLE_SOURCE_DIR} --path-prefix third_party/eigen3/)
endif()

set(EIGEN_INCLUDE_DIR ${SOURCE_DIR})
include_directories(${EIGEN_INCLUDE_DIR})

ExternalProject_Add(
  extern_eigen3
  ${EXTERNAL_PROJECT_LOG_ARGS}
  SOURCE_DIR ${SOURCE_DIR}
  PREFIX ${EIGEN_PREFIX_DIR}
  UPDATE_COMMAND ""
  PATCH_COMMAND ${EIGEN_PATCH_COMMAND}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND "")

add_library(eigen3 INTERFACE)

add_dependencies(eigen3 extern_eigen3)

# sw not support thread_local semantic
if(WITH_SW)
  add_definitions(-DEIGEN_AVOID_THREAD_LOCAL)
endif()
