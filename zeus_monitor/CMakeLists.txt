# Copyright (C) 2022 Jae-Won Chung <jwnchung@umich.edu>
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

#---------------------------------------------------------------------------------------
# Define Project-Wide Settings
#---------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.17)

# Define the Project Name and Description
project(zeus LANGUAGES CXX DESCRIPTION "Zeus: An Energy Optimization Framework for DNN Training")

# Set required C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS FALSE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
# Generate the compile_commands.json file
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
# Set the Relative Path Configurations
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# Configure the visibility of symbols in targets
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

#---------------------------------------------------------------------------------------
# Define CMake Module Imports
#---------------------------------------------------------------------------------------
# For feature_summary
include(FeatureSummary)

#---------------------------------------------------------------------------------------
# Find packages
#---------------------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

#---------------------------------------------------------------------------------------
# Print summary
#---------------------------------------------------------------------------------------
feature_summary(INCLUDE_QUIET_PACKAGES FATAL_ON_MISSING_REQUIRED_PACKAGES WHAT ALL)

#---------------------------------------------------------------------------------------
# Define Targets
#---------------------------------------------------------------------------------------
add_executable(zeus_monitor
  main.cpp
)
target_link_libraries(zeus_monitor
  nvidia-ml
  Threads::Threads
)
target_include_directories(zeus_monitor PRIVATE
  zemo
  ${CUDAToolkit_INCLUDE_DIRS}
)
install(TARGETS zeus_monitor DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
