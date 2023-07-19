// Copyright (C) 2023 Jae-Won Chung <jwnchung@umich.edu>
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <chrono>
#include <iostream>
#include <thread>
#include <signal.h>
#include <chrono>
#include <ctime>
#include "date.h"

#include "zemo.hpp"

// Add prefix for logging
std::string prefix() {
  return date::format("%F %T [ZeusMonitor] ", floor<std::chrono::milliseconds>(std::chrono::system_clock::now()));
}

// Catch CTRL-C and stop the monitor early
void endMonitoring(int signal) {
  stop_nvml_thread();
  std::cout << prefix() << "Caught signal " << signal << ", end monitoring." << std::endl;
  exit(signal);
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cout << "Usage: " << argv[0] << " LOGFILE DURATION SLEEP_MS [GPU_IDX]" << std::endl
              << "  Set DURATION to 0 to run indefinitely." << std::endl
              << "  If SLEEP_MS is 0, the monitor won't call sleep at all." << std::endl;
    return 0;
  }

  std::string logfilePath = argv[1];
  int64_t duration = std::atoi(argv[2]);
  int sleep_ms = std::atoi(argv[3]);
  int gpu_index = 0;
  if (argc > 4) {
    gpu_index = std::atoi(argv[4]);
  }
  std::cout << prefix() << "Monitor started." << std::endl;
  if (duration == 0) {
    std::cout << prefix() << "Running indefinitely. ";
  } else {
    std::cout << prefix() << "Running for " << duration << "s. ";
  }
  if (sleep_ms == 0) {
    std::cout << "High-speed polling mode. " << std::endl;
  } else {
    std::cout << "Sleeping " << sleep_ms << "ms after each poll. " << std::endl;
  }
  std::cout << prefix() << "Logfile path: " << logfilePath << std::endl;

  signal(SIGINT, endMonitoring);

  start_nvml_thread(logfilePath, /* wait */ true, /* sleep_ms */ sleep_ms, /* gpu_index */ gpu_index);
  if (duration == 0) {
    // Sleep indefinitely.
    std::this_thread::sleep_until(
      std::chrono::system_clock::now() + std::chrono::hours(std::numeric_limits<int>::max())
    );
  } else {
    std::this_thread::sleep_for(std::chrono::seconds(duration));
  }
  stop_nvml_thread();
}
