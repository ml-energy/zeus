#ifndef ENERGYPROF_H
#define ENERGYPROF_H

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

#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <filesystem>
#include <thread>

#include <nvml.h>
#include "date.h"

// Flags to enable/disable polling for specific stats
#define POLL_POWER 1
// #define POLL_TEMP 1
// #define POLL_SMFREQ 1
// #define POLL_MEM 1


using namespace date;

// Error checking.
#define die(s) do {                                                    \
    std::cout << "ERROR: " << s << " in " << __FILE__ << ':'           \
              << __LINE__ << "\nAborting.." << std::endl;              \
    exit(-1);                                                          \
} while (0)


#define checkNVML(status) do {                                         \
    std::stringstream _err;                                            \
    if (status != NVML_SUCCESS) {                                      \
      _err << "NVML failure (" << nvmlErrorString(status) << ')';      \
      die(_err.str());                                                 \
    }                                                                  \
} while (0)

// Logging
void ts_log(std::ofstream &file, const char *message) {
  file << std::chrono::system_clock::now() << "," << message << std::endl;
}

void ts_log(std::ofstream &file, std::stringstream &message) {
  file << std::chrono::system_clock::now() << "," << message.rdbuf() << std::endl;
}

void ts_log(std::ofstream &file, std::initializer_list<const char *> messages) {
  file << std::chrono::system_clock::now() << ",";
  for (auto message : messages) file << ' ' << message;
  file << std::endl;
}

void ts_log(std::ofstream &file, unsigned int val) {
  file << std::chrono::system_clock::now() << "," << val << std::endl;
}

void writeLogCsvHeader(std::ofstream &file) {
  file << "Time";
  #ifdef POLL_POWER
    file << ",Power";
  #endif
  #ifdef POLL_TEMP
    file << ",Temp";
  #endif
  #ifdef POLL_SMFREQ
    file << ",SMFreq";
  #endif
  #ifdef POLL_MEM
    file << ",Mem";
  #endif
  file << std::endl;
}

// NVML power monitoring
volatile bool nvml_running = false;
volatile bool do_nvml_poll = true;
std::thread nvml_thread;

// Poll power/smfreq/... and write to logfile
void nvmlPollStatsAndLog(std::ofstream &file, nvmlDevice_t &device) {
  std::stringstream nvmlStatsString;

  // Poll instantaneous power consumption
  #ifdef POLL_POWER
  {
    unsigned int power;
    checkNVML( nvmlDeviceGetPowerUsage(device, &power) );
    nvmlStatsString << power;
  }
  #endif

  #ifdef POLL_TEMP
  {
    unsigned int temp;
    checkNVML( nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp) );
    nvmlStatsString << "," << temp;
  }
  #endif

  #ifdef POLL_SMFREQ
  {
    unsigned int smClockMhz;
    checkNVML( nvmlDeviceGetClock(device, NVML_CLOCK_SM, NVML_CLOCK_ID_CURRENT, &smClockMhz) );
    nvmlStatsString << "," << smClockMhz;
  }
  #endif

  #ifdef POLL_MEM
  {
    nvmlMemory_t memoryInfo;
    checkNVML( nvmlDeviceGetMemoryInfo(device, &memoryInfo) );
    nvmlStatsString << "," << memoryInfo.used;
  }
  #endif

  ts_log(file, nvmlStatsString);
}

void nvml_poll(std::string logfile, int sleep_ms, int gpu_index) {
  // Initialize NVML and get the given gpu_index's handle
  nvmlDevice_t device;
  checkNVML( nvmlInit_v2() );
  checkNVML( nvmlDeviceGetHandleByIndex_v2(gpu_index, &device) );

  std::ofstream file {logfile, std::ofstream::out};

  // Write CSV header
  writeLogCsvHeader(file);

  // Poll NVML.
  nvml_running = true;
  if (sleep_ms > 0) {
    std::chrono::milliseconds ms{sleep_ms};
    while (do_nvml_poll) {
      nvmlPollStatsAndLog(file, device);
      std::this_thread::sleep_for(ms);
    }
  } else {
    while (do_nvml_poll) {
      nvmlPollStatsAndLog(file, device);
    }
  }
  nvml_running = false;

  // Shutdown NVML.
  checkNVML( nvmlShutdown() );

  // Flush and close log file.
  file.flush();
  file.close();
}

void start_nvml_thread(std::string logfile, bool wait, int sleep_ms, int gpu_index) {
  nvml_thread = std::thread(nvml_poll, logfile, sleep_ms, gpu_index);
  do_nvml_poll = true;
  if (wait) {
    while (!nvml_running) ;
  }
}

void stop_nvml_thread() {
  do_nvml_poll = false;
  nvml_thread.join();
}

#endif /* ENERGYPROF_H */
