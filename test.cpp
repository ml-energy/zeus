// Try to write it in C/C++ and call the real API.

// g++ -I/opt/rocm-6.0.2/include/amd_smi -L/opt/rocm-6.0.2/lib test.cpp

#include "amdsmi.h"
#include <iostream>

int main(){
    amdsmi_status_t res = amdsmi_init(0);

    res = amdsmi_shut_down();
}