

from amdsmi import * 

import pdb

switch = {
    amdsmi_wrapper.AMDSMI_STATUS_INVAL : "AMDSMI_STATUS_INVAL - Invalid parameters",
    amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED : "AMDSMI_STATUS_NOT_SUPPORTED - Feature not supported",
    amdsmi_wrapper.AMDSMI_STATUS_NOT_YET_IMPLEMENTED : "AMDSMI_STATUS_NOT_YET_IMPLEMENTED - Feature not yet implemented",
    amdsmi_wrapper.AMDSMI_STATUS_FAIL_LOAD_MODULE : "AMDSMI_STATUS_FAIL_LOAD_MODULE - Fail to load lib",
    amdsmi_wrapper.AMDSMI_STATUS_FAIL_LOAD_SYMBOL : "AMDSMI_STATUS_FAIL_LOAD_SYMBOL - Fail to load symbol",
    amdsmi_wrapper.AMDSMI_STATUS_DRM_ERROR : "AMDSMI_STATUS_DRM_ERROR - Error when called libdrm",
    amdsmi_wrapper.AMDSMI_STATUS_API_FAILED : "AMDSMI_STATUS_API_FAILED - API call failed",
    amdsmi_wrapper.AMDSMI_STATUS_TIMEOUT : "AMDSMI_STATUS_TIMEOUT - Timeout in API call",
    amdsmi_wrapper.AMDSMI_STATUS_RETRY : "AMDSMI_STATUS_RETRY - Retry operation",
    amdsmi_wrapper.AMDSMI_STATUS_INTERNAL_EXCEPTION : "AMDSMI_STATUS_INTERNAL_EXCEPTION -  Internal error",
    amdsmi_wrapper.AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS : "AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS - Out of bounds",
    amdsmi_wrapper.AMDSMI_STATUS_INIT_ERROR : "AMDSMI_STATUS_INIT_ERROR - Initialization error",
    amdsmi_wrapper.AMDSMI_STATUS_REFCOUNT_OVERFLOW : "AMDSMI_STATUS_REFCOUNT_OVERFLOW - Internal reference counter exceeded INT32_MAX",
    amdsmi_wrapper.AMDSMI_STATUS_BUSY : "AMDSMI_STATUS_BUSY - Device busy",
    amdsmi_wrapper.AMDSMI_STATUS_NOT_FOUND : "AMDSMI_STATUS_NOT_FOUND - Device Not found",
    amdsmi_wrapper.AMDSMI_STATUS_NOT_INIT : "AMDSMI_STATUS_NOT_INIT - Device not initialized",
    amdsmi_wrapper.AMDSMI_STATUS_NO_SLOT : "AMDSMI_STATUS_NO_SLOT - No more free slot",
    amdsmi_wrapper.AMDSMI_STATUS_DRIVER_NOT_LOADED : "AMDSMI_STATUS_DRIVER_NOT_LOADED - Driver not loaded",
    amdsmi_wrapper.AMDSMI_STATUS_NO_DATA : "AMDSMI_STATUS_NO_DATA - No data was found for given input",
    amdsmi_wrapper.AMDSMI_STATUS_INSUFFICIENT_SIZE : "AMDSMI_STATUS_INSUFFICIENT_SIZE - Insufficient size for operation",
    amdsmi_wrapper.AMDSMI_STATUS_UNEXPECTED_SIZE : "AMDSMI_STATUS_UNEXPECTED_SIZE - unexpected size of data was read",
    amdsmi_wrapper.AMDSMI_STATUS_UNEXPECTED_DATA : "AMDSMI_STATUS_UNEXPECTED_DATA - The data read or provided was unexpected",
    amdsmi_wrapper.AMDSMI_STATUS_NON_AMD_CPU : "AMDSMI_STATUS_NON_AMD_CPU - System has non-AMD CPU",
    amdsmi_wrapper.AMDSMI_NO_ENERGY_DRV : "AMD_SMI_NO_ENERGY_DRV - Energy driver not found",
    amdsmi_wrapper.AMDSMI_NO_MSR_DRV : "AMDSMI_NO_MSR_DRV - MSR driver not found",
    amdsmi_wrapper.AMDSMI_NO_HSMP_DRV : "AMD_SMI_NO_HSMP_DRV - HSMP driver not found",
    amdsmi_wrapper.AMDSMI_NO_HSMP_SUP : "AMD_SMI_NO_HSMP_SUP - HSMP not supported",
    amdsmi_wrapper.AMDSMI_NO_HSMP_MSG_SUP : "AMD_SMI_NO_HSMP_MSG_SUP - HSMP message/feature not supported",
    amdsmi_wrapper.AMDSMI_HSMP_TIMEOUT : "AMD_SMI_HSMP_TIMEOUT - HSMP message timeout",
    amdsmi_wrapper.AMDSMI_NO_DRV : "AMDSMI_NO_DRV - No Energy and HSMP driver present",
    amdsmi_wrapper.AMDSMI_FILE_NOT_FOUND : "AMDSMI_FILE_NOT_FOUND - File or directory not found",
    amdsmi_wrapper.AMDSMI_ARG_PTR_NULL : "AMDSMI_ARG_PTR_NULL - Parsed argument is invalid",
    amdsmi_wrapper.AMDSMI_STATUS_MAP_ERROR : "AMDSMI_STATUS_MAP_ERROR - The internal library error did not map to a status code",
    amdsmi_wrapper.AMDSMI_STATUS_UNKNOWN_ERROR : "AMDSMI_STATUS_UNKNOWN_ERROR - An unknown error occurred"
}


try:
    handles = amdsmi_get_processor_handles()
except amdsmi_exception.AmdSmiLibraryException as e:
    print(e.err_info)
    print(e.err_code)
    pdb.set_trace()
    print(e)