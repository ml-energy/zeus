# Security Policy

## Overview

Zeus is a framework for deep learning energy measurement and optimization developed by the ML.ENERGY Initiative.
We take security seriously and appreciate your efforts to responsibly disclose your findings.

## Reporting Security Vulnerabilities

If you discover a security vulnerability in Zeus, please report it responsibly:

### For Security Issues

**Do NOT open a public GitHub issue.** Instead, please email us at:

jwnchung@umich.edu (Jae-Won Chung, Project Lead)

### What to Include

- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Any suggested fixes (if available)

### Response Timeline

- **Initial Response**: Within 48 hours
- **Assessment**: Within 7 days
- **Resolution**: Timeline varies based on severity and complexity

## Security Considerations

### Privileged Operations

Zeus requires elevated privileges only for specific operations:

**Root privileges required for**:
1. **CPU energy measurement via Intel RAPL**: Due to [CVE-2020-8694](https://www.cve.org/CVERecord?id=CVE-2020-8694), reading RAPL energy counters through `/sys/class/powercap/intel-rapl` requires root access.
2. **GPU power limit/frequency adjustment**: The `SYS_ADMIN` Linux security capability (similar to root privileges) is needed to change GPU power limits or frequencies via NVML. Similar is true for AMD GPUs using AMD SMI.

**Normal user privileges sufficient for**:
- GPU energy monitoring (reading power/energy consumption)
- Most Zeus functionality including basic monitoring and measurement

### Zeus Daemon (`zeusd`)

To avoid running applications as root, Zeus provides a privileged daemon.

- **Purpose**: Runs with admin privileges and exposes API endpoints wrapping privileged RAPL counters and NVML methods
- **Implementation**: Written in Rust for low latency
- **Communication**: Unix domain socket (default: `/var/run/zeusd.sock`) or TCP
- **Security model**: Unprivileged applications request privileged operations through the daemon

**Daemon Security Considerations**
- Runs as root by design
- No authentication mechanism on its own
- Socket permissions configurable (default 666 allows any user access), so Linux file permissions should be used to restrict access
- Exposes GPU configuration changes to any process with socket access

## Additional Resources

- [Project Homepage](https://ml.energy/zeus)
- [Documentation](https://ml.energy/zeus)
- [GitHub Repository](https://github.com/ml-energy/zeus)
- [Contributing Guidelines](CONTRIBUTING.md)

---

**Last Updated**: September 2025
**Security Policy Version**: 1.0
