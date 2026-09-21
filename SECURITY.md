# Security Policy

## Overview

Zeus is a framework for deep learning energy measurement and optimization developed by the ML.ENERGY Initiative.
We take security seriously and appreciate your efforts to responsibly disclose your findings.

## Reporting Security Vulnerabilities

If you discover a security vulnerability in Zeus, please report it responsibly:

### For Security Issues

**Do NOT open a public GitHub issue.**
Instead, please email us at:

security@ml.energy

This will route your report to maintainers privately.

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
- **Communication**: Unix domain socket (default: `/run/zeusd/zeusd.sock`) or TCP
- **Security model**: Unprivileged applications request privileged operations through the daemon

**Daemon Security Considerations**

- Privileged API groups require the OS privileges needed by the underlying operation
  - Linux rejects root-required API groups at startup when `zeusd` is not root
  - Windows does not perform an equivalent startup elevation check; insufficient NVML privileges surface as request failures
- Unix-domain-socket access is controlled with filesystem permissions
  - 666 to allow all users
  - Create a dedicated user group, change the socket group to that, and set permissions to 660
- TCP supports JWT authentication with per-user API-group scopes
- Unauthenticated TCP is limited to loopback addresses by default; non-loopback listeners require a signing key unless the operator explicitly opts in with `--allow-unauthenticated-tcp`
- JWT authenticates requests but does not encrypt HTTP transport; use TLS or an encrypted tunnel across untrusted networks
- Exposes GPU configuration changes and privileged CPU measurements to any client authorized to access the corresponding API group

## Additional Resources

- [Project Homepage](https://ml.energy/zeus)
- [Documentation](https://ml.energy/zeus)
- [GitHub Repository](https://github.com/ml-energy/zeus)
- [Contributing Guidelines](CONTRIBUTING.md)

---

**Last Updated**: September 2025
**Security Policy Version**: 1.0
