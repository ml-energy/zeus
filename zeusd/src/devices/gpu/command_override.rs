//! External command overrides for privileged GPU control operations.

use std::collections::HashMap;
use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Context;
use regex::Regex;
use serde::Deserialize;

use super::GpuManager;
use crate::error::ZeusdError;

const DEFAULT_TIMEOUT_S: u64 = 60;
const POLL_INTERVAL: Duration = Duration::from_millis(10);
const OUTPUT_COLLECTION_GRACE: Duration = Duration::from_secs(1);

/// A GPU control operation that can be replaced by external commands.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GpuControlOperation {
    SetPersistenceMode,
    SetPowerLimit,
    SetGpuLockedClocks,
    ResetGpuLockedClocks,
    SetMemLockedClocks,
    ResetMemLockedClocks,
    ResetLockedClocks,
}

impl GpuControlOperation {
    fn name(self) -> &'static str {
        match self {
            Self::SetPersistenceMode => "set_persistence_mode",
            Self::SetPowerLimit => "set_power_limit",
            Self::SetGpuLockedClocks => "set_gpu_locked_clocks",
            Self::ResetGpuLockedClocks => "reset_gpu_locked_clocks",
            Self::SetMemLockedClocks => "set_mem_locked_clocks",
            Self::ResetMemLockedClocks => "reset_mem_locked_clocks",
            Self::ResetLockedClocks => "reset_locked_clocks",
        }
    }

    fn allowed_placeholders(self) -> &'static [&'static str] {
        match self {
            Self::SetPersistenceMode => &["gpu_id", "enabled"],
            Self::SetPowerLimit => &["gpu_id", "power_limit_mw", "power_limit_w"],
            Self::SetGpuLockedClocks | Self::SetMemLockedClocks => {
                &["gpu_id", "min_clock_mhz", "max_clock_mhz"]
            }
            Self::ResetGpuLockedClocks | Self::ResetMemLockedClocks | Self::ResetLockedClocks => {
                &["gpu_id"]
            }
        }
    }
}

#[derive(Debug)]
struct OverrideSpec {
    commands: Vec<Vec<String>>,
    error_pattern: Option<Regex>,
    timeout: Duration,
}

/// Parsed and validated external command overrides.
#[derive(Debug)]
pub struct GpuCommandOverrides {
    specs: HashMap<GpuControlOperation, OverrideSpec>,
}

impl GpuCommandOverrides {
    /// Read and validate command overrides from a TOML file.
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read GPU command override file '{path}'"))?;
        Self::parse(&contents)
            .with_context(|| format!("Failed to parse GPU command override file '{path}'"))
    }

    fn parse(contents: &str) -> anyhow::Result<Self> {
        let raw: RawOverrides = toml::from_str(contents)?;
        let entries = [
            (
                GpuControlOperation::SetPersistenceMode,
                raw.set_persistence_mode,
            ),
            (GpuControlOperation::SetPowerLimit, raw.set_power_limit),
            (
                GpuControlOperation::SetGpuLockedClocks,
                raw.set_gpu_locked_clocks,
            ),
            (
                GpuControlOperation::ResetGpuLockedClocks,
                raw.reset_gpu_locked_clocks,
            ),
            (
                GpuControlOperation::SetMemLockedClocks,
                raw.set_mem_locked_clocks,
            ),
            (
                GpuControlOperation::ResetMemLockedClocks,
                raw.reset_mem_locked_clocks,
            ),
            (
                GpuControlOperation::ResetLockedClocks,
                raw.reset_locked_clocks,
            ),
        ];
        let mut specs = HashMap::new();
        for (operation, raw_spec) in entries {
            if let Some(raw_spec) = raw_spec {
                specs.insert(operation, validate_spec(operation, raw_spec)?);
            }
        }
        Ok(Self { specs })
    }

    /// Return the configured operation names in declaration order.
    pub fn overridden_operation_names(&self) -> Vec<&'static str> {
        const OPERATIONS: [GpuControlOperation; 7] = [
            GpuControlOperation::SetPersistenceMode,
            GpuControlOperation::SetPowerLimit,
            GpuControlOperation::SetGpuLockedClocks,
            GpuControlOperation::ResetGpuLockedClocks,
            GpuControlOperation::SetMemLockedClocks,
            GpuControlOperation::ResetMemLockedClocks,
            GpuControlOperation::ResetLockedClocks,
        ];
        OPERATIONS
            .iter()
            .filter(|operation| self.specs.contains_key(operation))
            .map(|operation| operation.name())
            .collect()
    }

    /// Return whether at least one operation is overridden.
    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }

    fn get(&self, operation: GpuControlOperation) -> Option<&OverrideSpec> {
        self.specs.get(&operation)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawOverrides {
    set_persistence_mode: Option<RawOverrideSpec>,
    set_power_limit: Option<RawOverrideSpec>,
    set_gpu_locked_clocks: Option<RawOverrideSpec>,
    reset_gpu_locked_clocks: Option<RawOverrideSpec>,
    set_mem_locked_clocks: Option<RawOverrideSpec>,
    reset_mem_locked_clocks: Option<RawOverrideSpec>,
    reset_locked_clocks: Option<RawOverrideSpec>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawOverrideSpec {
    commands: Vec<String>,
    error_pattern: Option<String>,
    timeout_s: Option<u64>,
}

fn validate_spec(
    operation: GpuControlOperation,
    raw: RawOverrideSpec,
) -> anyhow::Result<OverrideSpec> {
    if raw.commands.is_empty() {
        anyhow::bail!(
            "Operation '{}' has an empty commands array",
            operation.name()
        );
    }

    let error_pattern = raw
        .error_pattern
        .map(|pattern| {
            Regex::new(&pattern).with_context(|| {
                format!(
                    "Operation '{}' has invalid error_pattern '{pattern}'",
                    operation.name()
                )
            })
        })
        .transpose()?;

    let mut commands = Vec::with_capacity(raw.commands.len());
    for command in raw.commands {
        let tokens = shell_words::split(&command).with_context(|| {
            format!(
                "Operation '{}' has a command that cannot be split: {command}",
                operation.name()
            )
        })?;
        if tokens.is_empty() {
            anyhow::bail!(
                "Operation '{}' has a command that splits to zero tokens",
                operation.name()
            );
        }
        for token in &tokens {
            validate_placeholders(operation, token)?;
        }
        commands.push(tokens);
    }

    Ok(OverrideSpec {
        commands,
        error_pattern,
        timeout: Duration::from_secs(raw.timeout_s.unwrap_or(DEFAULT_TIMEOUT_S)),
    })
}

fn validate_placeholders(operation: GpuControlOperation, token: &str) -> anyhow::Result<()> {
    let bytes = token.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'{' => {
                let name_start = index + 1;
                let mut end = name_start;
                while end < bytes.len() && bytes[end] != b'{' && bytes[end] != b'}' {
                    end += 1;
                }
                if end == bytes.len() || bytes[end] != b'}' || end == name_start {
                    anyhow::bail!(
                        "Operation '{}' has an invalid placeholder in token '{}'",
                        operation.name(),
                        token
                    );
                }
                let name = &token[name_start..end];
                if !operation.allowed_placeholders().contains(&name) {
                    anyhow::bail!(
                        "Operation '{}' has unknown placeholder '{{{}}}' in token '{}'",
                        operation.name(),
                        name,
                        token
                    );
                }
                index = end + 1;
            }
            b'}' => {
                anyhow::bail!(
                    "Operation '{}' has an invalid placeholder in token '{}'",
                    operation.name(),
                    token
                );
            }
            _ => index += 1,
        }
    }
    Ok(())
}

#[derive(Default)]
struct PlaceholderValues {
    gpu_id: usize,
    enabled: Option<bool>,
    power_limit_mw: Option<u32>,
    min_clock_mhz: Option<u32>,
    max_clock_mhz: Option<u32>,
}

impl PlaceholderValues {
    fn substitute(&self, token: &str) -> String {
        let mut result = token.replace("{gpu_id}", &self.gpu_id.to_string());
        if let Some(enabled) = self.enabled {
            result = result.replace("{enabled}", if enabled { "1" } else { "0" });
        }
        if let Some(power_limit_mw) = self.power_limit_mw {
            result = result.replace("{power_limit_mw}", &power_limit_mw.to_string());
            result = result.replace("{power_limit_w}", &(power_limit_mw / 1000).to_string());
        }
        if let Some(min_clock_mhz) = self.min_clock_mhz {
            result = result.replace("{min_clock_mhz}", &min_clock_mhz.to_string());
        }
        if let Some(max_clock_mhz) = self.max_clock_mhz {
            result = result.replace("{max_clock_mhz}", &max_clock_mhz.to_string());
        }
        result
    }
}

fn execute_override(spec: &OverrideSpec, values: &PlaceholderValues) -> Result<(), ZeusdError> {
    for command in &spec.commands {
        let command: Vec<String> = command
            .iter()
            .map(|token| values.substitute(token))
            .collect();
        execute_command(&command, spec)?;
    }
    Ok(())
}

fn execute_command(command: &[String], spec: &OverrideSpec) -> Result<(), ZeusdError> {
    let command_line = shell_words::join(command);
    let child_result = Command::new(&command[0])
        .args(&command[1..])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let mut child = match child_result {
        Ok(child) => child,
        Err(error) => {
            let reason = format!("could not be spawned: {error}");
            tracing::info!(command = %command_line, outcome = %reason, "GPU override command completed");
            return Err(command_error(&command_line, &reason, "", ""));
        }
    };

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| command_error(&command_line, "stdout capture was unavailable", "", ""))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| command_error(&command_line, "stderr capture was unavailable", "", ""))?;
    let stdout_rx = spawn_pipe_reader(stdout);
    let stderr_rx = spawn_pipe_reader(stderr);
    let started = Instant::now();

    let outcome = loop {
        match child.try_wait() {
            Ok(Some(status)) => break CommandOutcome::Exited(status),
            Ok(None) if started.elapsed() >= spec.timeout => {
                let kill_result = child.kill();
                let wait_result = child.wait();
                break CommandOutcome::TimedOut {
                    kill_result,
                    wait_result,
                };
            }
            Ok(None) => thread::sleep(POLL_INTERVAL),
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                break CommandOutcome::WaitError(error);
            }
        }
    };

    let stdout = collect_pipe(stdout_rx, "stdout");
    let stderr = collect_pipe(stderr_rx, "stderr");

    let failure = match outcome {
        CommandOutcome::Exited(status) if !status.success() => Some(describe_exit(status)),
        CommandOutcome::Exited(_) => match (&spec.error_pattern, &stdout, &stderr) {
            (None, _, _) => None,
            (Some(pattern), Ok(stdout), Ok(stderr)) => {
                let mut output = String::with_capacity(stdout.len() + stderr.len());
                output.push_str(stdout);
                output.push_str(stderr);
                pattern.is_match(&output).then(|| {
                    format!(
                        "succeeded but its output matched error_pattern '{}'",
                        pattern.as_str()
                    )
                })
            }
            (Some(_), _, _) => Some(
                "succeeded but its output could not be collected, so error_pattern could not \
                 be checked"
                    .to_string(),
            ),
        },
        CommandOutcome::TimedOut {
            kill_result,
            wait_result,
        } => {
            if let Err(error) = kill_result {
                Some(format!(
                    "timed out after {}s and could not be killed: {error}",
                    spec.timeout.as_secs()
                ))
            } else if let Err(error) = wait_result {
                Some(format!(
                    "timed out after {}s and could not be reaped: {error}",
                    spec.timeout.as_secs()
                ))
            } else {
                Some(format!("timed out after {}s", spec.timeout.as_secs()))
            }
        }
        CommandOutcome::WaitError(error) => Some(format!("could not be waited on: {error}")),
    };

    if let Some(reason) = failure {
        let stdout = stdout.unwrap_or_else(|reason| format!("<{reason}>"));
        let stderr = stderr.unwrap_or_else(|reason| format!("<{reason}>"));
        tracing::info!(command = %command_line, outcome = %reason, "GPU override command completed");
        return Err(command_error(&command_line, &reason, &stdout, &stderr));
    }
    tracing::info!(command = %command_line, outcome = "success", "GPU override command completed");
    Ok(())
}

fn describe_exit(status: std::process::ExitStatus) -> String {
    match status.code() {
        Some(code) => format!("exited with code {code}"),
        None => format!("was terminated abnormally ({status})"),
    }
}

enum CommandOutcome {
    Exited(std::process::ExitStatus),
    TimedOut {
        kill_result: std::io::Result<()>,
        wait_result: std::io::Result<std::process::ExitStatus>,
    },
    WaitError(std::io::Error),
}

fn drain_pipe<R: Read>(mut pipe: R) -> std::io::Result<Vec<u8>> {
    let mut output = Vec::new();
    pipe.read_to_end(&mut output)?;
    Ok(output)
}

fn spawn_pipe_reader<R: Read + Send + 'static>(
    pipe: R,
) -> std::sync::mpsc::Receiver<std::io::Result<Vec<u8>>> {
    let (tx, rx) = std::sync::mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(drain_pipe(pipe));
    });
    rx
}

/// Wait for a pipe reader with a bounded grace period instead of joining it.
/// The reader only returns once every write end of the pipe is closed, and a
/// process spawned by the command (e.g., sudo's child, which survives the kill
/// of sudo itself on timeout) can hold the pipe open indefinitely; joining
/// would then block this GPU's management task forever.
fn collect_pipe(
    rx: std::sync::mpsc::Receiver<std::io::Result<Vec<u8>>>,
    stream: &str,
) -> Result<String, String> {
    match rx.recv_timeout(OUTPUT_COLLECTION_GRACE) {
        Ok(Ok(bytes)) => Ok(String::from_utf8_lossy(&bytes).into_owned()),
        Ok(Err(error)) => Err(format!("{stream} could not be read: {error}")),
        Err(_) => Err(format!(
            "{stream} was still open {}s after the command ended, likely held by a \
             leftover child process",
            OUTPUT_COLLECTION_GRACE.as_secs()
        )),
    }
}

fn command_error(command_line: &str, reason: &str, stdout: &str, stderr: &str) -> ZeusdError {
    ZeusdError::CommandOverrideError(format!(
        "command '{command_line}' {reason}; stdout: {stdout}; stderr: {stderr}"
    ))
}

/// A GPU manager that executes configured control operations as commands.
pub struct OverriddenGpu<T: GpuManager> {
    gpu_id: usize,
    inner: T,
    overrides: Arc<GpuCommandOverrides>,
}

impl<T: GpuManager> OverriddenGpu<T> {
    /// Wrap a native GPU manager with command overrides.
    pub fn new(gpu_id: usize, inner: T, overrides: Arc<GpuCommandOverrides>) -> Self {
        Self {
            gpu_id,
            inner,
            overrides,
        }
    }

    fn execute(
        &self,
        operation: GpuControlOperation,
        values: PlaceholderValues,
    ) -> Option<Result<(), ZeusdError>> {
        self.overrides
            .get(operation)
            .map(|spec| execute_override(spec, &values))
    }
}

impl<T: GpuManager> GpuManager for OverriddenGpu<T> {
    fn device_count() -> Result<u32, ZeusdError>
    where
        Self: Sized,
    {
        T::device_count()
    }

    fn set_persistence_mode(&mut self, enabled: bool) -> Result<(), ZeusdError> {
        self.execute(
            GpuControlOperation::SetPersistenceMode,
            PlaceholderValues {
                gpu_id: self.gpu_id,
                enabled: Some(enabled),
                ..Default::default()
            },
        )
        .unwrap_or_else(|| self.inner.set_persistence_mode(enabled))
    }

    fn get_persistence_mode(&mut self) -> Result<bool, ZeusdError> {
        self.inner.get_persistence_mode()
    }

    fn set_power_management_limit(&mut self, power_limit_mw: u32) -> Result<(), ZeusdError> {
        self.execute(
            GpuControlOperation::SetPowerLimit,
            PlaceholderValues {
                gpu_id: self.gpu_id,
                power_limit_mw: Some(power_limit_mw),
                ..Default::default()
            },
        )
        .unwrap_or_else(|| self.inner.set_power_management_limit(power_limit_mw))
    }

    fn get_power_management_limit(&mut self) -> Result<u32, ZeusdError> {
        self.inner.get_power_management_limit()
    }

    fn get_power_management_limit_constraints(&mut self) -> Result<(u32, u32), ZeusdError> {
        self.inner.get_power_management_limit_constraints()
    }

    fn set_gpu_locked_clocks(
        &mut self,
        min_clock_mhz: u32,
        max_clock_mhz: u32,
    ) -> Result<(), ZeusdError> {
        self.execute(
            GpuControlOperation::SetGpuLockedClocks,
            PlaceholderValues {
                gpu_id: self.gpu_id,
                min_clock_mhz: Some(min_clock_mhz),
                max_clock_mhz: Some(max_clock_mhz),
                ..Default::default()
            },
        )
        .unwrap_or_else(|| {
            self.inner
                .set_gpu_locked_clocks(min_clock_mhz, max_clock_mhz)
        })
    }

    fn reset_gpu_locked_clocks(&mut self) -> Result<(), ZeusdError> {
        self.execute(
            GpuControlOperation::ResetGpuLockedClocks,
            PlaceholderValues {
                gpu_id: self.gpu_id,
                ..Default::default()
            },
        )
        .unwrap_or_else(|| self.inner.reset_gpu_locked_clocks())
    }

    fn set_mem_locked_clocks(
        &mut self,
        min_clock_mhz: u32,
        max_clock_mhz: u32,
    ) -> Result<(), ZeusdError> {
        self.execute(
            GpuControlOperation::SetMemLockedClocks,
            PlaceholderValues {
                gpu_id: self.gpu_id,
                min_clock_mhz: Some(min_clock_mhz),
                max_clock_mhz: Some(max_clock_mhz),
                ..Default::default()
            },
        )
        .unwrap_or_else(|| {
            self.inner
                .set_mem_locked_clocks(min_clock_mhz, max_clock_mhz)
        })
    }

    fn reset_mem_locked_clocks(&mut self) -> Result<(), ZeusdError> {
        self.execute(
            GpuControlOperation::ResetMemLockedClocks,
            PlaceholderValues {
                gpu_id: self.gpu_id,
                ..Default::default()
            },
        )
        .unwrap_or_else(|| self.inner.reset_mem_locked_clocks())
    }

    fn reset_locked_clocks(&mut self) -> Result<(), ZeusdError> {
        self.execute(
            GpuControlOperation::ResetLockedClocks,
            PlaceholderValues {
                gpu_id: self.gpu_id,
                ..Default::default()
            },
        )
        .unwrap_or_else(|| self.inner.reset_locked_clocks())
    }

    fn get_instant_power_mw(&mut self) -> Result<u32, ZeusdError> {
        self.inner.get_instant_power_mw()
    }

    fn get_total_energy_consumption(&mut self) -> Result<u64, ZeusdError> {
        self.inner.get_total_energy_consumption()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    fn parse_single(operation: &str, body: &str) -> GpuCommandOverrides {
        GpuCommandOverrides::parse(&format!("[{operation}]\n{body}")).unwrap()
    }

    #[test]
    fn loads_valid_config_with_all_operations_and_fields() {
        let contents = r#"
[set_persistence_mode]
commands = ["control --gpu={gpu_id} --enabled={enabled}"]
error_pattern = "(?i)error"
timeout_s = 1

[set_power_limit]
commands = ["control {gpu_id} {power_limit_mw} {power_limit_w}"]
error_pattern = "failed"
timeout_s = 2

[set_gpu_locked_clocks]
commands = ["control {gpu_id} {min_clock_mhz} {max_clock_mhz}"]
error_pattern = "failed"
timeout_s = 3

[reset_gpu_locked_clocks]
commands = ["control {gpu_id}"]
error_pattern = "failed"
timeout_s = 4

[set_mem_locked_clocks]
commands = ["control {gpu_id} {min_clock_mhz} {max_clock_mhz}"]
error_pattern = "failed"
timeout_s = 5

[reset_mem_locked_clocks]
commands = ["control {gpu_id}"]
error_pattern = "failed"
timeout_s = 6

[reset_locked_clocks]
commands = ["control {gpu_id}"]
error_pattern = "failed"
timeout_s = 7
"#;
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("overrides.toml");
        std::fs::write(&path, contents).unwrap();
        let config = GpuCommandOverrides::load(path.to_str().unwrap()).unwrap();

        assert_eq!(
            config.overridden_operation_names(),
            vec![
                "set_persistence_mode",
                "set_power_limit",
                "set_gpu_locked_clocks",
                "reset_gpu_locked_clocks",
                "set_mem_locked_clocks",
                "reset_mem_locked_clocks",
                "reset_locked_clocks",
            ]
        );
        assert_eq!(
            config
                .get(GpuControlOperation::ResetLockedClocks)
                .unwrap()
                .timeout,
            Duration::from_secs(7)
        );
    }

    #[test]
    fn rejects_unknown_operation() {
        let error =
            GpuCommandOverrides::parse("[set_fan_speed]\ncommands = [\"control {gpu_id}\"]\n")
                .unwrap_err();
        assert!(error.to_string().contains("unknown field `set_fan_speed`"));
    }

    #[test]
    fn rejects_unknown_operation_field() {
        let error = GpuCommandOverrides::parse(
            "[set_power_limit]\ncommands = [\"control\"]\nworking_dir = \"/tmp\"\n",
        )
        .unwrap_err();
        assert!(error.to_string().contains("unknown field `working_dir`"));
    }

    #[test]
    fn rejects_empty_commands() {
        let error = GpuCommandOverrides::parse("[set_power_limit]\ncommands = []\n").unwrap_err();
        assert!(error.to_string().contains("empty commands array"));
    }

    #[test]
    fn rejects_invalid_regex() {
        let error = GpuCommandOverrides::parse(
            "[set_power_limit]\ncommands = [\"control\"]\nerror_pattern = \"[\"\n",
        )
        .unwrap_err();
        assert!(error.to_string().contains("set_power_limit"));
        assert!(error.to_string().contains("invalid error_pattern"));
    }

    #[test]
    fn rejects_unknown_placeholder_with_operation_and_token() {
        let error = GpuCommandOverrides::parse(
            "[set_power_limit]\ncommands = [\"control --value={max_clock_mhz}\"]\n",
        )
        .unwrap_err();
        let message = error.to_string();
        assert!(message.contains("set_power_limit"));
        assert!(message.contains("--value={max_clock_mhz}"));
    }

    #[test]
    fn rejects_stray_or_unmatched_braces() {
        for token in ["value}", "{gpu_id", "{{gpu_id}"] {
            let input = format!("[reset_locked_clocks]\ncommands = [\"control {token}\"]\n");
            let error = GpuCommandOverrides::parse(&input).unwrap_err();
            let message = error.to_string();
            assert!(message.contains("reset_locked_clocks"));
            assert!(message.contains(token));
        }
    }

    #[test]
    fn rejects_commands_that_cannot_be_split_or_have_no_tokens() {
        let split_error =
            GpuCommandOverrides::parse("[reset_locked_clocks]\ncommands = [\"control '\"]\n")
                .unwrap_err();
        assert!(split_error.to_string().contains("cannot be split"));

        let empty_error =
            GpuCommandOverrides::parse("[reset_locked_clocks]\ncommands = [\"\"]\n").unwrap_err();
        assert!(empty_error.to_string().contains("zero tokens"));
    }

    #[test]
    fn substitutes_all_operation_values() {
        let power = PlaceholderValues {
            gpu_id: 3,
            power_limit_mw: Some(253_999),
            ..Default::default()
        };
        assert_eq!(
            power.substitute("--gpu={gpu_id}:{power_limit_mw}:{power_limit_w}"),
            "--gpu=3:253999:253"
        );

        let clocks = PlaceholderValues {
            gpu_id: 4,
            min_clock_mhz: Some(500),
            max_clock_mhz: Some(1_700),
            ..Default::default()
        };
        assert_eq!(
            clocks.substitute("{gpu_id}:{min_clock_mhz}:{max_clock_mhz}"),
            "4:500:1700"
        );

        let enabled = PlaceholderValues {
            gpu_id: 5,
            enabled: Some(true),
            ..Default::default()
        };
        assert_eq!(enabled.substitute("{gpu_id}:{enabled}"), "5:1");
        let disabled = PlaceholderValues {
            enabled: Some(false),
            ..Default::default()
        };
        assert_eq!(disabled.substitute("{enabled}"), "0");
    }

    #[cfg(unix)]
    fn write_script(directory: &std::path::Path, name: &str, body: &str) -> String {
        use std::os::unix::fs::PermissionsExt;

        let path = directory.join(name);
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        let mut permissions = std::fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&path, permissions).unwrap();
        path.to_string_lossy().into_owned()
    }

    #[cfg(unix)]
    fn spec_for_command(command: &str, extra: &str) -> GpuCommandOverrides {
        GpuCommandOverrides::parse(&format!(
            "[reset_locked_clocks]\ncommands = [{}]\n{extra}",
            toml::Value::String(command.to_string())
        ))
        .unwrap()
    }

    #[cfg(unix)]
    fn execute_reset(config: &GpuCommandOverrides) -> Result<(), ZeusdError> {
        execute_override(
            config.get(GpuControlOperation::ResetLockedClocks).unwrap(),
            &PlaceholderValues::default(),
        )
    }

    /// Serialize tests that write scripts and spawn them. A concurrent fork in
    /// another test inherits a script file descriptor that is still open for
    /// writing, and executing that script then fails with ETXTBSY.
    #[cfg(unix)]
    fn subprocess_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|poison| poison.into_inner())
    }

    #[cfg(unix)]
    #[test]
    fn executor_succeeds_for_zero_exit_without_pattern_match() {
        let _guard = subprocess_lock();
        let directory = tempfile::tempdir().unwrap();
        let script = write_script(directory.path(), "success.sh", "echo success");
        let config = spec_for_command(&script, "");
        execute_reset(&config).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn executor_rejects_nonzero_exit_and_captures_output() {
        let _guard = subprocess_lock();
        let directory = tempfile::tempdir().unwrap();
        let script = write_script(
            directory.path(),
            "failure.sh",
            "echo standard-output\necho standard-error >&2\nexit 7",
        );
        let error = execute_reset(&spec_for_command(&script, "")).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("exited with code 7"), "{message}");
        assert!(message.contains("standard-output"));
        assert!(message.contains("standard-error"));
        assert!(message.contains(&script));
    }

    #[cfg(unix)]
    #[test]
    fn executor_rejects_error_pattern_match_after_zero_exit() {
        let _guard = subprocess_lock();
        let directory = tempfile::tempdir().unwrap();
        let script = write_script(directory.path(), "pattern.sh", "echo ERROR >&2");
        let config = spec_for_command(&script, "error_pattern = \"(?i)error\"\n");
        let error = execute_reset(&config).unwrap_err();
        assert!(error.to_string().contains("matched error_pattern"));
        assert!(error.to_string().contains("ERROR"));
    }

    #[cfg(unix)]
    #[test]
    fn executor_kills_and_reaps_timed_out_command() {
        let _guard = subprocess_lock();
        let directory = tempfile::tempdir().unwrap();
        let script = write_script(directory.path(), "timeout.sh", "exec sleep 30");
        let config = spec_for_command(&script, "timeout_s = 1\n");
        let started = Instant::now();
        let error = execute_reset(&config).unwrap_err();
        assert!(error.to_string().contains("timed out after 1s"));
        assert!(started.elapsed() < Duration::from_secs(10));
    }

    #[cfg(unix)]
    #[test]
    fn executor_does_not_block_on_pipes_held_by_leftover_children() {
        let _guard = subprocess_lock();
        let directory = tempfile::tempdir().unwrap();
        let script = write_script(directory.path(), "linger.sh", "sleep 30 &\nexit 0");

        let started = Instant::now();
        execute_reset(&spec_for_command(&script, "")).unwrap();
        assert!(started.elapsed() < Duration::from_secs(10));

        let started = Instant::now();
        let error = execute_reset(&spec_for_command(
            &script,
            "error_pattern = \"(?i)error\"\n",
        ))
        .unwrap_err();
        assert!(started.elapsed() < Duration::from_secs(10));
        assert!(error
            .to_string()
            .contains("output could not be collected, so error_pattern could not be checked"));
    }

    #[cfg(unix)]
    #[test]
    fn executor_timeout_is_not_extended_by_surviving_grandchildren() {
        let _guard = subprocess_lock();
        let directory = tempfile::tempdir().unwrap();
        let script = write_script(directory.path(), "orphan.sh", "sleep 30 &\nexec sleep 30");
        let config = spec_for_command(&script, "timeout_s = 1\n");
        let started = Instant::now();
        let error = execute_reset(&config).unwrap_err();
        assert!(error.to_string().contains("timed out after 1s"));
        assert!(started.elapsed() < Duration::from_secs(10));
    }

    #[cfg(unix)]
    #[test]
    fn executor_rejects_spawn_failure() {
        let _guard = subprocess_lock();
        let missing = "/nonexistent/zeusd-command-override-test";
        let error = execute_reset(&spec_for_command(missing, "")).unwrap_err();
        assert!(error.to_string().contains("could not be spawned"));
        assert!(error.to_string().contains(missing));
    }

    #[cfg(unix)]
    #[test]
    fn executor_stops_after_first_failure() {
        let _guard = subprocess_lock();
        let directory = tempfile::tempdir().unwrap();
        let marker = directory.path().join("marker");
        let failure = write_script(directory.path(), "first.sh", "exit 1");
        let second = write_script(
            directory.path(),
            "second.sh",
            &format!("touch {}", marker.display()),
        );
        let config = GpuCommandOverrides::parse(&format!(
            "[reset_locked_clocks]\ncommands = [{}, {}]\n",
            toml::Value::String(failure),
            toml::Value::String(second)
        ))
        .unwrap();
        execute_reset(&config).unwrap_err();
        assert!(!marker.exists());
    }

    struct MockGpu {
        power_limit_calls: Arc<AtomicUsize>,
        gpu_clock_calls: Arc<AtomicUsize>,
    }

    impl GpuManager for MockGpu {
        fn device_count() -> Result<u32, ZeusdError> {
            Ok(1)
        }

        fn set_persistence_mode(&mut self, _enabled: bool) -> Result<(), ZeusdError> {
            Ok(())
        }

        fn get_persistence_mode(&mut self) -> Result<bool, ZeusdError> {
            Ok(true)
        }

        fn set_power_management_limit(&mut self, _power_limit: u32) -> Result<(), ZeusdError> {
            self.power_limit_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn get_power_management_limit(&mut self) -> Result<u32, ZeusdError> {
            Ok(200_000)
        }

        fn get_power_management_limit_constraints(&mut self) -> Result<(u32, u32), ZeusdError> {
            Ok((100_000, 300_000))
        }

        fn set_gpu_locked_clocks(
            &mut self,
            _min_clock_mhz: u32,
            _max_clock_mhz: u32,
        ) -> Result<(), ZeusdError> {
            self.gpu_clock_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn reset_gpu_locked_clocks(&mut self) -> Result<(), ZeusdError> {
            Ok(())
        }

        fn set_mem_locked_clocks(
            &mut self,
            _min_clock_mhz: u32,
            _max_clock_mhz: u32,
        ) -> Result<(), ZeusdError> {
            Ok(())
        }

        fn reset_mem_locked_clocks(&mut self) -> Result<(), ZeusdError> {
            Ok(())
        }

        fn reset_locked_clocks(&mut self) -> Result<(), ZeusdError> {
            Ok(())
        }

        fn get_instant_power_mw(&mut self) -> Result<u32, ZeusdError> {
            Ok(150_000)
        }

        fn get_total_energy_consumption(&mut self) -> Result<u64, ZeusdError> {
            Ok(1_000)
        }
    }

    #[cfg(unix)]
    #[test]
    fn wrapper_executes_overrides_and_delegates_other_operations() {
        let _guard = subprocess_lock();
        let directory = tempfile::tempdir().unwrap();
        let marker = directory.path().join("marker");
        let script = write_script(
            directory.path(),
            "marker.sh",
            &format!("echo \"$1:$2\" > {}", marker.display()),
        );
        let config = GpuCommandOverrides::parse(&format!(
            "[set_power_limit]\ncommands = [{}]\n",
            toml::Value::String(format!("{script} {{gpu_id}} {{power_limit_w}}"))
        ))
        .unwrap();
        let power_limit_calls = Arc::new(AtomicUsize::new(0));
        let gpu_clock_calls = Arc::new(AtomicUsize::new(0));
        let inner = MockGpu {
            power_limit_calls: power_limit_calls.clone(),
            gpu_clock_calls: gpu_clock_calls.clone(),
        };
        let mut gpu = OverriddenGpu::new(7, inner, Arc::new(config));

        gpu.set_power_management_limit(251_999).unwrap();
        assert_eq!(std::fs::read_to_string(marker).unwrap().trim(), "7:251");
        assert_eq!(power_limit_calls.load(Ordering::SeqCst), 0);

        gpu.set_gpu_locked_clocks(500, 1_500).unwrap();
        assert_eq!(gpu_clock_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn defaults_timeout_to_sixty_seconds() {
        let config = parse_single("reset_locked_clocks", "commands = [\"control\"]\n");
        assert_eq!(
            config
                .get(GpuControlOperation::ResetLockedClocks)
                .unwrap()
                .timeout,
            Duration::from_secs(60)
        );
    }
}
