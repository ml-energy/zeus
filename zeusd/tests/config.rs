use clap::Parser;
use zeusd::config::Cli;

#[test]
fn power_poll_frequencies_reject_zero_and_sub_microsecond_periods() {
    for option in ["--gpu-power-poll-hz", "--cpu-power-poll-hz"] {
        for value in ["0", "1000001"] {
            let result = Cli::try_parse_from(["zeusd", "serve", option, value]);
            assert!(result.is_err(), "expected {option}={value} to be rejected");
        }
    }
}

#[test]
fn power_poll_frequencies_accept_supported_boundaries() {
    for option in ["--gpu-power-poll-hz", "--cpu-power-poll-hz"] {
        for value in ["1", "1000000"] {
            let result = Cli::try_parse_from(["zeusd", "serve", option, value]);
            assert!(result.is_ok(), "expected {option}={value} to be accepted");
        }
    }
}
