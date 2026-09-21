use clap::Parser;
use zeusd::config::Cli;

#[test]
fn power_poll_frequencies_remain_unrestricted_u32_values() {
    for option in ["--gpu-power-poll-hz", "--cpu-power-poll-hz"] {
        for value in ["0", "1", "1000", "1001", "1000001", "4294967295"] {
            let result = Cli::try_parse_from(["zeusd", "serve", option, value]);
            assert!(result.is_ok(), "expected {option}={value} to be accepted");
        }
    }
}
