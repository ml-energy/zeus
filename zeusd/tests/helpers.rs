use once_cell::sync::Lazy;
use zeusd::startup::init_tracing;

static TRACING: Lazy<()> = Lazy::new(|| {
    if std::env::var("TEST_LOG").is_ok() {
        init_tracing(std::io::stdout).expect("Failed to initialize tracing");
    } else {
        init_tracing(std::io::sink).expect("Failed to initialize tracing");
    };
});

pub async fn spawn_app() {
    Lazy::force(&TRACING);
}
