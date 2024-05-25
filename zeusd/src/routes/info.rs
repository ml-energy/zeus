#[actix_web::get("/info")]
pub async fn info() -> &'static str {
    "Hello, Actix!"
}
