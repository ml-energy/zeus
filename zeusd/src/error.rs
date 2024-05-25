use nvml_wrapper::error::NvmlError;
use actix_web::ResponseError;

#[derive(thiserror::Error, Debug)]
pub enum ZeusdError {
    #[error("NVML error: {0}")]
    NvmlError(#[from] NvmlError),
}

impl ResponseError for ZeusdError {
    fn status_code(&self) -> actix_web::http::StatusCode {
        match self {
            ZeusdError::NvmlError(e) => {
                match e {
                    NvmlError::NoPermission => actix_web::http::StatusCode::FORBIDDEN,
                    NvmlError::InvalidArg => actix_web::http::StatusCode::BAD_REQUEST,
                    _ => actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
                }
            }
        }
    }
}
