//! JWT-based authentication and authorization.
//!
//! When a signing key is configured (`--signing-key-path`), the daemon
//! requires a valid `Authorization: Bearer <JWT>` header on all requests
//! except `/discover` and `/time`.
//!
//! Tokens are issued via `zeusd token issue` and carry user identity and
//! scopes (which API groups the bearer may access).

use std::future::{ready, Future, Ready};
use std::pin::Pin;
use std::sync::Arc;

use actix_web::body::EitherBody;
use actix_web::dev::{Service, ServiceRequest, ServiceResponse, Transform};
use actix_web::http::header::AUTHORIZATION;
use actix_web::{web, HttpMessage, HttpResponse};
use jsonwebtoken::{DecodingKey, Validation};
use serde::{Deserialize, Serialize};

use crate::config::ApiGroup;
use crate::startup::EnabledGroups;

/// JWT claims embedded in every Zeusd token.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Claims {
    /// User identity (for audit logging).
    pub sub: String,
    /// API groups the bearer is allowed to access.
    pub scopes: Vec<ApiGroup>,
    /// Expiry as Unix timestamp. Standard JWT registered claim.
    /// Omitted (and validation disabled) for tokens with no expiry.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exp: Option<usize>,
}

/// Wrapper so the decoding key can be shared via `web::Data`.
#[derive(Clone)]
pub struct SigningKeyData(pub Arc<DecodingKey>);

/// Encode a JWT with the given claims.
pub fn issue_token(
    key_bytes: &[u8],
    sub: &str,
    scopes: Vec<ApiGroup>,
    expires_at: Option<usize>,
) -> Result<String, jsonwebtoken::errors::Error> {
    let claims = Claims {
        sub: sub.to_string(),
        scopes,
        exp: expires_at,
    };
    let encoding_key = jsonwebtoken::EncodingKey::from_secret(key_bytes);
    let header = jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256);
    jsonwebtoken::encode(&header, &claims, &encoding_key)
}

/// Determine the required `ApiGroup` for a given request path.
///
/// Returns `None` for paths that are always unauthenticated (`/discover`,
/// `/time`).
fn required_scope(path: &str) -> Option<ApiGroup> {
    if path == "/discover" || path == "/time" {
        return None;
    }
    if path.starts_with("/gpu/set_") || path.starts_with("/gpu/reset_") {
        return Some(ApiGroup::GpuControl);
    }
    if path.starts_with("/gpu/") {
        return Some(ApiGroup::GpuRead);
    }
    if path.starts_with("/cpu/") {
        return Some(ApiGroup::CpuRead);
    }
    // Unknown path. Require no specific scope but still require a valid
    // token when auth is enabled.
    None
}

// ---------------------------------------------------------------------------
// Actix-web middleware
// ---------------------------------------------------------------------------

/// Middleware factory that enforces JWT authentication when a signing key
/// is present in application data.
pub struct AuthMiddleware;

impl<S, B> Transform<S, ServiceRequest> for AuthMiddleware
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = actix_web::Error> + 'static,
    B: 'static,
{
    type Response = ServiceResponse<EitherBody<B>>;
    type Error = actix_web::Error;
    type Transform = AuthMiddlewareService<S>;
    type InitError = ();
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ready(Ok(AuthMiddlewareService { service }))
    }
}

pub struct AuthMiddlewareService<S> {
    service: S,
}

impl<S, B> Service<ServiceRequest> for AuthMiddlewareService<S>
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = actix_web::Error> + 'static,
    B: 'static,
{
    type Response = ServiceResponse<EitherBody<B>>;
    type Error = actix_web::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>>>>;

    fn poll_ready(
        &self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        self.service.poll_ready(cx)
    }

    fn call(&self, req: ServiceRequest) -> Self::Future {
        let path = req.path().to_string();

        // If the path maps to a disabled API group, return 404 immediately
        // regardless of auth state.
        if let Some(scope) = required_scope(&path) {
            if let Some(enabled) = req.app_data::<web::Data<EnabledGroups>>() {
                if !enabled.0.contains(&scope) {
                    let resp = HttpResponse::NotFound().json(serde_json::json!({
                        "error": format!("API group '{}' is not enabled on this server.", scope)
                    }));
                    return Box::pin(
                        async move { Ok(req.into_response(resp).map_into_right_body()) },
                    );
                }
            }
        }

        // Check whether a signing key is configured.
        let key_data: Option<web::Data<SigningKeyData>> =
            req.app_data::<web::Data<SigningKeyData>>().cloned();

        let key_data = match key_data {
            Some(k) => k,
            None => {
                // No signing key. Auth disabled.
                // /auth/* endpoints don't exist when auth is disabled.
                if path.starts_with("/auth/") {
                    let resp = HttpResponse::NotFound().json(serde_json::json!({
                        "error": "Authentication is not enabled on this server."
                    }));
                    return Box::pin(
                        async move { Ok(req.into_response(resp).map_into_right_body()) },
                    );
                }
                // Pass through all other requests.
                let fut = self.service.call(req);
                return Box::pin(async move {
                    let res = fut.await?;
                    Ok(res.map_into_left_body())
                });
            }
        };

        // Always allow /discover and /time without auth.
        if path == "/discover" || path == "/time" {
            let fut = self.service.call(req);
            return Box::pin(async move {
                let res = fut.await?;
                Ok(res.map_into_left_body())
            });
        }

        // Extract Bearer token from Authorization header.
        let token = req
            .headers()
            .get(AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "));

        let token = match token {
            Some(t) => t.to_string(),
            None => {
                let resp = HttpResponse::Unauthorized()
                    .insert_header(("WWW-Authenticate", "Bearer realm=\"zeusd\""))
                    .json(serde_json::json!({"error": "Authentication required. Provide an Authorization: Bearer <token> header."}));
                return Box::pin(async move { Ok(req.into_response(resp).map_into_right_body()) });
            }
        };

        // Validate the JWT.
        let mut validation = Validation::new(jsonwebtoken::Algorithm::HS256);
        // Allow tokens without `exp` claim (no-expiry tokens).
        validation.required_spec_claims.remove("exp");
        validation.validate_exp = true; // Still validate if present.

        let token_data = jsonwebtoken::decode::<Claims>(&token, &key_data.0, &validation);

        let claims = match token_data {
            Ok(data) => data.claims,
            Err(e) => {
                let msg = match e.kind() {
                    jsonwebtoken::errors::ErrorKind::ExpiredSignature => {
                        "Token has expired.".to_string()
                    }
                    _ => format!("Invalid token: {e}"),
                };
                let resp = HttpResponse::Unauthorized()
                    .insert_header(("WWW-Authenticate", "Bearer realm=\"zeusd\""))
                    .json(serde_json::json!({"error": msg}));
                return Box::pin(async move { Ok(req.into_response(resp).map_into_right_body()) });
            }
        };

        // Check scope.
        if let Some(scope) = required_scope(&path) {
            if !claims.scopes.contains(&scope) {
                let scope_list: Vec<String> =
                    claims.scopes.iter().map(|s| format!("'{s}'")).collect();
                let resp = HttpResponse::Forbidden().json(serde_json::json!({
                    "error": format!(
                        "Token for user '{}' lacks required scope '{}'. Token scopes: [{}]",
                        claims.sub, scope, scope_list.join(", "),
                    )
                }));
                return Box::pin(async move { Ok(req.into_response(resp).map_into_right_body()) });
            }
        }

        // Attach claims to request extensions for downstream handlers.
        req.extensions_mut().insert(claims.clone());
        tracing::info!(user = %claims.sub, "Authenticated request");

        let fut = self.service.call(req);
        Box::pin(async move {
            let res = fut.await?;
            Ok(res.map_into_left_body())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_issue_and_decode_token() {
        let key = b"test-secret-key-for-unit-tests!!";
        let token = issue_token(
            key,
            "testuser",
            vec![ApiGroup::GpuRead, ApiGroup::CpuRead],
            Some(9999999999),
        )
        .unwrap();

        let decoding_key = DecodingKey::from_secret(key);
        let mut validation = Validation::new(jsonwebtoken::Algorithm::HS256);
        validation.required_spec_claims.remove("exp");

        let data = jsonwebtoken::decode::<Claims>(&token, &decoding_key, &validation).unwrap();
        assert_eq!(data.claims.sub, "testuser");
        assert_eq!(
            data.claims.scopes,
            vec![ApiGroup::GpuRead, ApiGroup::CpuRead]
        );
        assert_eq!(data.claims.exp, Some(9999999999));
    }

    #[test]
    fn test_issue_token_no_expiry() {
        let key = b"test-secret-key-for-unit-tests!!";
        let token = issue_token(key, "testuser", vec![ApiGroup::GpuControl], None).unwrap();

        let decoding_key = DecodingKey::from_secret(key);
        let mut validation = Validation::new(jsonwebtoken::Algorithm::HS256);
        validation.required_spec_claims.remove("exp");
        validation.validate_exp = false;

        let data = jsonwebtoken::decode::<Claims>(&token, &decoding_key, &validation).unwrap();
        assert_eq!(data.claims.sub, "testuser");
        assert_eq!(data.claims.exp, None);
    }

    #[test]
    fn test_wrong_key_fails() {
        let key = b"test-secret-key-for-unit-tests!!";
        let token =
            issue_token(key, "testuser", vec![ApiGroup::GpuRead], Some(9999999999)).unwrap();

        let wrong_key = DecodingKey::from_secret(b"wrong-key!!!!!!!!!!!!!!!!!!!!!!!");
        let mut validation = Validation::new(jsonwebtoken::Algorithm::HS256);
        validation.required_spec_claims.remove("exp");

        let result = jsonwebtoken::decode::<Claims>(&token, &wrong_key, &validation);
        assert!(result.is_err());
    }

    #[test]
    fn test_required_scope() {
        assert_eq!(required_scope("/discover"), None);
        assert_eq!(required_scope("/time"), None);
        assert_eq!(
            required_scope("/gpu/set_power_limit"),
            Some(ApiGroup::GpuControl)
        );
        assert_eq!(
            required_scope("/gpu/set_persistence_mode"),
            Some(ApiGroup::GpuControl)
        );
        assert_eq!(
            required_scope("/gpu/reset_gpu_locked_clocks"),
            Some(ApiGroup::GpuControl)
        );
        assert_eq!(required_scope("/gpu/get_power"), Some(ApiGroup::GpuRead));
        assert_eq!(required_scope("/gpu/stream_power"), Some(ApiGroup::GpuRead));
        assert_eq!(
            required_scope("/gpu/get_cumulative_energy"),
            Some(ApiGroup::GpuRead)
        );
        assert_eq!(required_scope("/cpu/get_power"), Some(ApiGroup::CpuRead));
        assert_eq!(required_scope("/cpu/stream_power"), Some(ApiGroup::CpuRead));
        assert_eq!(
            required_scope("/cpu/get_cumulative_energy"),
            Some(ApiGroup::CpuRead)
        );
    }

    #[test]
    fn test_api_group_serde_roundtrip() {
        let groups = vec![ApiGroup::GpuControl, ApiGroup::GpuRead, ApiGroup::CpuRead];
        let json = serde_json::to_string(&groups).unwrap();
        assert_eq!(json, r#"["gpu-control","gpu-read","cpu-read"]"#);

        let parsed: Vec<ApiGroup> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, groups);
    }
}
