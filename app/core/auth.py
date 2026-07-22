"""FastAPI authentication dependency for Supabase-issued access JWTs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Annotated, Any
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import get_settings
from app.services.persistence.supabase import supabase_gateway


logger = logging.getLogger(__name__)
bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class CurrentUser:
    id: str
    email: str | None = None


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="A valid bearer access token is required.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _claims_from_token(access_token: str) -> dict[str, Any]:
    """Verify a Supabase JWT, including the SDK's legacy-HS256 fallback."""
    claims_response = supabase_gateway.client.auth.get_claims(access_token)
    # supabase-py 2.x returns a TypedDict, not an object with a ``claims``
    # attribute. Attribute access therefore returns None even after the JWT
    # has been successfully verified.
    claims = (
        claims_response.get("claims")
        if isinstance(claims_response, dict)
        else getattr(claims_response, "claims", None)
    )
    if not isinstance(claims, dict):
        raise ValueError("Supabase did not return verified JWT claims.")
    return claims


def get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer_scheme),
    ],
) -> CurrentUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized()

    try:
        claims = _claims_from_token(credentials.credentials)
        expected_issuer = f"{get_settings().supabase_url.rstrip('/')}/auth/v1"
        if not expected_issuer or claims.get("iss") != expected_issuer:
            raise ValueError("Unexpected JWT issuer.")
        if claims.get("role") != "authenticated":
            raise ValueError("JWT is not an authenticated user session.")

        user_id = str(UUID(str(claims["sub"])))
        email_claim = claims.get("email")
        email = str(email_claim) if isinstance(email_claim, str) else None
        return CurrentUser(id=user_id, email=email)
    except Exception as error:
        if isinstance(error, HTTPException):
            raise
        # Keep the response deliberately generic, but retain the verification
        # failure in server logs. Never log the submitted access token.
        logger.warning(
            "Bearer access-token verification failed (%s): %s",
            type(error).__name__,
            error,
        )
        raise _unauthorized() from error
