from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.core import auth
from app.core.config import Settings


USER_ID = "59b3d0fc-2d4a-40a0-8bb1-99e19da406ee"


class _ClaimsClient:
    class Auth:
        @staticmethod
        def get_claims(access_token: str) -> dict[str, object]:
            assert access_token == "access-jwt"
            return {"claims": {"sub": USER_ID, "role": "authenticated"}}

    auth = Auth()


def _credentials() -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials="access-jwt")


def test_claims_from_token_reads_supabase_typed_dict_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auth.supabase_service, "_client", _ClaimsClient())

    claims = auth._claims_from_token("access-jwt")

    assert claims == {"sub": USER_ID, "role": "authenticated"}


def test_get_current_user_returns_verified_authenticated_subject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        auth,
        "get_settings",
        lambda: Settings("https://project.supabase.co", "service-key"),
    )
    monkeypatch.setattr(
        auth,
        "_claims_from_token",
        lambda token: {
            "iss": "https://project.supabase.co/auth/v1",
            "role": "authenticated",
            "sub": USER_ID,
            "email": "owner@example.com",
        },
    )

    user = auth.get_current_user(_credentials())

    assert user.id == USER_ID
    assert user.email == "owner@example.com"


@pytest.mark.parametrize(
    "credentials, claims",
    [
        (None, {}),
        (_credentials(), {"iss": "https://wrong.supabase.co/auth/v1"}),
        (
            _credentials(),
            {
                "iss": "https://project.supabase.co/auth/v1",
                "role": "anon",
                "sub": USER_ID,
            },
        ),
    ],
)
def test_get_current_user_rejects_missing_or_invalid_jwt(
    monkeypatch: pytest.MonkeyPatch,
    credentials: HTTPAuthorizationCredentials | None,
    claims: dict[str, str],
) -> None:
    monkeypatch.setattr(
        auth,
        "get_settings",
        lambda: Settings("https://project.supabase.co", "service-key"),
    )
    monkeypatch.setattr(auth, "_claims_from_token", lambda token: claims)

    with pytest.raises(HTTPException) as error:
        auth.get_current_user(credentials)

    assert error.value.status_code == 401
