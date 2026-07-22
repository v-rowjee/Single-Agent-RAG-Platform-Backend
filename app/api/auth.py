"""Authentication dependencies shared by API routers."""

from typing import Annotated

from fastapi import Depends

from app.core.auth import CurrentUser, get_current_user


AuthenticatedUser = Annotated[CurrentUser, Depends(get_current_user)]
