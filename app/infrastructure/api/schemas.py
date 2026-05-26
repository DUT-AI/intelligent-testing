from pydantic import BaseModel, EmailStr


class UserCreateRequest(BaseModel):
    """
    Pydantic schema to validate user creation request payload.
    """

    name: str
    email: EmailStr


class UserResponse(BaseModel):
    """
    Pydantic schema to serialize user information in API responses.
    """

    id: int
    name: str
    email: EmailStr
    is_active: bool

    class Config:
        # Allows Pydantic to read fields from pure dataclasses and objects
        from_attributes = True
        orm_mode = True
