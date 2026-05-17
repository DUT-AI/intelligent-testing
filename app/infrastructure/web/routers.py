from fastapi import APIRouter, HTTPException, status
from dishka.integrations.fastapi import FromDishka, inject

from app.use_cases.user.create_user import CreateUserUseCase
from app.use_cases.user.get_user import GetUserUseCase
from app.infrastructure.web.schemas import UserCreateRequest, UserResponse

router = APIRouter(prefix="/users", tags=["Users"])


@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
@inject
def create_user(request: UserCreateRequest, use_case: FromDishka[CreateUserUseCase]):
    """
    Endpoint to register a new user.
    Uses CreateUserUseCase injected automatically by Dishka.
    """
    try:
        created_user = use_case.execute(name=request.name, email=request.email)
        return created_user
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/{user_id}", response_model=UserResponse)
@inject
def get_user(user_id: int, use_case: FromDishka[GetUserUseCase]):
    """
    Endpoint to retrieve a user by ID.
    Uses GetUserUseCase injected automatically by Dishka.
    """
    user = use_case.execute(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found.",
        )
    return user
