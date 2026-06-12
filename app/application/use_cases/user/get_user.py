from typing import Optional
from app.domain.entities.user import User
from app.domain.interfaces.user_repository import UserRepository


class GetUserUseCase:
    """
    Use Case: Retrieve a User by ID.
    """

    def __init__(self, user_repository: UserRepository):
        self.user_repository = user_repository

    def execute(self, user_id: int) -> Optional[User]:
        return self.user_repository.find_by_id(user_id)
