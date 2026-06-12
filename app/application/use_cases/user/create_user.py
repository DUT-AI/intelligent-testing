from app.domain.entities.user import User
from app.domain.interfaces.user_repository import UserRepository


class CreateUserUseCase:
    """
    Use Case: Create a new User.
    Contains application-specific business logic for user creation.
    """

    def __init__(self, user_repository: UserRepository):
        self.user_repository = user_repository

    def execute(self, name: str, email: str) -> User:
        # Core Business Rule: Email must be unique
        existing_user = self.user_repository.find_by_email(email)
        if existing_user:
            raise ValueError(f"Email {email} is already in use.")

        user = User(id=None, name=name, email=email)
        return self.user_repository.save(user)
