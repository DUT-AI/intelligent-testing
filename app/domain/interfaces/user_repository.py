from abc import ABC, abstractmethod
from typing import List, Optional
from app.domain.entities.user import User


class UserRepository(ABC):
    """
    Abstract Interface for User Repository.
    Defined in the domain layer to specify data access contracts.
    """

    @abstractmethod
    def save(self, user: User) -> User:
        """Saves a user domain entity to the database."""
        pass

    @abstractmethod
    def find_by_id(self, user_id: int) -> Optional[User]:
        """Finds a user domain entity by their ID."""
        pass

    @abstractmethod
    def find_by_email(self, email: str) -> Optional[User]:
        """Finds a user domain entity by their email address."""
        pass

    @abstractmethod
    def list_all(self) -> List[User]:
        """Lists all user domain entities."""
        pass
