from typing import List, Optional
from sqlalchemy.orm import Session
from app.domain.entities.user import User
from app.domain.interfaces.user_repository import UserRepository
from app.infrastructure.database.models import UserORM


class SQLAlchemyUserRepository(UserRepository):
    """
    SQLAlchemy-specific concrete implementation of UserRepository.
    Handles persistence and ORM to Domain Entity mappings.
    """

    def __init__(self, session: Session):
        self.session = session

    def _to_domain(self, orm: UserORM) -> User:
        """Helper to map a database ORM object into a pure Domain Entity."""
        return User(id=orm.id, name=orm.name, email=orm.email, is_active=orm.is_active)

    def save(self, user: User) -> User:
        """
        Saves or updates a User.
        Translates a domain entity into a database model before performing save.
        """
        orm: UserORM | None = None
        if user.id is not None:
            # Update operation
            orm = self.session.query(UserORM).filter(UserORM.id == user.id).first()
            if orm:
                orm.name = user.name
                orm.email = user.email
                orm.is_active = user.is_active

        if orm is None:
            # Create operation (or fallback if ID not found)
            orm = UserORM(name=user.name, email=user.email, is_active=user.is_active)
            self.session.add(orm)

        self.session.commit()
        self.session.refresh(orm)
        return self._to_domain(orm)

    def find_by_id(self, user_id: int) -> Optional[User]:
        """Finds a user by primary key."""
        orm = self.session.query(UserORM).filter(UserORM.id == user_id).first()
        return self._to_domain(orm) if orm else None

    def find_by_email(self, email: str) -> Optional[User]:
        """Finds a user by email address."""
        orm = self.session.query(UserORM).filter(UserORM.email == email).first()
        return self._to_domain(orm) if orm else None

    def list_all(self) -> List[User]:
        """Retrieves all users."""
        orms = self.session.query(UserORM).all()
        return [self._to_domain(orm) for orm in orms]
