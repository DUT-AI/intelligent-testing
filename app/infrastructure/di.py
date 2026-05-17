from typing import Iterator
from dishka import Provider, Scope, provide
from sqlalchemy.orm import Session
from sqlalchemy import Engine

from app.domain.interfaces.user_repository import UserRepository
from app.infrastructure.database.repositories import SQLAlchemyUserRepository
from app.use_cases.user.create_user import CreateUserUseCase
from app.use_cases.user.get_user import GetUserUseCase
from app.infrastructure.database.connection import engine, SessionLocal


class AppProvider(Provider):
    """
    Dishka Provider to register application dependencies.
    Organizes dependencies into proper scopes (APP vs REQUEST).
    """

    @provide(scope=Scope.APP)
    def get_engine(self) -> Engine:
        """
        Database engine is created once and lives for the entire application life.
        """
        return engine

    @provide(scope=Scope.REQUEST)
    def get_db(self) -> Iterator[Session]:
        """
        Database session is request-scoped.
        Automatically closed when the request lifecycle ends.
        """
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    @provide(scope=Scope.REQUEST)
    def get_user_repository(self, session: Session) -> UserRepository:
        """
        Repository depends on the current DB Session.
        Implements UserRepository interface.
        """
        return SQLAlchemyUserRepository(session)

    @provide(scope=Scope.REQUEST)
    def get_create_user_use_case(self, repo: UserRepository) -> CreateUserUseCase:
        """
        Use Case depends on UserRepository interface.
        """
        return CreateUserUseCase(repo)

    @provide(scope=Scope.REQUEST)
    def get_get_user_use_case(self, repo: UserRepository) -> GetUserUseCase:
        """
        Use Case depends on UserRepository interface.
        """
        return GetUserUseCase(repo)
