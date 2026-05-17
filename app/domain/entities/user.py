from dataclasses import dataclass
from typing import Optional


@dataclass
class User:
    """
    Pure Domain Entity.
    This class is completely independent of frameworks, ORMs (SQLAlchemy), or databases.
    It represents the core domain object and business rules.
    """

    id: Optional[int]
    name: str
    email: str
    is_active: bool = True
