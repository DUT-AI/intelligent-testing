from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import settings

# Setup SQLAlchemy engine and Session factory
engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
