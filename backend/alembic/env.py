from alembic import context
from sqlalchemy import engine_from_config, pool

from config import settings
from app.db.base import Base
# import all models before target_metadata is set
from app.db import models  # noqa: register it to Base.metadata
from logging_setup import setup_logger

logger = setup_logger()
config = context.config
config.set_main_option(
    "sqlalchemy.url",
    f"postgresql+psycopg://{settings.db_user}:{settings.db_password}@{settings.db_host}:{settings.db_port}/{settings.db_name}",
)
target_metadata = Base.metadata


def run_migrations_online() -> None:
    engine = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool
    )

    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True
        )

        with context.begin_transaction():
            context.run_migrations()


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        literal_binds=True
    )

    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
