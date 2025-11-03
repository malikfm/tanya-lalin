from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    log_level: str = "INFO"
    db_host: str = "localhost"
    db_port: int = 5432
    db_user: str = "admin"
    db_password: str = "admin"
    db_name: str = "test_db"
    db_schema: str = "test"
    embedding_dim: int = 768

    class Config:
        env_file = ".env"


settings = Settings()
