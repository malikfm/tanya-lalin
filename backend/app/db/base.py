from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

metadata = MetaData(schema="public")


class Base(DeclarativeBase):
    metadata = metadata
