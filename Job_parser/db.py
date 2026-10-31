from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine


def make_engine(url: str) -> Engine:
    """SQLAlchemy engine; SQLite gets WAL + a busy timeout so concurrent API requests queue for the
    write lock instead of failing with 'database is locked'."""
    engine = create_engine(url, echo=False, future=True)
    if url.startswith("sqlite") and ":memory:" not in url:
        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_connection, _record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=15000")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()
    return engine
