"""
Database Connection Manager — Repository pattern with context manager.

Abstracts all SQLite operations behind a clean interface.
Uses context managers for safe connection handling (auto-commit/rollback).
"""

import logging
import sqlite3
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite database connections and operations.

    Usage:
        with DatabaseManager('path/to/db.sqlite') as db:
            db.insert_prices(df)
            df = db.get_prices(['AAPL', 'MSFT'], '2020-01-01', '2025-01-01')
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.conn: Optional[sqlite3.Connection] = None

    def __enter__(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        logger.info(f"Connected to database: {self.db_path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
                logger.error(f"Database error, rolled back: {exc_val}")
            self.conn.close()
            self.conn = None
        return False

    def _get_conn(self) -> sqlite3.Connection:
        """Get the active connection or raise."""
        if self.conn is None:
            raise RuntimeError(
                "No active connection. Use 'with DatabaseManager(...) as db:'"
            )
        return self.conn

    def initialize_schema(self) -> None:
        """Create all tables from schema.sql."""
        schema_path = Path(__file__).parent / "schema.sql"
        with open(schema_path, "r") as f:
            schema_sql = f.read()
        self._get_conn().executescript(schema_sql)
        logger.info("Database schema initialized")

    # ─────────────────────────────────────────
    # Securities
    # ─────────────────────────────────────────
    def upsert_security(
        self, ticker: str, name: str, sector: str,
        market_cap: float = 0, shares_outstanding: float = 0,
        free_float_factor: float = 1.0,
    ) -> None:
        """Insert or update a security record."""
        self._get_conn().execute(
            """
            INSERT INTO securities (ticker, name, sector, market_cap,
                                    shares_outstanding, free_float_factor)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                name=excluded.name,
                sector=excluded.sector,
                market_cap=excluded.market_cap,
                shares_outstanding=excluded.shares_outstanding,
                free_float_factor=excluded.free_float_factor
            """,
            (ticker, name, sector, market_cap, shares_outstanding, free_float_factor),
        )

    def get_securities(self) -> pd.DataFrame:
        """Get all securities as a DataFrame."""
        return pd.read_sql("SELECT * FROM securities", self._get_conn())

    # ─────────────────────────────────────────
    # Prices
    # ─────────────────────────────────────────
    def insert_prices(self, df: pd.DataFrame) -> int:
        """Bulk insert price data.

        Args:
            df: DataFrame with columns: ticker, trade_date, open_price,
                high_price, low_price, close_price, adj_close, volume.

        Returns:
            Number of rows inserted.
        """
        conn = self._get_conn()
        rows_inserted = 0

        for _, row in df.iterrows():
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO daily_prices
                    (ticker, trade_date, open_price, high_price, low_price,
                     close_price, adj_close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["ticker"],
                        str(row["trade_date"]),
                        row.get("open_price"),
                        row.get("high_price"),
                        row.get("low_price"),
                        row.get("close_price"),
                        row.get("adj_close"),
                        row.get("volume"),
                    ),
                )
                rows_inserted += 1
            except sqlite3.Error as e:
                logger.warning(f"Failed to insert price for {row.get('ticker')}: {e}")

        conn.commit()
        logger.info(f"Inserted {rows_inserted} price rows")
        return rows_inserted

    def get_prices(
        self,
        tickers: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Retrieve price data with optional filters.

        Returns a pivot table: index=trade_date, columns=ticker, values=adj_close.
        """
        query = "SELECT ticker, trade_date, adj_close FROM daily_prices WHERE 1=1"
        params: list = []

        if tickers:
            placeholders = ",".join("?" * len(tickers))
            query += f" AND ticker IN ({placeholders})"
            params.extend(tickers)
        if start_date:
            query += " AND trade_date >= ?"
            params.append(str(start_date))
        if end_date:
            query += " AND trade_date <= ?"
            params.append(str(end_date))

        query += " ORDER BY trade_date, ticker"

        df = pd.read_sql(query, self._get_conn(), params=params)

        if df.empty:
            return df

        # Pivot to wide format: dates as rows, tickers as columns
        pivot = df.pivot(index="trade_date", columns="ticker", values="adj_close")
        pivot.index = pd.to_datetime(pivot.index)
        pivot.index.name = "date"

        return pivot

    def get_price_count(self) -> int:
        """Count total price records."""
        cursor = self._get_conn().execute("SELECT COUNT(*) FROM daily_prices")
        return cursor.fetchone()[0]

    # ─────────────────────────────────────────
    # Index Levels
    # ─────────────────────────────────────────
    def insert_index_levels(self, index_name: str, df: pd.DataFrame) -> None:
        """Store computed index levels.

        Args:
            index_name: Name of the index.
            df: DataFrame with 'index_level' and 'daily_return' columns,
                indexed by date.
        """
        conn = self._get_conn()

        for dt, row in df.iterrows():
            conn.execute(
                """
                INSERT OR REPLACE INTO index_levels
                (index_name, trade_date, index_level, daily_return)
                VALUES (?, ?, ?, ?)
                """,
                (index_name, str(dt.date() if hasattr(dt, 'date') else dt),
                 row["index_level"], row["daily_return"]),
            )
        conn.commit()

    def get_index_levels(self, index_name: str) -> pd.DataFrame:
        """Retrieve index level series."""
        df = pd.read_sql(
            "SELECT trade_date, index_level, daily_return FROM index_levels "
            "WHERE index_name = ? ORDER BY trade_date",
            self._get_conn(),
            params=[index_name],
        )
        if not df.empty:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df.set_index("trade_date", inplace=True)
        return df

    # ─────────────────────────────────────────
    # Index Constituents
    # ─────────────────────────────────────────
    def insert_constituents(
        self, index_name: str, weights: dict, as_of_date: str,
    ) -> None:
        """Store a constituent snapshot."""
        conn = self._get_conn()
        for ticker, weight in weights.items():
            conn.execute(
                """
                INSERT OR REPLACE INTO index_constituents
                (index_name, ticker, as_of_date, weight, is_active)
                VALUES (?, ?, ?, ?, 1)
                """,
                (index_name, ticker, as_of_date, weight),
            )
        conn.commit()

    # ─────────────────────────────────────────
    # Rebalance Log
    # ─────────────────────────────────────────
    def log_rebalance(
        self, index_name: str, rebalance_date: str,
        ticker: str, action: str,
        old_weight: float = 0, new_weight: float = 0,
    ) -> None:
        """Log a single rebalance action."""
        self._get_conn().execute(
            """
            INSERT INTO rebalance_log
            (index_name, rebalance_date, ticker, action, old_weight, new_weight)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (index_name, rebalance_date, ticker, action, old_weight, new_weight),
        )

    # ─────────────────────────────────────────
    # Raw query execution
    # ─────────────────────────────────────────
    def execute_query(self, query: str, params: tuple = ()) -> pd.DataFrame:
        """Execute an arbitrary SELECT query and return results as DataFrame."""
        return pd.read_sql(query, self._get_conn(), params=params)
