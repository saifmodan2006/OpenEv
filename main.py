import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("sql-query-generator")

APP_TITLE = "SQL Query Generator"
MAX_STEPS = 5
SESSION_TTL_SECONDS = 60 * 60
FORBIDDEN_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH|PRAGMA|VACUUM)\b",
    re.IGNORECASE,
)

SCHEMA_DESCRIPTION = """
users:
- id (INTEGER PRIMARY KEY)
- name (TEXT)
- age (INTEGER)
- email (TEXT)
- country (TEXT)

products:
- id (INTEGER PRIMARY KEY)
- name (TEXT)
- category (TEXT)
- price (REAL)

orders:
- id (INTEGER PRIMARY KEY)
- user_id (INTEGER) -> users.id
- product_id (INTEGER) -> products.id
- quantity (INTEGER)
- order_date (TEXT)
""".strip()

SCHEMA_VISUALIZATION = """
users
  id PK
  name
  age
  email
  country

products
  id PK
  name
  category
  price

orders
  id PK
  user_id FK -> users.id
  product_id FK -> products.id
  quantity
  order_date
""".strip()


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    description: str
    difficulty: str
    ground_truth_sql: str
    order_sensitive: bool
    required_patterns: tuple[tuple[str, str], ...] = ()


TASKS: dict[str, TaskDefinition] = {
    "task_easy": TaskDefinition(
        task_id="task_easy",
        description="Find all users older than 25",
        difficulty="easy",
        ground_truth_sql=(
            "SELECT id, name, age, email, country "
            "FROM users "
            "WHERE age > 25 "
            "ORDER BY id ASC;"
        ),
        order_sensitive=False,
        required_patterns=(
            (r"\bWHERE\b", "Add a WHERE clause that filters the users table."),
            (r"\bage\s*>\s*25\b", "Use the correct age filter: older than 25."),
        ),
    ),
    "task_medium": TaskDefinition(
        task_id="task_medium",
        description="Find total quantity ordered per product with product name, sorted descending",
        difficulty="medium",
        ground_truth_sql=(
            "SELECT p.name, SUM(o.quantity) AS total_quantity "
            "FROM orders o "
            "JOIN products p ON o.product_id = p.id "
            "GROUP BY p.id, p.name "
            "ORDER BY total_quantity DESC, p.name ASC;"
        ),
        order_sensitive=True,
        required_patterns=(
            (r"\bJOIN\b", "Missing JOIN: combine orders with products on product_id."),
            (r"\bGROUP\s+BY\b", "Wrong grouping: aggregate quantity per product."),
            (r"\bSUM\s*\(\s*o?\.?quantity\s*\)", "Sum the quantity column for each product."),
            (
                r"\bORDER\s+BY\b",
                "Sort the aggregated results by total quantity in descending order.",
            ),
        ),
    ),
    "task_hard": TaskDefinition(
        task_id="task_hard",
        description="Find users who ordered Electronics but NEVER ordered Clothing",
        difficulty="hard",
        ground_truth_sql=(
            "SELECT DISTINCT u.id, u.name, u.age, u.email, u.country "
            "FROM users u "
            "JOIN orders o ON u.id = o.user_id "
            "JOIN products p ON o.product_id = p.id "
            "WHERE p.category = 'Electronics' "
            "AND u.id NOT IN ("
            "    SELECT DISTINCT o2.user_id "
            "    FROM orders o2 "
            "    JOIN products p2 ON o2.product_id = p2.id "
            "    WHERE p2.category = 'Clothing'"
            ") "
            "ORDER BY u.id ASC;"
        ),
        order_sensitive=False,
        required_patterns=(
            (
                r"\bElectronics\b",
                "Incorrect filter: keep users who ordered products in the Electronics category.",
            ),
            (
                r"\bNOT\s+IN\b|\bNOT\s+EXISTS\b",
                "Missing exclusion: remove users who have any Clothing orders.",
            ),
            (
                r"\bJOIN\b",
                "Missing JOIN: connect users, orders, and products to evaluate purchase categories.",
            ),
        ),
    ),
}


class ResetRequest(BaseModel):
    task_id: str = Field(..., description="Task identifier to start.")


class ResetObservation(BaseModel):
    task_description: str
    schema_text: str = Field(..., alias="schema")
    schema_visualization: str
    difficulty: str

    model_config = {"populate_by_name": True}


class ResetResponse(BaseModel):
    session_id: str
    observation: ResetObservation
    reward: float = 0.0
    done: bool = False


class StepRequest(BaseModel):
    session_id: str
    sql_query: str


class QueryFeedback(BaseModel):
    status: str
    error: str | None = None
    expected_rows: int = 0
    returned_rows: int = 0
    hint: str
    retry_hint: str | None = None
    execution_time_ms: float = 0.0
    expected_execution_time_ms: float = 0.0


class StepObservation(BaseModel):
    submitted_query: str
    feedback: QueryFeedback


class StepResponse(BaseModel):
    observation: StepObservation
    reward: float
    done: bool


class HistoryEntry(BaseModel):
    step: int
    sql_query: str
    reward: float
    feedback: QueryFeedback
    created_at: str


class StateResponse(BaseModel):
    session_id: str
    task_id: str
    task_description: str
    difficulty: str
    steps_taken: int
    history: list[HistoryEntry]
    done: bool
    created_at: str


class TaskSummary(BaseModel):
    task_id: str
    description: str
    difficulty: str


class TasksResponse(BaseModel):
    tasks: list[TaskSummary]


class SchemaResponse(BaseModel):
    schema_text: str = Field(..., alias="schema")
    schema_visualization: str
    allowed_query_types: list[str]
    max_steps: int

    model_config = {"populate_by_name": True}


@dataclass
class SessionRecord:
    session_id: str
    task: TaskDefinition
    connection: sqlite3.Connection
    created_at: datetime
    history: list[HistoryEntry] = field(default_factory=list)
    steps_taken: int = 0
    done: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def to_state(self) -> StateResponse:
        return StateResponse(
            session_id=self.session_id,
            task_id=self.task.task_id,
            task_description=self.task.description,
            difficulty=self.task.difficulty,
            steps_taken=self.steps_taken,
            history=self.history,
            done=self.done,
            created_at=self.created_at.isoformat(),
        )


SESSIONS: dict[str, SessionRecord] = {}
SESSIONS_LOCK = threading.Lock()

app = FastAPI(
    title=APP_TITLE,
    version="1.0.0",
    description=(
        "OpenEnv RL environment for evaluating SQL generation agents against "
        "deterministic SQLite tasks."
    ),
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_sql(sql_query: str) -> str:
    return sql_query.strip().rstrip(";").strip()


def validate_select_query(sql_query: str) -> tuple[bool, str]:
    normalized = normalize_sql(sql_query)
    if not normalized:
        return False, "Only SELECT queries are allowed"

    if FORBIDDEN_PATTERN.search(normalized):
        return False, "Only SELECT queries are allowed"

    if not normalized.upper().startswith(("SELECT", "WITH")):
        return False, "Only SELECT queries are allowed"

    if ";" in normalized:
        return False, "Only SELECT queries are allowed"

    return True, ""


def create_session_database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.execute("PRAGMA foreign_keys = ON;")
    connection.executescript(
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            email TEXT NOT NULL,
            country TEXT NOT NULL
        );

        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL
        );

        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            order_date TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        );
        """
    )

    connection.executemany(
        "INSERT INTO users (id, name, age, email, country) VALUES (?, ?, ?, ?, ?)",
        [
            (1, "Alice Johnson", 29, "alice.johnson@example.com", "United States"),
            (2, "Bob Chen", 24, "bob.chen@example.com", "Canada"),
            (3, "Carla Mendes", 31, "carla.mendes@example.com", "Brazil"),
            (4, "Diego Singh", 27, "diego.singh@example.com", "India"),
            (5, "Evelyn Hart", 22, "evelyn.hart@example.com", "Germany"),
            (6, "Farah Noor", 35, "farah.noor@example.com", "United Arab Emirates"),
        ],
    )
    connection.executemany(
        "INSERT INTO products (id, name, category, price) VALUES (?, ?, ?, ?)",
        [
            (1, "Apex Laptop", "Electronics", 1299.00),
            (2, "Nimbus Headphones", "Electronics", 199.99),
            (3, "Classic T-Shirt", "Clothing", 29.99),
            (4, "Trail Jacket", "Clothing", 89.50),
            (5, "Metro Watch", "Accessories", 149.00),
            (6, "Urban Backpack", "Accessories", 59.99),
        ],
    )
    connection.executemany(
        """
        INSERT INTO orders (id, user_id, product_id, quantity, order_date)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (1, 1, 1, 1, "2026-01-10"),
            (2, 1, 5, 2, "2026-01-11"),
            (3, 2, 3, 3, "2026-01-12"),
            (4, 3, 2, 1, "2026-02-01"),
            (5, 3, 4, 1, "2026-02-03"),
            (6, 4, 2, 2, "2026-02-10"),
            (7, 4, 6, 1, "2026-02-11"),
            (8, 5, 5, 1, "2026-03-01"),
            (9, 6, 1, 1, "2026-03-05"),
            (10, 6, 6, 2, "2026-03-06"),
        ],
    )
    connection.commit()
    return connection


def close_session(record: SessionRecord) -> None:
    try:
        record.connection.close()
    except sqlite3.Error:
        logger.exception("Failed to close session database", extra={"session_id": record.session_id})


def prune_stale_sessions() -> None:
    cutoff = utc_now().timestamp() - SESSION_TTL_SECONDS
    stale_session_ids: list[str] = []

    with SESSIONS_LOCK:
        for session_id, record in SESSIONS.items():
            if record.created_at.timestamp() < cutoff:
                stale_session_ids.append(session_id)

        for session_id in stale_session_ids:
            record = SESSIONS.pop(session_id)
            close_session(record)

    if stale_session_ids:
        logger.info("Pruned stale sessions", extra={"count": len(stale_session_ids)})


def fetch_session(session_id: str) -> SessionRecord:
    prune_stale_sessions()
    with SESSIONS_LOCK:
        record = SESSIONS.get(session_id)

    if record is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' was not found")

    return record


def execute_query(connection: sqlite3.Connection, sql_query: str) -> tuple[list[tuple[Any, ...]], float]:
    start = time.perf_counter()
    cursor = connection.execute(sql_query)
    rows = [tuple(row) for row in cursor.fetchall()]
    elapsed_ms = round((time.perf_counter() - start) * 1000, 3)
    return rows, elapsed_ms


def normalize_rows(rows: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    return sorted(rows, key=lambda item: tuple("" if value is None else str(value) for value in item))


def compare_results(
    expected_rows: list[tuple[Any, ...]],
    returned_rows: list[tuple[Any, ...]],
    *,
    order_sensitive: bool,
) -> float:
    left = expected_rows if order_sensitive else normalize_rows(expected_rows)
    right = returned_rows if order_sensitive else normalize_rows(returned_rows)

    if left == right:
        return 1.0

    expected_set = set(expected_rows)
    returned_set = set(returned_rows)
    if expected_set and returned_set and (
        expected_set.issubset(returned_set) or returned_set.issubset(expected_set)
    ):
        return 0.6

    if expected_set.intersection(returned_set):
        return 0.3

    return 0.0


def infer_hint(task: TaskDefinition, sql_query: str, reward: float, error: str | None) -> tuple[str, str | None]:
    normalized = normalize_sql(sql_query)

    if error:
        lowered = error.lower()
        if "no such table" in lowered:
            return "Use only the available tables: users, products, and orders.", "Check table names against the schema."
        if "no such column" in lowered:
            return "Reference only columns that exist in the provided schema.", "Verify column names and table aliases."
        if "syntax error" in lowered:
            return "The SQL syntax is invalid for SQLite.", "Return a single valid SQLite SELECT statement."
        return "The query did not execute successfully.", "Adjust the SQL based on the error and try again."

    if reward == 1.0:
        return "Query matched the expected result set.", None

    for pattern, message in task.required_patterns:
        if not re.search(pattern, normalized, re.IGNORECASE):
            return message, message

    if task.task_id == "task_easy":
        return "Incorrect filter: return only users older than 25.", "Recheck the WHERE clause against the age requirement."
    if task.task_id == "task_medium":
        return "Wrong grouping or ordering for per-product totals.", "Revisit the JOIN, aggregation, and descending sort."
    if task.task_id == "task_hard":
        return "Incorrect filter or exclusion logic for Electronics vs Clothing orders.", "Use a subquery that excludes any user with Clothing purchases."

    return "Query result did not match the expected rows.", "Compare the selected columns and filters with the task."


def make_feedback(
    *,
    status: str,
    error: str | None,
    expected_rows: int,
    returned_rows: int,
    hint: str,
    retry_hint: str | None,
    execution_time_ms: float,
    expected_execution_time_ms: float,
) -> QueryFeedback:
    return QueryFeedback(
        status=status,
        error=error,
        expected_rows=expected_rows,
        returned_rows=returned_rows,
        hint=hint,
        retry_hint=retry_hint,
        execution_time_ms=execution_time_ms,
        expected_execution_time_ms=expected_execution_time_ms,
    )


def add_history_entry(record: SessionRecord, sql_query: str, reward: float, feedback: QueryFeedback) -> None:
    entry = HistoryEntry(
        step=record.steps_taken,
        sql_query=sql_query,
        reward=reward,
        feedback=feedback,
        created_at=utc_now().isoformat(),
    )
    record.history.append(entry)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks", response_model=TasksResponse)
def list_tasks() -> TasksResponse:
    return TasksResponse(
        tasks=[
            TaskSummary(
                task_id=task.task_id,
                description=task.description,
                difficulty=task.difficulty,
            )
            for task in TASKS.values()
        ]
    )


@app.get("/schema", response_model=SchemaResponse)
def get_schema() -> SchemaResponse:
    return SchemaResponse(
        schema_text=SCHEMA_DESCRIPTION,
        schema_visualization=SCHEMA_VISUALIZATION,
        allowed_query_types=["SELECT"],
        max_steps=MAX_STEPS,
    )


@app.post("/reset", response_model=ResetResponse)
def reset_environment(request: ResetRequest) -> ResetResponse:
    task = TASKS.get(request.task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Unknown task_id '{request.task_id}'")

    prune_stale_sessions()
    session_id = str(uuid4())
    record = SessionRecord(
        session_id=session_id,
        task=task,
        connection=create_session_database(),
        created_at=utc_now(),
    )

    with SESSIONS_LOCK:
        SESSIONS[session_id] = record

    logger.info("Session reset", extra={"session_id": session_id, "task_id": task.task_id})

    return ResetResponse(
        session_id=session_id,
        observation=ResetObservation(
            task_description=task.description,
            schema_text=SCHEMA_DESCRIPTION,
            schema_visualization=SCHEMA_VISUALIZATION,
            difficulty=task.difficulty,
        ),
        reward=0.0,
        done=False,
    )


@app.post("/step", response_model=StepResponse)
def step_environment(request: StepRequest) -> StepResponse:
    record = fetch_session(request.session_id)

    with record.lock:
        if record.done:
            feedback = make_feedback(
                status="error",
                error="Session has already ended",
                expected_rows=0,
                returned_rows=0,
                hint="Start a new session with /reset to continue.",
                retry_hint=None,
                execution_time_ms=0.0,
                expected_execution_time_ms=0.0,
            )
            return StepResponse(
                observation=StepObservation(
                    submitted_query=request.sql_query,
                    feedback=feedback,
                ),
                reward=0.0,
                done=True,
            )

        record.steps_taken += 1
        task = record.task
        valid, validation_message = validate_select_query(request.sql_query)

        try:
            expected_rows, expected_execution_time_ms = execute_query(
                record.connection, task.ground_truth_sql
            )
        except sqlite3.Error as exc:
            logger.exception("Ground truth query failed")
            raise HTTPException(status_code=500, detail=f"Ground truth query failed: {exc}") from exc

        if not valid:
            feedback = make_feedback(
                status="error",
                error=validation_message,
                expected_rows=len(expected_rows),
                returned_rows=0,
                hint=validation_message,
                retry_hint="Return a single SQLite SELECT statement that answers the task.",
                execution_time_ms=0.0,
                expected_execution_time_ms=expected_execution_time_ms,
            )
            reward = 0.0
            record.done = record.steps_taken >= MAX_STEPS
            add_history_entry(record, request.sql_query, reward, feedback)
            logger.info(
                "Rejected non-select query",
                extra={
                    "session_id": record.session_id,
                    "task_id": task.task_id,
                    "step": record.steps_taken,
                    "done": record.done,
                },
            )
            return StepResponse(
                observation=StepObservation(submitted_query=request.sql_query, feedback=feedback),
                reward=reward,
                done=record.done,
            )

        try:
            returned_rows, execution_time_ms = execute_query(record.connection, request.sql_query)
            reward = compare_results(
                expected_rows,
                returned_rows,
                order_sensitive=task.order_sensitive,
            )
            hint, retry_hint = infer_hint(task, request.sql_query, reward, None)
            feedback = make_feedback(
                status="success",
                error=None,
                expected_rows=len(expected_rows),
                returned_rows=len(returned_rows),
                hint=hint,
                retry_hint=retry_hint,
                execution_time_ms=execution_time_ms,
                expected_execution_time_ms=expected_execution_time_ms,
            )
        except sqlite3.Error as exc:
            reward = 0.0
            execution_time_ms = 0.0
            hint, retry_hint = infer_hint(task, request.sql_query, reward, str(exc))
            feedback = make_feedback(
                status="error",
                error=str(exc),
                expected_rows=len(expected_rows),
                returned_rows=0,
                hint=hint,
                retry_hint=retry_hint,
                execution_time_ms=execution_time_ms,
                expected_execution_time_ms=expected_execution_time_ms,
            )

        record.done = reward == 1.0 or record.steps_taken >= MAX_STEPS
        add_history_entry(record, request.sql_query, reward, feedback)
        logger.info(
            "Processed step",
            extra={
                "session_id": record.session_id,
                "task_id": task.task_id,
                "step": record.steps_taken,
                "reward": reward,
                "done": record.done,
            },
        )

        return StepResponse(
            observation=StepObservation(
                submitted_query=request.sql_query,
                feedback=feedback,
            ),
            reward=reward,
            done=record.done,
        )


@app.get("/state", response_model=StateResponse)
def get_state(session_id: str = Query(..., description="Session identifier returned by /reset")) -> StateResponse:
    record = fetch_session(session_id)
    with record.lock:
        return record.to_state()
