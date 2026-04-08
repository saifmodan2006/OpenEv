# SQL Query Generator

## Project Overview

SQL Query Generator is an OpenEnv RL environment for benchmarking agents that translate natural language analytics requests into SQLite queries. Each session creates a deterministic in-memory database, evaluates the submitted SQL against a ground-truth query, and returns structured feedback with reward signals, execution timing, and retry guidance.

The environment is designed for the OpenEnv Hackathon Round 1 workflow:

- deterministic task definitions
- reproducible SQLite data
- strict read-only SQL validation
- session-based grading over multiple attempts
- OpenAI-compatible inference script for rollout evaluation

## Task Descriptions

### `task_easy`
Find all users older than 25.

Target skills:
- `SELECT`
- `WHERE`

### `task_medium`
Find total quantity ordered per product with product name, sorted descending.

Target skills:
- `JOIN`
- `GROUP BY`
- `ORDER BY`

### `task_hard`
Find users who ordered Electronics but NEVER ordered Clothing.

Target skills:
- subquery
- exclusion logic with `NOT IN` or `NOT EXISTS`
- multi-table joins

## API Documentation

### `POST /reset`
Starts a new session for a task.

Request body:

```json
{
  "task_id": "task_easy"
}
```

Response body:

```json
{
  "session_id": "uuid",
  "observation": {
    "task_description": "Find all users older than 25",
    "schema": "users/products/orders schema text",
    "schema_visualization": "ASCII schema diagram",
    "difficulty": "easy"
  },
  "reward": 0.0,
  "done": false
}
```

### `POST /step`
Submits a SQL query for the active session.

Request body:

```json
{
  "session_id": "uuid",
  "sql_query": "SELECT id, name FROM users WHERE age > 25;"
}
```

Response body:

```json
{
  "observation": {
    "submitted_query": "SELECT id, name FROM users WHERE age > 25;",
    "feedback": {
      "status": "success",
      "error": null,
      "expected_rows": 4,
      "returned_rows": 4,
      "hint": "Query matched the expected result set.",
      "retry_hint": null,
      "execution_time_ms": 0.182,
      "expected_execution_time_ms": 0.071
    }
  },
  "reward": 1.0,
  "done": true
}
```

Reward rules:

- exact match: `1.0`
- subset match: `0.6`
- partial overlap: `0.3`
- wrong result or SQL error: `0.0`

Session termination rules:

- end immediately on `reward == 1.0`
- end after `steps >= 5`

Validation rules:

- only `SELECT` and read-only `WITH ... SELECT ...` statements are accepted
- queries containing `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, or other mutating keywords are rejected

### `GET /state?session_id=<uuid>`
Returns the current session state, including task metadata, history, total steps, and done flag.

### `GET /tasks`
Returns the full task catalog.

### `GET /schema`
Returns the schema text, ASCII visualization, allowed query type, and max step count.

### `GET /health`
Basic health check.

## Setup Instructions

### Local Python setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run Instructions

### Start the environment server

```bash
uvicorn main:app --host 0.0.0.0 --port 7860
```

### Run with Docker

```bash
docker build -t sql-query-generator .
docker run --rm -p 7860:7860 sql-query-generator
```

### Execute the inference script

Required environment variables:

- `ENV_URL`: environment base URL, for example `http://localhost:7860`
- `MODEL_NAME`: model identifier exposed by the OpenAI-compatible API
- `HF_TOKEN`: API key used by the OpenAI client
- `API_BASE_URL`: OpenAI-compatible base URL, for example `https://router.huggingface.co/v1`

Example:

```bash
export ENV_URL=http://localhost:7860
export MODEL_NAME=gpt-4.1-mini
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
python inference.py
```

The rollout script:

- iterates through every task
- maintains attempt history and feedback context
- logs each step in the required format
- computes final score as `sum(task_rewards) / max_total_reward`
- clamps the final score to `[0.0, 1.0]`
- uses a success threshold of `0.8`

## Notes

- The SQLite dataset is recreated per session in memory.
- Query timing is tracked for both the submitted SQL and ground-truth SQL.
- The grader is deterministic because the schema, seed data, and result comparison logic are fixed.
