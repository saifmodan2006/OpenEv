import json
import os
import re
from typing import Any

import httpx
from openai import OpenAI


MAX_STEPS = 5
SUCCESS_THRESHOLD = 0.8
PROMPT_TEMPLATE = """You are a SQL expert.

Task:
{task_description}

Schema:
{schema}

Previous attempts:
{history}

Feedback:
{feedback}

Generate ONLY a valid SQLite SELECT query.
Do not include explanation."""


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(value, upper))


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def build_openai_client() -> OpenAI:
    api_key = os.getenv("HF_TOKEN", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing required environment variable: HF_TOKEN")

    base_url = os.getenv("API_BASE_URL", "").strip()
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs)


def normalize_model_output(content: str) -> str:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:sql)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def format_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return "None"

    lines = []
    for item in history:
        lines.append(
            f"step={item['step']} query={item['sql_query']} reward={item['reward']} hint={item['hint']}"
        )
    return "\n".join(lines)


def compact_sql(sql_query: str) -> str:
    return " ".join(sql_query.split())


def generate_sql_query(
    openai_client: OpenAI,
    *,
    model_name: str,
    task_description: str,
    schema: str,
    history: list[dict[str, Any]],
    feedback: dict[str, Any] | str,
) -> str:
    prompt = PROMPT_TEMPLATE.format(
        task_description=task_description,
        schema=schema,
        history=format_history(history),
        feedback=feedback if isinstance(feedback, str) else json.dumps(feedback, ensure_ascii=True),
    )
    response = openai_client.chat.completions.create(
        model=model_name,
        temperature=0,
        max_tokens=200,
        messages=[
            {
                "role": "system",
                "content": "Return only a SQLite SELECT query with no markdown or explanation.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    message = response.choices[0].message.content or ""
    sql_query = normalize_model_output(message)
    if not sql_query:
        raise RuntimeError("Model returned an empty response")
    return sql_query


def main() -> None:
    env_url = require_env("ENV_URL").rstrip("/")
    model_name = require_env("MODEL_NAME")
    openai_client = build_openai_client()

    timeout = httpx.Timeout(60.0, connect=20.0)
    overall_rewards: list[float] = []

    with httpx.Client(timeout=timeout) as client:
        tasks_response = client.get(f"{env_url}/tasks")
        tasks_response.raise_for_status()
        tasks = tasks_response.json()["tasks"]

        for task in tasks:
            task_id = task["task_id"]
            reset_response = client.post(f"{env_url}/reset", json={"task_id": task_id})
            reset_response.raise_for_status()
            reset_payload = reset_response.json()

            observation = reset_payload["observation"]
            session_id = reset_payload["session_id"]
            schema_text = (
                f"{observation['schema']}\n\nSchema visualization:\n{observation['schema_visualization']}"
            )
            history: list[dict[str, Any]] = []
            latest_feedback: dict[str, Any] | str = "None"
            step_rewards: list[float] = []
            task_success = False

            print(f"[START] task={task_id} env={env_url} model={model_name}")

            for step_number in range(1, MAX_STEPS + 1):
                sql_query = generate_sql_query(
                    openai_client,
                    model_name=model_name,
                    task_description=observation["task_description"],
                    schema=schema_text,
                    history=history,
                    feedback=latest_feedback,
                )

                step_response = client.post(
                    f"{env_url}/step",
                    json={"session_id": session_id, "sql_query": sql_query},
                )
                step_response.raise_for_status()
                step_payload = step_response.json()

                reward = float(step_payload.get("reward", 0.0) or 0.0)
                done = bool(step_payload["done"])
                feedback = step_payload["observation"]["feedback"]
                error = feedback.get("error")

                print(
                    f"[STEP] step={step_number} action={compact_sql(sql_query)} "
                    f"reward={reward:.1f} done={str(done).lower()} error={error or 'None'}"
                )

                history.append(
                    {
                        "step": step_number,
                        "sql_query": sql_query,
                        "reward": reward,
                        "hint": feedback.get("hint", ""),
                    }
                )
                latest_feedback = feedback
                step_rewards.append(reward)

                if reward == 1.0:
                    task_success = True

                if done:
                    break

            task_score = max(step_rewards, default=0.0)
            overall_rewards.append(task_score)
            print(
                f"[END] success={str(task_success).lower()} "
                f"steps={len(step_rewards)} score={task_score:.2f} rewards={step_rewards}"
            )

    max_total_reward = float(len(overall_rewards) or 1)
    final_score = clamp(sum(overall_rewards) / max_total_reward)
    overall_success = final_score >= SUCCESS_THRESHOLD
    print(
        f"Final score={final_score:.2f} success={str(overall_success).lower()} "
        f"threshold={SUCCESS_THRESHOLD:.1f}"
    )


if __name__ == "__main__":
    main()
