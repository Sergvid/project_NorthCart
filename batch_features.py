from datetime import datetime
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from airflow import DAG
from airflow.decorators import task
from airflow.hooks.base import BaseHook
from airflow.models import Variable
from airflow.providers.postgres.hooks.postgres import PostgresHook
from botocore.exceptions import ClientError

from calculate_batch_features import (
    build_batch_features,
    load_source_tables,
    normalize_run_date,
    save_features,
)


# Имя DAG в интерфейсе Airflow
DAG_ID = "batch_features"

# Соединения, созданные в Airflow
POSTGRES_CONN_ID = "postgres_raw"
S3_CONN_ID = "aws_default"

DEFAULT_SOURCE_SCHEMA = "public"
DEFAULT_OUTPUT_DIR = "/tmp/batch_features"



def get_postgres_uri() -> str:
    """Получает параметры подключения из Airflow Connection."""
    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    return hook.get_uri()


def get_s3_client_and_bucket() -> tuple[Any, str]:
    """Создаёт S3-клиент из параметров Airflow Connection."""
    connection = BaseHook.get_connection(S3_CONN_ID)
    extras = connection.extra_dejson

    required = [
        "bucket",
        "endpoint_url",
        "aws_access_key_id",
        "aws_secret_access_key",
    ]

    missing = [field for field in required if not extras.get(field)]
    if missing:
        raise ValueError(
            f"Не заполнены поля Connection {S3_CONN_ID}: {missing}"
        )

    client = boto3.client(
        "s3",
        aws_access_key_id=extras["aws_access_key_id"],
        aws_secret_access_key=extras["aws_secret_access_key"],
        endpoint_url=extras["endpoint_url"],
        region_name="ru-central1",
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
        ),
    )

    return client, extras["bucket"]


with DAG(
    dag_id=DAG_ID,
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["batch-features"],
) as dag:

    @task(task_id="build_and_upload_features")
    def build_and_upload_features() -> dict[str, str | int]:
        """Читает данные, вызывает расчёт и загружает CSV в S3."""
        run_date = normalize_run_date(
            Variable.get("batch_features_run_date")
        )

        # В проекте используем срезы на начало дня.
        if run_date != run_date.normalize():
            raise ValueError("run_date должна задавать начало дня в UTC")

        date_label = run_date.strftime("%Y-%m-%d")

        tables = load_source_tables(
            postgres_uri=get_postgres_uri(),
            run_date=run_date,
            schema=DEFAULT_SOURCE_SCHEMA,
        )

        features = build_batch_features(
            tables=tables,
            run_date=run_date,
        )

        local_path = (
            Path(DEFAULT_OUTPUT_DIR)
            / f"run_date={date_label}"
            / "batch_features.csv"
        )
        save_features(features, local_path)

        s3_client, s3_bucket = get_s3_client_and_bucket()
        s3_key = f"run_date={date_label}/batch_features.csv"

        s3_client.upload_file(
            str(local_path),
            s3_bucket,
            s3_key,
        )

        return {
            "run_date": date_label,
            "rows": len(features),
            "s3_bucket": s3_bucket,
            "s3_key": s3_key,
        }

    @task(task_id="validate_saved_result")
    def validate_saved_result(
        result_info: dict[str, str | int],
    ) -> None:
        """Проверяет наличие и ненулевой размер файла в S3."""
        if int(result_info["rows"]) <= 0:
            raise ValueError("Таблица признаков пуста")

        s3_client, _ = get_s3_client_and_bucket()
        s3_bucket = str(result_info["s3_bucket"])
        s3_key = str(result_info["s3_key"])

        try:
            metadata = s3_client.head_object(
                Bucket=s3_bucket,
                Key=s3_key,
            )
        except ClientError as error:
            error_code = error.response.get("Error", {}).get("Code")

            if error_code in {"404", "NoSuchKey", "NotFound"}:
                raise FileNotFoundError(
                    f"Файл не найден: s3://{s3_bucket}/{s3_key}"
                ) from error
            raise

        if int(metadata.get("ContentLength", 0)) <= 0:
            raise ValueError(
                f"Файл пуст: s3://{s3_bucket}/{s3_key}"
            )

    validate_saved_result(build_and_upload_features())
