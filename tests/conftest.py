import os
from pathlib import Path

import boto3
import pooch
import pytest


def _r2_downloader(url: str, output_file, _) -> None:
    """Fetch a private-R2 object via boto3.

    Required env vars:
        R2_ACCOUNT_ID, R2_BUCKET, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
    """
    account_id = os.environ["R2_ACCOUNT_ID"]
    bucket = os.environ["R2_BUCKET"]
    key = url.rsplit("/", 1)[-1]

    client = boto3.client(
        service_name="s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    if hasattr(output_file, "write"):
        client.download_fileobj(bucket, key, output_file)
    else:
        client.download_file(bucket, key, str(output_file))


_ARCHIVE = pooch.create(
    path=pooch.os_cache("pymps-tests"),
    base_url="r2://pymps/",
    registry={"maros-meszaros.zip": None},
)


@pytest.fixture(scope="session")
def problem_paths() -> dict[str, str]:
    files = _ARCHIVE.fetch(
        "maros-meszaros.zip",
        processor=pooch.Unzip(),
        downloader=_r2_downloader,
    )
    return {Path(f).stem.lower().replace("_", ""): f for f in files}
