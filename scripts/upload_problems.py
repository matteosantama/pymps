"""Zip MPS/QPS files from a directory and upload to a Cloudflare R2 bucket.

Required environment variables:
    R2_ACCOUNT_ID          Cloudflare account ID
    R2_ACCESS_KEY_ID       R2 access key
    R2_SECRET_ACCESS_KEY   R2 secret key
"""

import argparse
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import boto3

EXTENSIONS = {".mps", ".qps"}


def collect_files(directory: Path) -> list[Path]:
    return sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )


def build_zip(files: list[Path], output: Path) -> None:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=f.name)


def upload(zip_path: Path, bucket: str, key: str) -> None:
    account_id = os.environ["R2_ACCOUNT_ID"]
    client = boto3.client(
        service_name="s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    client.upload_file(str(zip_path), bucket, key)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory",
        type=Path,
        help="directory containing .mps/.qps files (searched recursively)",
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("R2_BUCKET"),
        help="R2 bucket name (env: R2_BUCKET)",
    )
    parser.add_argument("--key", required=True, help="R2 object key")
    parser.add_argument(
        "--dry-run", action="store_true", help="build the zip but skip upload"
    )
    args = parser.parse_args()

    if not args.directory.is_dir():
        sys.exit(f"error: {args.directory} is not a directory")

    if not args.bucket:
        sys.exit("error: --bucket or $R2_BUCKET is required")

    if not args.key.endswith(".zip"):
        sys.exit("error: --key should end with '.zip'")

    files = collect_files(args.directory)
    if not files:
        sys.exit(f"error: no .mps or .qps files found in {args.directory}")

    with tempfile.TemporaryDirectory(prefix="pymps-upload-") as tmpdir:
        zip_path = Path(tmpdir) / Path(args.key).name
        print(f"zipping {len(files)} files -> {zip_path}")
        build_zip(files, zip_path)

        if args.dry_run:
            print("dry-run: skipping upload")
            return 0

        print(f"uploading -> r2://{args.bucket}/{args.key}")
        upload(zip_path, args.bucket, args.key)

    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
