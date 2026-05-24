#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from itertools import islice
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatlm_encoder.config import load_config
from chatlm_encoder.data import iter_texts
from chatlm_encoder.preprocess import ensure_preprocessed_data, preprocessed_data_config
from chatlm_encoder.tokenizer import load_tokenizer

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **_: Any):
        return iterable


def default_output_path(config: dict[str, Any], dtype_name: str) -> Path:
    token_ids_path = config["data"].get("token_ids_path")
    if token_ids_path:
        return Path(token_ids_path).expanduser()

    preprocess_path = Path(config["preprocess"]["output_path"]).expanduser()
    return preprocess_path.with_suffix(f".tokens.{dtype_name}.bin")


def resolve_dtype(dtype_name: str, tokenizer_size: int):
    import numpy as np

    if dtype_name == "auto":
        dtype_name = "uint16" if tokenizer_size <= int(np.iinfo(np.uint16).max) else "uint32"
    dtype = np.dtype(dtype_name)
    if dtype.kind != "u":
        raise ValueError("Packed token dtype must be an unsigned integer dtype such as uint16 or uint32.")
    if tokenizer_size - 1 > int(np.iinfo(dtype).max):
        raise ValueError(f"Tokenizer size {tokenizer_size} does not fit in dtype {dtype}.")
    return dtype


def write_buffer(handle, buffer: list[int], dtype) -> int:
    if not buffer:
        return 0

    import numpy as np

    array = np.asarray(buffer, dtype=dtype)
    array.tofile(handle)
    written = int(array.size)
    buffer.clear()
    return written


def packing_layout(config: dict[str, Any], data_config: dict[str, Any], tokenizer) -> tuple[int | None, int, int]:
    add_cls = bool(data_config.get("add_cls", True))
    cls_token_id = tokenizer.cls_token_id if add_cls else None
    if add_cls and cls_token_id is None:
        raise ValueError("data.add_cls is true, but the tokenizer does not define cls_token_id.")

    block_size = int(config["model"]["block_size"])
    payload_tokens_per_block = block_size - (1 if cls_token_id is not None else 0)
    if payload_tokens_per_block < 1:
        raise ValueError("model.block_size must leave room for at least one non-CLS token.")
    return int(cls_token_id) if cls_token_id is not None else None, block_size, payload_tokens_per_block


def count_packed_tokens(path: Path, dtype) -> int:
    byte_size = path.stat().st_size
    if byte_size % int(dtype.itemsize) != 0:
        raise ValueError(f"Packed token file size is not divisible by dtype item size: {path}")
    return byte_size // int(dtype.itemsize)


def read_existing_rows(manifest_path: Path) -> int | None:
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle).get("rows")
    except Exception:
        return None
    try:
        return int(rows)
    except (TypeError, ValueError):
        return None


def write_manifest(
    manifest_path: Path,
    output_path: Path,
    dtype,
    rows: int | None,
    tokens: int,
    block_size: int,
    cls_token_id: int | None,
    payload_tokens_per_block: int,
    config: dict[str, Any],
) -> None:
    manifest = {
        "output_path": str(output_path),
        "manifest_path": str(manifest_path),
        "dtype": dtype.name,
        "rows": rows,
        "tokens": tokens,
        "block_size": block_size,
        "add_cls": bool(cls_token_id is not None),
        "cls_token_id": cls_token_id,
        "payload_tokens_per_block": payload_tokens_per_block,
        "full_blocks": tokens // payload_tokens_per_block,
        "dropped_remainder_tokens_per_epoch": tokens % payload_tokens_per_block,
        "tokenizer_path": str(config["tokenizer"]["path"]),
        "config_path": str(config["_config_path"]),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack normalized text into a memory-mapped token id binary.")
    parser.add_argument("--config", default="configs/h20_8gpu_bert_0p2b_deepspeed.yaml", help="Path to a YAML config.")
    parser.add_argument("--output", default=None, help="Output .bin path. Defaults to data.token_ids_path.")
    parser.add_argument("--dtype", default=None, help="uint16, uint32, or auto. Defaults to data.token_ids_dtype.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing packed token file.")
    parser.add_argument("--force-prepare", action="store_true", help="Rebuild normalized JSONL before packing tokens.")
    parser.add_argument("--force-download", action="store_true", help="Refresh Hugging Face snapshots before preprocessing.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional text-row cap for a quick debug pack.")
    parser.add_argument("--flush-tokens", type=int, default=10_000_000, help="Token buffer size before writing to disk.")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_preprocessed_data(config, force=args.force_prepare, force_download=args.force_download)

    tokenizer = load_tokenizer(config["tokenizer"]["path"])
    dtype_name = str(args.dtype or config["data"].get("token_ids_dtype") or "auto")
    dtype = resolve_dtype(dtype_name, tokenizer_size=len(tokenizer))
    output_path = Path(args.output).expanduser() if args.output else default_output_path(config, dtype.name)
    manifest_path = Path(config["data"].get("token_ids_manifest_path") or f"{output_path}.manifest.json").expanduser()
    data_config = preprocessed_data_config(config)
    cls_token_id, block_size, payload_tokens_per_block = packing_layout(config, data_config, tokenizer)

    if output_path.exists() and not args.force:
        tokens = count_packed_tokens(output_path, dtype)
        write_manifest(
            manifest_path=manifest_path,
            output_path=output_path,
            dtype=dtype,
            rows=read_existing_rows(manifest_path),
            tokens=tokens,
            block_size=block_size,
            cls_token_id=cls_token_id,
            payload_tokens_per_block=payload_tokens_per_block,
            config=config,
        )
        print(f"Packed token ids already exist: {output_path}")
        print(f"Refreshed manifest for {block_size}-token blocks with {payload_tokens_per_block} payload tokens.")
        print("Use --force to rebuild the token file.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    texts = iter_texts(data_config)
    if args.max_samples is not None:
        texts = islice(texts, int(args.max_samples))

    eos_id = tokenizer.eos_token_id
    add_eos = bool(data_config.get("add_eos", True))
    buffer: list[int] = []
    rows = 0
    tokens = 0

    with tmp_path.open("wb") as handle:
        for text in tqdm(texts, desc="pack tokens", unit="row"):
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if add_eos and eos_id is not None and (not ids or ids[-1] != eos_id):
                ids.append(eos_id)
            buffer.extend(int(token_id) for token_id in ids)
            rows += 1

            if len(buffer) >= int(args.flush_tokens):
                tokens += write_buffer(handle, buffer, dtype)

        tokens += write_buffer(handle, buffer, dtype)

    tmp_path.replace(output_path)
    write_manifest(
        manifest_path=manifest_path,
        output_path=output_path,
        dtype=dtype,
        rows=rows,
        tokens=tokens,
        block_size=block_size,
        cls_token_id=cls_token_id,
        payload_tokens_per_block=payload_tokens_per_block,
        config=config,
    )

    print(f"Packed token ids: {output_path}")
    print(f"Rows: {rows:,}")
    print(f"Tokens: {tokens:,}")
    print(f"Full {block_size}-token blocks: {tokens // payload_tokens_per_block:,}")
    if cls_token_id is not None:
        print(f"Training will prepend CLS token id {int(cls_token_id)} to each block.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
