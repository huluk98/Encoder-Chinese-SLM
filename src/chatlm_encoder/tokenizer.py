from __future__ import annotations

from pathlib import Path
from typing import Iterable

from transformers import PreTrainedTokenizerFast

PAD_TOKEN = "<|pad|>"
UNK_TOKEN = "<|unk|>"
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
MASK_TOKEN = "<|mask|>"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
SYSTEM_TOKEN = "<|system|>"

SPECIAL_TOKENS = [
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    MASK_TOKEN,
    USER_TOKEN,
    ASSISTANT_TOKEN,
    SYSTEM_TOKEN,
]


def train_tokenizer_from_iterator(
    texts: Iterable[str],
    output_dir: str | Path,
    vocab_size: int = 29298,
    min_frequency: int = 2,
    model_max_length: int = 512,
) -> PreTrainedTokenizerFast:
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=int(vocab_size),
        min_frequency=int(min_frequency),
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        cls_token=BOS_TOKEN,
        sep_token=EOS_TOKEN,
        mask_token=MASK_TOKEN,
        additional_special_tokens=[USER_TOKEN, ASSISTANT_TOKEN, SYSTEM_TOKEN],
        model_max_length=int(model_max_length),
    )
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(output_path)
    return fast


def load_tokenizer(path: str | Path) -> PreTrainedTokenizerFast:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(Path(path).expanduser()))
    additions: dict[str, str | list[str]] = {}

    if tokenizer.pad_token is None:
        additions["pad_token"] = PAD_TOKEN
    if tokenizer.unk_token is None:
        additions["unk_token"] = UNK_TOKEN
    if tokenizer.bos_token is None:
        additions["bos_token"] = BOS_TOKEN
    if tokenizer.eos_token is None:
        additions["eos_token"] = EOS_TOKEN
    if tokenizer.mask_token is None:
        additions["mask_token"] = MASK_TOKEN
    if tokenizer.cls_token is None:
        additions["cls_token"] = BOS_TOKEN
    if tokenizer.sep_token is None:
        additions["sep_token"] = EOS_TOKEN

    special_tokens_map = getattr(tokenizer, "special_tokens_map", {}) or {}
    existing = list(special_tokens_map.get("additional_special_tokens") or [])
    existing_set = set(existing)
    extra = [token for token in [USER_TOKEN, ASSISTANT_TOKEN, SYSTEM_TOKEN] if token not in existing_set]
    if extra:
        additions["additional_special_tokens"] = list(existing) + extra

    if additions:
        tokenizer.add_special_tokens(additions)
    return tokenizer
