"""
import torch,gc
from torch.utils.data import Dataset,DataLoader
from architecture.tokenizer import get_tokenizer
from datasets import load_dataset
from tqdm.notebook import tqdm
batch_size=5
context_len=4000

dataset = load_dataset("roneneldan/TinyStories")
train_text = " ".join([ex["text"] for ex in dataset['train']])
val_text = " ".join([ex["text"] for ex in dataset['validation']])

tokenizer = get_tokenizer()
print("tokenizing...")
train_tokens = tokenizer.encode(train_text)
val_tokens = tokenizer.encode(val_text)
print("tokenized")
class TextDataset(Dataset):
    def __init__(self, tokens, max_length=8192, stride=8192):
        self.input_ids = []
        self.target_ids = []
        for i in tqdm(range(0, len(tokens) - max_length, stride)):
            input_chunk = tokens[i:i + max_length]
            target_chunk = tokens[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

train_dataset = TextDataset(train_tokens, max_length=context_len, stride=context_len)
val_dataset = TextDataset(val_tokens, max_length=context_len, stride=context_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

del dataset, train_text, val_text
gc.collect()
"""

import os, gc, torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from datasets import load_dataset
from architecture.tokenizer import get_tokenizer

# ====== Config (env-overridable) ======
BATCH_SIZE   = int(os.getenv("BATCH_SIZE", "5"))
CONTEXT_LEN  = int(os.getenv("CONTEXT_LEN", "2048"))   # 2048 is plenty; raise if you have VRAM
STRIDE       = int(os.getenv("STRIDE", str(CONTEXT_LEN)))  # set < CONTEXT_LEN for overlap
NUM_WORKERS  = int(os.getenv("NUM_WORKERS", "4"))
PIN_MEMORY   = os.getenv("PIN_MEMORY", "1") == "1"
SPLIT        = os.getenv("SPLIT", "train")  # dataset split

# ====== Pick your dataset here ======
HF_DATASET = "MohamedSaeed-dev/python-text-to-code"

# ====== Helpers ======
def decode_code_placeholders(code: str) -> str:
    """
    Convert placeholder tokens to real Python formatting.
    The dataset often uses NEW_LINE / INDENT / DEDENT markers.
    - We treat INDENT/DEDENT approximately with 4-space indentation levels.
    """
    # Quick pass: normalize spacing around markers
    s = code.replace("NEW_LINE", "\n")
    # crude indentation handling:
    # Convert token stream like "INDENT" / "DEDENT" to spaces based on a stack depth.
    lines = []
    indent = 0
    tokens = s.replace("\t", "    ").split()
    reconstructed = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "INDENT":
            indent += 1
            i += 1
            continue
        if tok == "DEDENT":
            indent = max(0, indent - 1)
            i += 1
            continue
        # Start a new line token
        if tok == "\n":
            reconstructed.append("\n" + ("    " * indent))
        else:
            # add token with a space if needed
            if reconstructed and not reconstructed[-1].endswith(("\n", " ")):
                reconstructed.append(" ")
            reconstructed.append(tok)
        i += 1
    s = "".join(reconstructed)

    # After first pass, ensure new lines get the current indent
    final_lines = []
    for line in s.splitlines():
        if line.strip() == "":
            final_lines.append("")
        else:
            # make sure each non-empty line is prefixed with proper indentation
            stripped = line.lstrip()
            leading = "    " * indent if line.startswith("\n") else ""
            final_lines.append(leading + stripped)
    return "\n".join(final_lines)

def format_example(text: str, code: str) -> str:
    """
    Build a single training sample string: prompt + solution.
    No need to add special tokens to the tokenizer; these tags are just text.
    """
    code_decoded = decode_code_placeholders(code or "")
    text = (text or "").strip()
    return (
        "<|bos|>\n"
        "### Task\n"
        f"{text}\n\n"
        "### Solution (Python)\n"
        f"{code_decoded}\n"
        "<|eos|>"
    )

# ====== Load dataset ======
# (If RAM is tight, consider streaming=True and implement an IterableDataset.)
dataset = load_dataset(HF_DATASET, split=SPLIT)  # 'train' split is ~9k rows

tokenizer = get_tokenizer()

def tokenize_all():
    toks = []
    for ex in tqdm(dataset, desc=f"Tokenizing {HF_DATASET}/{SPLIT}"):
        sample = format_example(ex.get("text", ""), ex.get("code", ""))
        toks.extend(tokenizer.encode(sample))
    return torch.tensor(toks, dtype=torch.long)

print("Tokenizing...")
all_tokens = tokenize_all()
print("Done tokenizing. Total tokens:", len(all_tokens))

class TextDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, max_length: int, stride: int):
        assert tokens.dtype == torch.long
        self.input_ids = []
        self.target_ids = []

        end = len(tokens) - max_length - 1
        if end < 0:
            return

        for i in tqdm(range(0, end + 1, stride), desc="Chunking"):
            inp  = tokens[i : i + max_length]
            tgt  = tokens[i + 1 : i + max_length + 1]

            # If you want to train ONLY on the code portion, you can build a mask
            # that sets prompt positions in `tgt` to -100 (ignored by CE).
            # This requires you to change your trainer loss to:
            #   F.cross_entropy(logits, tgt, ignore_index=-100)
            #
            # For the default (predict everything), keep as-is:
            self.input_ids.append(inp)
            self.target_ids.append(tgt)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# Build datasets/loaders (single split used for both train/val here; adjust as needed)
# If you have a separate validation split, load that split and tokenize separately.
train_dataset = TextDataset(all_tokens, max_length=CONTEXT_LEN, stride=STRIDE)
val_dataset   = TextDataset(all_tokens, max_length=CONTEXT_LEN, stride=STRIDE)  # simple: same data

persistent = NUM_WORKERS > 0
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=persistent,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=persistent,
)

del dataset, all_tokens
gc.collect()
