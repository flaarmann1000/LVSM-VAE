import torch
from pathlib import Path
from tqdm import tqdm

# ===============================
# CONFIG
# ===============================
DATA_ROOT = Path("./data/re10k_subset")
TRAIN_ROOT = DATA_ROOT / "train"
TEST_ROOT  = DATA_ROOT / "test"

N_FILES = 1  # number of .torch files to take from train/test


DST_TRAIN_FILE = DATA_ROOT / f"train-overfit-{N_FILES}/00000.torch"
DST_TEST_FILE  = DATA_ROOT / f"test-overfit-{N_FILES}/00000.torch"

# ===============================
# HELPER
# ===============================
def load_torch_files(root: Path, n_files: int):
    files = sorted(root.glob("*.torch"))[:1]
    merged = []
    for f in tqdm(files, desc=f"Processing {root.name}"):
        data = torch.load(f)[:n_files]
        merged.extend(data)
    return merged

# ===============================
# LOAD AND SAVE
# ===============================
train_data = load_torch_files(TRAIN_ROOT, N_FILES)
test_data  = load_torch_files(TEST_ROOT, N_FILES)

DST_TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
DST_TEST_FILE.parent.mkdir(parents=True, exist_ok=True)

torch.save(train_data, DST_TRAIN_FILE)
torch.save(test_data, DST_TEST_FILE)

print(f"Saved {len(train_data)} examples from {N_FILES} train files → {DST_TRAIN_FILE}")
print(f"Saved {len(test_data)} examples from {N_FILES} test files  → {DST_TEST_FILE}")
