import subprocess

# Read muse_attention_decode submission
cmd = [
    "wsl",
    "-d",
    "Ubuntu",
    "-e",
    "bash",
    "-c",
    "cat ~/okc/muse_attention_decode/submission.py",
]
r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
content = r.stdout

# Create variant with TILE_K = 64 instead of 32
modified = content.replace("TILE_K = 32", "TILE_K = 64")
dst = "C:/Users/Admin/muse_attention_decode_tk_tilek64.py"
with open(dst, "w") as f:
    f.write(modified)
print(f"Created {dst} (TILE_K=64)")

# Also try TILE_K = 16
modified = content.replace("TILE_K = 32", "TILE_K = 16")
dst = f"C:/Users/Admin/muse_attention_decode_tk16_tilek.py"
with open(dst, "w") as f:
    f.write(modified)
print(f"Created {dst} (TILE_KEYS=16)")
