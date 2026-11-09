import subprocess


# Helper to submit a variant
def submit_variant(prob, content, filename):
    dst = f"C:/Users/Admin/{filename}"
    with open(dst, "w") as f:
        f.write(content)
    cmd = [
        "wsl",
        "-d",
        "Ubuntu",
        "-e",
        "bash",
        "-c",
        f"cp /mnt/c/Users/Admin/{filename} ~/okc/{prob}/submission.py && cd ~/okc && ~/.okc/bin/okc submit {prob} {prob}/submission.py --no-wait --json",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    print(
        f"{filename}: {r.stdout.strip()} {r.stderr.strip()[:100] if r.stderr else ''}"
    )


# muse_attention_decode variants with TILE_KEYS=512, 1024
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

for tk in [512, 1024]:
    modified = content.replace("TILE_KEYS = 256", f"TILE_KEYS = {tk}")
    submit_variant("muse_attention_decode", modified, f"attn_tk{tk}.py")

# muse_moe_w4a16 variants with CHUNK_TOKENS=2048, 4096
cmd = [
    "wsl",
    "-d",
    "Ubuntu",
    "-e",
    "bash",
    "-c",
    "cat ~/okc/muse_moe_w4a16/submission.py",
]
r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
content = r.stdout

for ct in [2048, 4096]:
    modified = content.replace("CHUNK_TOKENS = 1024", f"CHUNK_TOKENS = {ct}")
    submit_variant("muse_moe_w4a16", modified, f"moe_chunk{ct}.py")

# muse_decode_layer variants with PDL_MAX_T=192, 256
cmd = [
    "wsl",
    "-d",
    "Ubuntu",
    "-e",
    "bash",
    "-c",
    "cat ~/okc/muse_decode_layer/submission.py",
]
r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
content = r.stdout

for pdl in [192, 256]:
    modified = content.replace("PDL_MAX_T = 96", f"PDL_MAX_T = {pdl}")
    submit_variant("muse_decode_layer", modified, f"decode_pdl{pdl}.py")

# muse_prefill_layer variants with MOE_FAST_SCALE=False, True
cmd = [
    "wsl",
    "-d",
    "Ubuntu",
    "-e",
    "bash",
    "-c",
    "cat ~/okc/muse_prefill_layer/submission.py",
]
r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
content = r.stdout

for fs in [False, True]:
    modified = content.replace("MOE_FAST_SCALE = True", f"MOE_FAST_SCALE = {fs}")
    submit_variant("muse_prefill_layer", modified, f"prefill_fast{int(fs)}.py")

print("All variants submitted!")
