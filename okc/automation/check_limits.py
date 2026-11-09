import subprocess, json

probs = [
    "muse_bf16_proj",
    "muse_attention_decode",
    "muse_moe_w4a16",
    "muse_decode_layer",
    "muse_prefill_layer",
]
for prob in probs:
    cmd = [
        "wsl",
        "-d",
        "Ubuntu",
        "-e",
        "bash",
        "-c",
        f'curl -s --max-time 15 -H "Authorization: Bearer kc_w0vq-VLPsprUFOg4JSXZUjv8ni6QUZvmx6TF-DeXiQk" https://openkernelchallenge.dev/api/v1/problems/{prob}',
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    try:
        d = json.loads(r.stdout)
        limits = d.get("limits", {})
        scored = limits.get("scored", {})
        print(
            f"{prob}: used={scored.get('used', '?')}, left={scored.get('left', '?')}, per_day={scored.get('per_day', '?')}"
        )
    except:
        print(f"{prob}: error")
