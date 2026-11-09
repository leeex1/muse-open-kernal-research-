cd ~/okc
for p in muse_bf16_proj muse_attention_decode muse_moe_w4a16 muse_decode_layer muse_prefill_layer; do
    echo "=== $p ==="
    ~/.okc/bin/okc leaderboard "$p" 2>&1 | head -60
    echo
done
