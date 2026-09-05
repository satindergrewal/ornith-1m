#!/usr/bin/env python3
"""Bounded bench for Mia single-Spark Qwen3.8-Flash-Next NVFP4 on :8888.
No NIAH (standing order). Measures: smoke, single decode t/s, conc-4 agg,
prefill @~32k, vision smoke. Prints a compact receipt."""
import base64
import json
import struct
import time
import urllib.request
import zlib

URL = "http://127.0.0.1:8888/v1/chat/completions"
MODEL = "qwen3.8-flash-next"


def post(payload, timeout=600):
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return body, time.time() - t0


def chat(messages, max_tokens=512, timeout=600):
    body, wall = post({"model": MODEL, "messages": messages,
                       "max_tokens": max_tokens, "temperature": 0.6,
                       "top_p": 0.95, "stream": False}, timeout)
    m = body["choices"][0]["message"]
    txt = m.get("content") or ""
    reas = m.get("reasoning_content") or m.get("reasoning") or ""
    u = body["usage"]
    return {"wall": round(wall, 2), "tok": u["completion_tokens"],
            "tps": round(u["completion_tokens"] / wall, 1),
            "text": txt, "reason": reas, "prompt_tok": u["prompt_tokens"]}


print("== 1. smoke ==")
r = chat([{"role": "user", "content": "What is 17*23? Answer with just the number."}], 300)
print(json.dumps({k: r[k] for k in ("wall", "tok", "tps")}), "|", (r["text"] or r["reason"])[:80].replace("\n", " "))

print("== 2. single-stream prose decode (512 tok) ==")
r = chat([{"role": "user", "content": "Write a vivid 400-word essay about tides."}], 512)
print(json.dumps({k: r[k] for k in ("wall", "tok", "tps")}), "|", r["text"][:60].replace("\n", " "))

print("== 3. concurrency 4 (aggregate) ==")
import threading
results = [None] * 4
def one(i):
    results[i] = chat([{"role": "user",
                        "content": f"Write a 350-word story about a lighthouse keeper, variant {i}."}], 448)
t0 = time.time()
threads = [threading.Thread(target=one, args=(i,)) for i in range(4)]
[t.start() for t in threads]
[t.join() for t in threads]
wall = time.time() - t0
tot = sum(x["tok"] for x in results)
print(f"wall {wall:.1f}s | total {tot} tok | agg {tot/wall:.1f} t/s | per-stream {[x['tps'] for x in results]}")

print("== 4. prefill @~32k ==")
filler = ("The harbor log records weather, tide, and cargo for the day. " * 850)
msgs = [{"role": "user", "content": filler + "\n\nHow many times does the word 'tide' appear above? Think briefly, then answer with just the count."}]
r = chat(msgs, 200, timeout=900)
print(json.dumps({k: r[k] for k in ("wall", "tok", "tps", "prompt_tok")}), "|", (r["text"] or r["reason"])[:100].replace("\n", " "))

print("== 5. vision smoke ==")
def tiny_png():
    def chunk(t, d):
        c = t + d
        return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", 16, 16, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x30\x60\xff" * 16 for _ in range(16))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) +
            chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
b64 = base64.b64encode(tiny_png()).decode()
r = chat([{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
    {"type": "text", "text": "What dominant color is this image? One word."}]}], 200)
print(json.dumps({k: r[k] for k in ("wall", "tok", "tps")}), "|", (r["text"] or r["reason"])[:60].replace("\n", " "))

print("BENCH-DONE")
