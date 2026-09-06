#!/usr/bin/env python3
"""tool_battery_agentic.py - loop detection for TOOL-USING sessions.

Same failure class as loop_battery.py (deep-context repetition attractor),
seen through an agentic lens: the model gets a web_search tool and a
research task at deep context; the attractor then shows up as repeated
identical tool calls, circular search cycles, or the text loops the
text battery already detects. Detection runs on BOTH channels:

  1. tool-call signatures  - exact cycle over the whole call sequence
     (any rotation of >= 4 repeats, any period) plus single-call
     dominance over the recent window
  2. assistant text        - the same word-granularity detectors as
     loop_battery, run over reasoning + content

Verdict per session: TOOL_LOOP, TEXT_LOOP, CLEAN, or ERROR (a harness or
server failure - excluded from loop counts, reported separately, never
silently read as an answer).

The web_search tool is served by this harness over the Brave Search API
(web results only; no browsing). The model never sees or holds the API
key: the harness performs the search and returns result snippets. Key
comes from --brave-key-file or $BRAVE_API_KEY; keep the key file OUTSIDE
this repository (never commit it - a bad 401/403 key aborts the run with
a clear message instead of burning the battery blind).
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loop_battery import build_filler, detect_loop


def load_cases(case_dir):
    path = os.path.join(case_dir, "cases.jsonl")
    return [json.loads(l) for l in open(path) if l.strip()]


BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


class BraveAuthError(Exception):
    """Fatal: the API key is missing/rejected. Aborts the battery."""


class BraveSearch:
    """Throttled, caching web search over the Brave API.

    - every live attempt (success OR failure) consumes a rate-limit slot
    - 429 honors Retry-Once (bounded) then surfaces an error result
    - 401/403 raises BraveAuthError (fail fast, do not burn the battery)
    - successful queries are cached for the whole run (deterministic
      filler means repeat sessions re-ask the same queries)
    """

    def __init__(self, key, count=5, min_interval=1.05, timeout=15):
        self.key = key
        self.count = count
        self.min_interval = min_interval
        self.timeout = timeout
        self._last = 0.0
        self._cache = {}
        self.queries = 0

    def _throttle(self):
        gap = self.min_interval - (time.time() - self._last)
        if gap > 0:
            time.sleep(gap)

    def _fetch(self, q):
        self._throttle()
        url = BRAVE_ENDPOINT + "?" + urllib.parse.urlencode(
            {"q": q, "count": self.count})
        req = urllib.request.Request(url, headers={
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.key,
        })
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read()
                if (r.headers.get("Content-Encoding") or "").lower() == "gzip":
                    import gzip
                    raw = gzip.decompress(raw)
                data = json.loads(raw)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read()[:300].decode("utf-8", "replace")
            except Exception:
                pass
            if e.code in (401, 403):
                raise BraveAuthError(
                    f"Brave rejected the key (HTTP {e.code}): {body[:120]}")
            if e.code == 429:
                ra = e.headers.get("Retry-After")
                try:
                    wait = min(float(ra), 30.0) if ra else 5.0
                except ValueError:
                    wait = 5.0
                time.sleep(wait)
                return None  # one bounded retry
            return {"error": f"search backend HTTP {e.code}: {body[:120]}"}
        except Exception as e:
            return {"error": f"search backend error: {str(e)[:120]}"}
        out = []
        for item in (data.get("web") or {}).get("results", [])[:self.count]:
            out.append({
                "title": (item.get("title") or "")[:200],
                "url": (item.get("url") or "")[:300],
                "snippet": (item.get("description") or "")[:400],
            })
        return {"results": out} if out else {
            "results": [], "note": "no results; vary the query"}

    @staticmethod
    def _fit(payload, budget=6000):
        """Trim results until the serialized payload fits the budget,
        always leaving valid JSON (never cut mid-string)."""
        s = json.dumps(payload)
        while len(s) > budget and payload.get("results"):
            payload["results"] = payload["results"][:-1]
            payload["note"] = "results trimmed to fit"
            s = json.dumps(payload)
        return s

    def search(self, query):
        q = " ".join(query.split())[:400]
        if q in self._cache:
            return self._cache[q], True
        for attempt in (0, 1):  # at most one 429 retry
            self._last = time.time()  # failures consume a slot too
            res = self._fetch(q)
            if res is not None:
                break
        else:
            res = {"error": "search backend rate-limited; vary the query"}
        if "error" not in res:
            self.queries += 1
            self._cache[q] = res
        return res, False


SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the public web. Returns a list of results with "
            "title, url, and a short snippet. Use it to research facts, "
            "advisories, documentation, and current information. Frame "
            "precise queries; if a query returns nothing, reformulate it "
            "rather than repeating it."),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "The search query."},
            },
            "required": ["query"],
        },
    },
}


def call_sig(name, args):
    """Canonical signature of one tool call. String argument values are
    whitespace-normalized so 'CVE  x' / 'CVE x' produce the same
    signature (the search cache already treats them as one query)."""
    if not isinstance(args, dict):
        args = {"__args__": str(args)[:80]}
    norm = {k: (" ".join(v.split()) if isinstance(v, str) else v)
            for k, v in args.items()}
    return f"{name}({json.dumps(norm, sort_keys=True, separators=(',', ':'))})"


def detect_tool_loop(signatures, min_repeats=4, coverage=0.6,
                     dominance_window=24):
    """Loop detection over the tool-call signature sequence.

    1. exact cycle over the WHOLE sequence: any period p with the last
       min_repeats*p calls equal to the last p repeated (catches query
       rotations of any size, not just tiny ones)
    2. single-signature dominance over the last `dominance_window` calls
    3. repeated 3-gram coverage (mostly subsumed by 1; kept as a
       backstop for noisy near-rotations)

    Needs at least min_repeats calls before any verdict (each detector's
    own repeat threshold does the rest of the guarding): run the battery
    with --max-steps >= 12 or tool detection is thin.
    """
    n = len(signatures)
    if n < min_repeats:
        return {"loop": False, "why": f"too-few-calls ({n})"}
    # 1. exact cycle, full sequence
    for p in range(1, n // min_repeats + 1):
        unit = signatures[n - p:]
        reps = 1
        while (reps + 1) * p <= n and \
                signatures[n - (reps + 1) * p:n - reps * p] == unit:
            reps += 1
        if reps >= min_repeats:
            return {"loop": True, "kind": "exact-cycle",
                    "period": p, "repeats": reps,
                    "span": " | ".join(unit[:3])}
    # 2. single-signature dominance
    from collections import Counter
    tail = signatures[-dominance_window:]
    c = Counter(tail)
    sig, cnt = c.most_common(1)[0]
    if cnt >= min_repeats and cnt / len(tail) >= coverage:
        return {"loop": True, "kind": "same-call-dominance",
                "span": sig, "repeats": cnt,
                "coverage": round(cnt / len(tail), 2)}
    # 3. repeated n-gram backstop
    ngram = 3
    grams = [" | ".join(tail[i:i + ngram])
             for i in range(len(tail) - ngram + 1)]
    cg = Counter(grams)
    g, gcnt = cg.most_common(1)[0] if grams else (None, 0)
    if gcnt >= min_repeats and gcnt / max(len(grams), 1) >= 0.5:
        return {"loop": True, "kind": "ngram-cycle", "span": g[:200],
                "repeats": gcnt}
    return {"loop": False}


class Client:
    def __init__(self, base, key, model, timeout=3600):
        self.base = base.rstrip("/")
        self.key = key
        self.model = model
        self.timeout = timeout

    def chat(self, messages, tools=None, max_tokens=2048, temperature=0.8):
        body = {"model": self.model, "messages": messages,
                "max_tokens": max_tokens, "temperature": temperature}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        req = urllib.request.Request(
            self.base + "/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.key}"})
        d = json.load(urllib.request.urlopen(req, timeout=self.timeout))
        choice = d["choices"][0]
        return choice["message"], choice.get("finish_reason"), \
            d.get("usage", {})

    def tokenize(self, text):
        req = urllib.request.Request(
            self.base + "/tokenize",
            data=json.dumps({"model": self.model, "prompt": text}).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.key}"})
        try:
            return len(json.load(
                urllib.request.urlopen(req, timeout=60))["tokens"])
        except Exception:
            return int(len(text) * 0.28)


def parse_args(raw):
    """Tool arguments arrive as a JSON string (or, on some servers, a
    dict). A truncated/malformed string (finish_reason=length) must not
    crash the battery - degrade to a marker the detector can still see."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        v = json.loads(raw)
        return v if isinstance(v, dict) else {"__value__": str(v)[:80]}
    except Exception:
        return {"__unparsed__": str(raw)[:80]}


def run_session(client, brave, case, fillers, depth, max_steps, max_tokens,
                temperature):
    """One agentic session: filler + task, model searches until it answers."""
    if depth not in fillers:
        fillers[depth] = build_filler(
            max(depth - client.tokenize(case["task"]), 1000))
    filler_text = fillers[depth]
    task = ("Using web search where it helps, research deeply and answer "
            "with a thorough, structured response.\n\n" + case["task"])
    messages = [{"role": "user",
                 "content": filler_text + "\n\n=== NEW TASK (answer now) ===\n\n"
                 + task}]
    sigs, texts, queries = [], [], []
    truncated, error = False, None
    q0 = brave.queries
    steps_used = 0
    for step in range(max_steps):
        steps_used = step + 1
        try:
            msg, finish, _usage = client.chat(
                messages, tools=[SEARCH_TOOL], max_tokens=max_tokens,
                temperature=temperature)
        except Exception as e:
            error = str(e)[:200]
            break
        if finish == "length":
            truncated = True
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning") or \
            msg.get("reasoning_content") or ""
        if reasoning:
            texts.append(reasoning)
        if content:
            texts.append(content)
        calls = msg.get("tool_calls") or []
        if not calls:
            messages.append({"role": "assistant", "content": content})
            break
        messages.append({
            "role": "assistant", "content": content,
            "tool_calls": [{"id": c.get("id") or f"c{step}_{j}",
                            "type": "function",
                            "function": c["function"]}
                           for j, c in enumerate(calls)]})
        for j, c in enumerate(calls):
            fn = c.get("function") or {}
            args = parse_args(fn.get("arguments"))
            sigs.append(call_sig(fn.get("name") or "?", args))
            if fn.get("name") == "web_search":
                res, cached = brave.search(args.get("query", ""))
                queries.append({"q": args.get("query", ""),
                                "cached": cached,
                                "n": len(res.get("results", []))
                                if isinstance(res, dict) else 0})
            else:
                res = {"error": f"unknown tool {fn.get('name')}"}
            messages.append({
                "role": "tool",
                "tool_call_id": c.get("id") or f"c{step}_{j}",
                "content": brave._fit(res) if isinstance(res, dict)
                else json.dumps(res)})
    words = " ".join(" ".join(texts).split()).split()
    text_verdict = detect_loop(words)
    tool_verdict = detect_tool_loop(sigs)
    if error:
        verdict = "ERROR"
    else:
        verdict = ("TOOL_LOOP" if tool_verdict.get("loop")
                   else "TEXT_LOOP" if text_verdict.get("loop") else "CLEAN")
    return {
        "case": case["name"], "depth": depth, "steps_used": steps_used,
        "tool_calls": len(sigs), "brave_queries": brave.queries - q0,
        "verdict": verdict, "truncated": truncated, "error": error,
        "tool_detail": tool_verdict, "text_detail": text_verdict,
        "queries": queries[:40],
        "tail": " ".join(words[-40:])[:300],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--cases",
                    default=os.path.join(os.path.dirname(
                        os.path.abspath(__file__)), "cases-cybergym"))
    ap.add_argument("--depths", default="350000,500000")
    ap.add_argument("--case-filter", default=None,
                    help="substring; run only matching cases")
    ap.add_argument("--max-steps", type=int, default=24,
                    help="per-session tool rounds; >= 12 recommended or "
                         "tool-loop detection is vacuous")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--brave-key-file", default=None,
                    help="path to the Brave API key file - keep it OUTSIDE "
                         "this repository")
    ap.add_argument("--out", default="results")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        ok = True
        ident = detect_tool_loop(["web_search({\"query\":\"CVE\"})"] * 10)
        ok &= ident["loop"] and ident["kind"] == "exact-cycle"
        mixed = ["web_search({\"query\":\"CVE-2024\"})"] * 8 + [
            "web_search({\"query\":\"mitigation\"})"]
        dom = detect_tool_loop(mixed)
        ok &= dom["loop"] and dom["kind"] == "same-call-dominance"
        for p in (2, 9, 16):
            rot = detect_tool_loop([f"s(q{i % p})" for i in range(4 * p)])
            if not (rot["loop"] and rot.get("period") == p):
                print(f"rotation p={p} FAILED: {rot}")
                ok = False
        tail3 = ["s(q1)", "s(q2)", "s(q3)", "s(q4)",
                 "s(q5)", "s(q5)", "s(q5)"]
        ok &= not detect_tool_loop(tail3)["loop"]
        clean = detect_tool_loop([f"s({i})" for i in range(16)])
        ok &= not clean["loop"]
        short = detect_tool_loop(["s(1)"] * 3)
        ok &= not short["loop"]
        if call_sig("web_search", {"query": "CVE  x"}) != \
                call_sig("web_search", {"query": "CVE x"}):
            print("whitespace normalization FAILED")
            ok = False
        print("selftest:", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    key = os.environ.get("BRAVE_API_KEY", "")
    if args.brave_key_file:
        key = open(args.brave_key_file).read().strip()
    if not (args.base_url and args.model):
        print("need --base-url and --model", file=sys.stderr)
        return 2
    if not key:
        print("need --brave-key-file or $BRAVE_API_KEY", file=sys.stderr)
        return 2

    cases = load_cases(args.cases)
    if args.case_filter:
        cases = [c for c in cases if args.case_filter in c["name"]]
    brave = BraveSearch(key)
    client = Client(args.base_url, args.api_key, args.model)
    fillers = {}
    os.makedirs(args.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(args.out, f"agentic-{stamp}.jsonl")
    depths = [int(d) for d in args.depths.split(",")]
    rows = []
    with open(path, "w") as fh:
        for case in cases:
            for depth in depths:
                try:
                    r = run_session(client, brave, case, fillers, depth,
                                    args.max_steps, args.max_tokens,
                                    args.temperature)
                except BraveAuthError as e:
                    print(f"FATAL: {e}", file=sys.stderr)
                    return 3
                rows.append(r)
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                print(f"{case['name']:32s} d={depth:7d} "
                      f"{r['verdict']:10s} calls={r['tool_calls']:3d} "
                      f"q={r['brave_queries']:3d}"
                      + (" TRUNCATED" if r["truncated"] else ""),
                      flush=True)
    counted = [r for r in rows if r["verdict"] != "ERROR"]
    errors = len(rows) - len(counted)
    loops = sum(1 for r in counted if r["verdict"] != "CLEAN")
    print(f"\nAGENTIC BATTERY: {loops}/{len(counted)} sessions looped"
          + (f" ({errors} ERROR rows excluded)" if errors else "")
          + f" (results: {path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
