#!/usr/bin/env python3
"""PS5LM memory test: one chat that grows like a long OpenWebUI session.

Each turn adds about 1,500 tokens of text (random words: the memory depends on
the token count, not the meaning) and asks for a short answer; the whole
history (and, with --image, one picture) is sent again every time, as
OpenWebUI does. With --reprocess, it then sends requests whose first message
differs, as OpenWebUI's title and tag requests do, so the server runs the whole
history again. After each request it prints the prompt size and, when the
console's FTP server answers, the memory line PS5LM wrote to app.log.

    python3 scripts/ps5lm-memtest.py <PS5 address> --ftp-port 2121
    python3 scripts/ps5lm-memtest.py <PS5 address> --ftp-port 2121 --image --target 48000 --reprocess 3

Only standard Python 3. It sends chat requests to your own PS5 and reads its
log; it changes nothing on the console.
"""
import argparse
import base64
import ftplib
import io
import json
import random
import re
import struct
import sys
import time
import urllib.request
import zlib

WORDS = ("memory heap arena segment thread request context token cache model console "
         "driver vulkan buffer image prompt server planner budget pool layer attention "
         "state checkpoint batch kernel direct mapped free trim growth history answer").split()


def png(width=512, height=512, seed=1):
    """A noisy gradient PNG, so the vision encoder has real work."""
    rnd = random.Random(seed)
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            rows += bytes(((x * 255 // width + rnd.randrange(40)) & 255,
                           (y * 255 // height + rnd.randrange(40)) & 255,
                           rnd.randrange(256)))

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(rows), 6)) + chunk(b"IEND", b""))


def section(n, rnd):
    paras = []
    for p in range(6):
        words = [rnd.choice(WORDS) for _ in range(160)]
        paras.append(f"Section {n}.{p + 1}: " + " ".join(words) + ".")
    return "\n\n".join(paras)


def post(base, body, timeout):
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def memory_line(host, port):
    """The last 'memory after request' line of app.log over FTP, or None."""
    try:
        ftp = ftplib.FTP()
        ftp.connect(host, port, timeout=5)
        ftp.login()
        buf = io.BytesIO()
        ftp.retrbinary("RETR /data/PS5LM/app.log", buf.write)
        ftp.quit()
    except Exception:
        return None
    lines = [l for l in buf.getvalue().decode("utf-8", "replace").splitlines() if "memory after request" in l]
    return lines[-1] if lines else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("host", help="the PS5's address, e.g. 192.168.0.19")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--ftp-port", type=int, default=1337, help="etaHEN's FTP server (default 1337)")
    ap.add_argument("--target", type=int, default=56000, help="stop once the prompt is this many tokens")
    ap.add_argument("--image", action="store_true", help="put one picture in the first message")
    ap.add_argument("--timeout", type=int, default=900, help="seconds to wait for one answer")
    ap.add_argument("--reprocess", type=int, default=0,
                    help="after the target, this many requests that change the first message (full reprocess), "
                         "each followed by the chat again")
    args = ap.parse_args()
    base = f"http://{args.host}:{args.port}"
    rnd = random.Random(7)

    first = "You are helping review a long technical document, one section at a time. Keep every answer to three short bullets."
    content = [{"type": "text", "text": first}]
    if args.image:
        data = base64.b64encode(png()).decode()
        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}})
        content.append({"type": "text", "text": "Also keep this picture in mind."})
    messages = [{"role": "user", "content": content if args.image else first},
                {"role": "assistant", "content": "Understood. Send the first section."}]

    print(f"PS5LM memory test against {base}, up to {args.target} prompt tokens"
          f"{' with an image' if args.image else ''}\n")
    print(f"{'turn':>4} {'prompt tok':>10} {'secs':>6}  memory (from app.log)")
    state = {"first_free": None}

    def ask(label, msgs, max_tokens):
        """One request; prints a row and returns the reply, or None if it failed."""
        body = {"model": "ps5lm", "messages": msgs, "max_tokens": max_tokens, "stream": False,
                "chat_template_kwargs": {"enable_thinking": False}}
        t0 = time.time()
        try:
            reply = post(base, body, args.timeout)
        except Exception as e:
            print(f"{label:>4} {'-':>10} {time.time() - t0:6.0f}  REQUEST FAILED: {e}")
            print("\nThe server stopped answering. Check the TV and app.log / llama.log (and the .prev logs).")
            return None
        secs = time.time() - t0
        tokens = reply.get("usage", {}).get("prompt_tokens", 0)
        line = memory_line(args.host, args.ftp_port)
        mem = "n/a (no FTP; watch the memory on the TV's Home page)"
        if line:
            m = re.search(r"pool free ([\d.]+) GiB, cpu heap ([\d.]+) GiB \(peak ([\d.]+)", line)
            if m:
                free, heap, peak = map(float, m.groups())
                if state["first_free"] is None:
                    state["first_free"] = free
                mem = f"pool free {free:.2f} GiB ({free - state['first_free']:+.2f}), cpu heap {heap:.2f} GiB, peak {peak:.2f}"
            else:
                mem = line.split("ps5lm-app: ")[-1]
        print(f"{label:>4} {tokens:>10} {secs:6.0f}  {mem}", flush=True)
        return reply

    turn, prompt_tokens = 0, 0
    while prompt_tokens < args.target:
        turn += 1
        messages.append({"role": "user", "content": section(turn, rnd) + "\n\nSummarize this section in three bullets."})
        reply = ask(str(turn), messages, 200)
        if reply is None:
            return 1
        prompt_tokens = reply.get("usage", {}).get("prompt_tokens", 0)
        messages.append({"role": "assistant", "content": reply["choices"][0]["message"].get("content") or ""})

    # Full reprocess: a request whose first message differs, as OpenWebUI's
    # title and tag requests or an edited message are, so the server cannot
    # reuse its cache and runs the whole history again in full ubatches. Then
    # the chat itself again, which differs from that one at the start too.
    for i in range(args.reprocess):
        task = f"Task {i + 1}: write a five-word title for this chat. " + first
        opening = [{"type": "text", "text": task}] + content[1:] if args.image else task
        altered = [{"role": "user", "content": opening}] + messages[1:]
        if ask(f"R{i + 1}", altered, 20) is None:
            return 1
        if ask(f"C{i + 1}", messages, 20) is None:
            return 1

    print(f"\nDone: {turn} turns, {prompt_tokens} prompt tokens"
          f"{f', then {args.reprocess} full reprocesses and their returns' if args.reprocess else ''}"
          ", the server answered every one.")
    if state["first_free"] is not None:
        print("Pass if 'pool free' stayed roughly level (a small dip early is normal) instead of falling every turn.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
