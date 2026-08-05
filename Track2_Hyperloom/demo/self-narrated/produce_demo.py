#!/usr/bin/env python3
"""
produce_demo.py — build the self-narrated Track 2 demo, every asset made LOCALLY
on one AMD Radeon R9700 via Lemonade (:13305): Flux images, kokoro narration, and
real captured terminal output rendered to frames. Stitched with ffmpeg. No cloud.

Usage: python3 produce_demo.py [out.mp4]
"""
import base64, json, os, subprocess, sys, tempfile, urllib.request
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

URL = os.environ.get("LEMONADE_URL", "http://localhost:13305")
HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "demo.mp4"
W, H = 1920, 1080
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
CAP = Path("/tmp/demo-build")  # real captured hyperloom output

def api(path, payload, out=None, timeout=300):
    req = urllib.request.Request(URL + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    if out: Path(out).write_bytes(data)
    return data

def load(model):
    try: api("/api/v1/load", {"model_name": model}, timeout=600)
    except Exception: pass

def flux_card(prompt, path):
    """Generate a Flux image; fall back to a rendered title card on failure."""
    try:
        d = json.loads(api("/api/v1/images/generations",
              {"model": "Flux-2-Klein-9B-GGUF", "prompt": prompt, "n": 1, "size": "1024x1024"}, timeout=600))
        raw = base64.b64decode(d["data"][0]["b64_json"])
        tmp = path.with_suffix(".src.png"); tmp.write_bytes(raw)
        img = Image.open(tmp).convert("RGB").resize((H, H))          # square -> center on 1920x1080
        canvas = Image.new("RGB", (W, H), (8, 8, 12)); canvas.paste(img, ((W - H)//2, 0))
        canvas.save(path); return True
    except Exception as e:
        print(f"  flux fallback ({e})"); title_card(prompt.split(",")[0], path); return False

def title_card(text, path):
    img = Image.new("RGB", (W, H), (10, 10, 14)); d = ImageDraw.Draw(img)
    f = ImageFont.truetype(SERIF, 64)
    d.rectangle([80, H//2-4, 80+140, H//2], fill=(210, 40, 40))
    _wrap(d, text, ImageFont.truetype(SERIF, 60), (80, H//2+30), (235, 235, 240), W-160)
    img.save(path)

def term_frame(title, body_file, path):
    """Render real captured terminal output as a dark 'terminal window' frame."""
    img = Image.new("RGB", (W, H), (6, 8, 10)); d = ImageDraw.Draw(img)
    d.rectangle([60, 60, W-60, H-60], fill=(14, 16, 20), outline=(40, 44, 52), width=2)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([90+i*34, 88, 110+i*34, 108], fill=c)
    d.text((150, 86), title, font=ImageFont.truetype(MONO, 26), fill=(150, 156, 168))
    body = Path(body_file).read_text() if Path(body_file).exists() else "(no output)"
    f = ImageFont.truetype(MONO, 30); y = 150
    for line in body.splitlines()[:24]:
        col = (120, 230, 140) if any(k in line for k in ("PASS", "PRESENT", "Speedup", "1.49")) else (208, 214, 222)
        d.text((110, y), line[:96], font=f, fill=col); y += 40
    img.save(path)

def _wrap(d, text, font, xy, fill, maxw):
    x, y = xy; words = text.split(); line = ""
    for w in words:
        t = (line + " " + w).strip()
        if d.textlength(t, font=font) > maxw and line:
            d.text((x, y), line, font=font, fill=fill); y += font.size + 14; line = w
        else: line = t
    if line: d.text((x, y), line, font=font, fill=fill)

def narrate(text, path):
    load("kokoro-v1")
    api("/api/v1/audio/speech", {"model": "kokoro-v1", "input": text, "voice": "af_sky"}, out=path)
    if Path(path).stat().st_size < 1000: raise RuntimeError("no audio")

def dur(mp3):
    return float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", mp3]).strip())

SECTIONS = [
    ("cold-open", "flux:sleek dark tech title card, glowing red AMD Radeon GPU die with circuit traces, cinematic, minimal",
     "This entire video was produced and narrated locally, on a single AMD Radeon AI PRO R9700, by the very agent it demonstrates."),
    ("the-gap", "term:hyperloom scan gfx1201  —  ISA ground truth|" + str(CAP / "scan.txt"),
     "The agent scans the silicon, with the assembler as ground truth, and finds the native eight-wide int four dot that the kernels never use."),
    ("the-fix", "term:hyperloom bench  —  correctness-gated, measured on the card|" + str(CAP / "bench.txt"),
     "It writes the kernel, correctness gates it bit-exact against a CPU reference, and measures the win on the card."),
    ("the-result", "flux:bold victory title card, glowing 1.49x speedup, dark background, red and white, cinematic tech",
     "Native dot eight int four decode: one point four nine times faster than the dp4a route. Discovered, fixed, and proven, with no human in the loop."),
]

def main():
    work = Path(tempfile.mkdtemp()); segs = []
    print(f"producing {OUT} — all assets local on the R9700")
    for i, (name, visual, text) in enumerate(SECTIONS):
        print(f"  [{i}] {name}")
        frame = work / f"{i}.png"
        if visual.startswith("flux:"): flux_card(visual[5:], frame)
        else:
            spec = visual[5:]; title, bf = spec.split("|", 1); term_frame(title, bf, frame)
        mp3 = work / f"{i}.mp3"; narrate(text, str(mp3)); d = dur(str(mp3)) + 0.6
        seg = work / f"{i}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loop", "1", "-i", str(frame), "-i", str(mp3),
                        "-t", f"{d:.2f}", "-vf", f"scale={W}:{H},format=yuv420p",
                        "-c:v", "libx264", "-c:a", "aac", "-shortest", str(seg)],
                       check=True, capture_output=True)
        segs.append(seg)
    lst = work / "list.txt"; lst.write_text("".join(f"file '{s}'\n" for s in segs))
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", str(OUT)], check=True, capture_output=True)
    sz = OUT.stat().st_size
    print(f"wrote {OUT} ({sz//1024} KB, {len(segs)} sections — 100% local on gfx1201)")

if __name__ == "__main__":
    main()
