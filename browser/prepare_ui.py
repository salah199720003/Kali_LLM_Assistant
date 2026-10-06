"""Prepare browser-only Qwen effort controls from the installed llama.cpp UI.

Run once while a Qwen server is running. The existing launchers serve these
static assets on their normal browser URLs at the next model launch.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import urllib.request
from urllib.parse import urlsplit


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "runtime" / "qwen-webui"
BUDGET_CODE = "ne>=0&&(ue.thinking_budget_tokens=ne),"
SEND_ANCHOR = "try{const ae={...nC()};k&&n&&"
# The menu wins over stale custom JSON. Default leaves advanced settings alone.
# High maps to xhigh, the strongest instruction supported by Qwen's template.
EFFORT_CODE = (
    'if(m!==void 0){'
    'delete ue.reasoning_effort;'
    'ue.chat_template_kwargs={...ue.chat_template_kwargs??{},enable_thinking:m};'
    'if(m&&O){const qwenEffort=O===ls.LOW?"low":O===ls.MEDIUM?"medium":'
    'O===ls.HIGH||O===ls.MAX?"xhigh":void 0;'
    'if(qwenEffort)ue.chat_template_kwargs.reasoning_effort=qwenEffort;'
    '}}'
)
TOKEN_LABEL = (
    'tokenLabel(s){if(s.value===ls.DEFAULT)return"Model default";'
    'const o=oDe[s.value];return o===void 0?null:o===-1?"Unlimited":'
    '`Max ${o.toLocaleString()} tokens`}'
)
EFFORT_LABEL = (
    'tokenLabel(s){return s.value===ls.DEFAULT?"Model default":'
    's.value===ls.OFF?null:s.value===ls.LOW?"low effort":'
    's.value===ls.MEDIUM?"medium effort":"xhigh effort"}'
)


def patch_bundle(source):
    replacements = [
        (BUDGET_CODE, ""),
        (SEND_ANCHOR, EFFORT_CODE + SEND_ANCHOR),
        (TOKEN_LABEL, EFFORT_LABEL),
        ("Maximum reasoning effort with extended context usage",
         "Qwen xhigh effort, the same instruction as High"),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError(
                "This llama.cpp UI version needs a new browser patch. "
                "The existing browser files have not been replaced."
            )
        source = source.replace(old, new, 1)
    return source


def read_asset(base, path):
    request = urllib.request.Request(base.rstrip("/") + "/" + path.lstrip("/"),
                                     headers={"Accept-Encoding": "gzip"})
    with urllib.request.urlopen(request, timeout=15) as response:
        data = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            data = gzip.decompress(data)
        return data


def safe_asset_path(path):
    path = path.removeprefix("./")
    parts = PurePosixPath(path).parts
    if not parts or path.startswith("/") or ".." in parts or "\\" in path or ":" in path:
        raise RuntimeError(f"Unsupported static asset path: {path}")
    return path


def prepare(base):
    parsed = urlsplit(base)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("Use the local Qwen server URL")
    props = json.loads(read_asset(base, "props"))
    template = props.get("chat_template", "")
    if "reasoning_effort" not in template or "xhigh" not in template:
        raise RuntimeError("This server does not expose Qwen's effort template")
    page = read_asset(base, "").decode()
    match = re.search(r'href="(\./_app/immutable/bundle\.[^"/]+\.js)"', page)
    if not match:
        raise RuntimeError("The installed UI bundle could not be found")
    original_path = safe_asset_path(match[1])
    patched = patch_bundle(read_asset(base, original_path).decode()).encode()
    digest = hashlib.sha256(patched).hexdigest()[:12]
    patched_path = f"_app/immutable/bundle.qwen-effort-{digest}.js"
    # Use the upstream precache list to include fonts, PDF workers, icons and
    # other lazily loaded assets, not just files referenced by index.html.
    worker = read_asset(base, "sw.js").decode()
    paths = {safe_asset_path(path) for path in re.findall(r'\{url:"([^"]+)"', worker)
             if path not in {"./", "sw.js", "index.html"}}
    if original_path not in paths:
        raise RuntimeError("The upstream asset list does not contain the main UI bundle")
    paths.remove(original_path)
    paths.update({"manifest.webmanifest", "build.json", "_app/version.json"})
    # Fetch and validate everything before updating any files.
    with ThreadPoolExecutor(max_workers=6) as pool:
        assets = dict(zip(sorted(paths), pool.map(lambda path: read_asset(base, path), sorted(paths))))
    assets[patched_path] = patched
    page = page.replace(original_path, patched_path)
    # Let llama.cpp supply its retirement worker; copying the old PWA worker
    # would re-cache the unpatched embedded UI on this same browser origin.
    if (OUTPUT / "sw.js").exists():
        raise RuntimeError("Remove the old custom sw.js before preparing this UI")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for path, content in assets.items():
        target = OUTPUT / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (OUTPUT / "effort-ui.json").write_text(json.dumps({
        "source": base,
        "bundle": patched_path,
        "sha256": hashlib.sha256(patched).hexdigest(),
        "assets": sorted(assets),
        "reasoning": {"off": False, "low": "low", "medium": "medium",
                      "high": "xhigh", "max": "xhigh"},
    }, indent=2), encoding="utf-8")
    # Publish the entry point last, after all its assets are present.
    (OUTPUT / "index.html").write_text(page, encoding="utf-8")
    print(f"Prepared {len(assets)} browser assets at {OUTPUT}.")
    print("Next Qwen launch: Reasoning menu selects low / medium / xhigh / off.")
    print("The running model and terminal agent have not been changed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="http://127.0.0.1:8083")
    prepare(parser.parse_args().backend)
