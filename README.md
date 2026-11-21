# SysOne

A fast, local decision inference engine. System-1 speed for edge AI.

SysOne takes a structured state (a JSON object or a piece of text) plus a set of
questions and options, and returns a probability for every option. The shipped
Apple Silicon path is split across two devices: ANE programs for the projection
and FFN stages, and runtime-compiled Metal kernels for attention and the GDN
state. Both halves share IOSurfaces zero-copy.

This README is the complete public path from a fresh clone to a running service
on an Apple Silicon Mac. It does not require any private repository.

## What is supported

| Backend | Family | Model | Verified path |
|---|---|---|---|
| `split` | mude | [`qybaihe/mude-1.0`](https://huggingface.co/qybaihe/mude-1.0) | ANE + Metal, code-row readout |
| `pooled` | h2o | [`h2oai/h2o-lightning-4b`](https://huggingface.co/h2oai/h2o-lightning-4b) | ANE + Metal, vendor pooled readout |

`--backend reference` is the M0 semantic judge used by tests. It is not a
production path and requires an explicit `--descriptor`. Non-macOS platforms
cannot run the Apple backends; `reference` is the only path there.

Both released models support `choice` and `noul` questions. h2o additionally
supports `score` through the vendor `format_answer`; mude's code-row readout
does not.

## Requirements

- Apple Silicon (M1 or newer). The checked configuration is an M2 Pro with
  16 GB of memory on macOS 15.7.2.
- arm64 Python 3.11 or newer.
- Xcode Command Line Tools (`xcode-select --install`) for `clang`.
- Disk: about 4 GB of weights for mude, about 9.3 GB for h2o, plus several
  GiB of compiled artifacts (h2o is about 3.33 GiB). The weights are downloaded
  on first use.

Versions are pinned: engine `0.1.0`; compiled artifact format version 3; model
revisions are pinned in `src/sysone/models/sources.py`.

## Install

```sh
git clone https://github.com/JuLian1130/SysOne.git sysone
cd sysone

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ./src
```

`pip install -e ./src` is the supported install. It pulls `huggingface_hub`, which
is also what the first run uses to fetch the weights. A non-editable wheel is not
supported today: the Metal kernel sources live in `src/sysone/devices/apple/shaders/`,
and `env.scratch_root()` is anchored at the source tree. Both expect the checkout
to stay in place.

If you do not want to install the package, every command below also works as:

```sh
PYTHONPATH=src python -m sysone.cli <command> ...
```

## Build the native dylibs

The three dylibs are build inputs, not checked-in binaries:

- `libe8ane.dylib` — the ANE private-API host (`tools/native/e8_ane.m`);
- `libe8mtl.dylib` — the Metal host (`tools/native/e8_mtl.m`);
- `libe8image.dylib` — the ImageIO decoder used only by the vision path
  (`tools/native/e8_image.m`).

```sh
tools/build_ane.sh
ls .tmp/e8
```

The output defaults to `.tmp/e8` (ignored by git). `SYS1_E8_DIR` overrides it.
Do not use `/tmp`: macOS clears it on reboot, and a missing dylib makes the whole
Apple path unavailable. `ANE_SRC` can point the script at another source
directory; the repository source is `tools/native/`.

The engine finds `.tmp/e8` on its own (`SYS1_E8_DIR` overrides it); if a dylib
is missing, the startup error names the path and tells you to run the build
script.

## Weights

`sysone serve` resolves the weight package itself. If `--weights` is omitted
and the package is not cached, the first start downloads it from Hugging Face at
a pinned revision. The download prints one line with the repository, revision,
size and directory; later starts read the `source.json` marker and touch no
network.

- Default directory: `~/Library/Caches/sysone/weights/<repo directory>`
  (`mude-1.0`, `h2o-lightning-4b`). `SYS1_WEIGHTS_DIR` overrides it.
- `--offline` forbids the download. A missing package then fails with the exact
  pinned `hf download` command; an air-gapped host copies a finished package
  (including its `source.json`) under `SYS1_WEIGHTS_DIR`.
- An explicit `--weights` is read-only: it never downloads, so a typo in a path
  cannot start a multi-GB fetch.
- The download list contains only the files the engine reads plus each
  repository's license files; it is pinned in `src/sysone/models/sources.py`.
- `HF_TOKEN` is honoured through `huggingface_hub`'s standard environment for
  private mirrors.

## Start the service

```sh
sysone serve --model mude
```

That one command is the whole path. On the first start it downloads the pinned
weights, compiles the missing ANE programs (the vision tower's 26 included) and
then prints `sysone serve: ready in ...s`; later starts read the artifact cache
and report `compile 0.00s`. Run it again at any time: it is idempotent and never
rebuilds what is already on disk.

The port is bound before loading, so `/health` and `/v1/systemone` answer `503`
until the ready line. For h2o use `--model h2o`. The bidirectional
`noncausal_full` mode requires the whole prompt to fit in one ANE bucket, so
long prompts need `--bucket 4096` (the default bucket is 1024); the causal
default has no such constraint.

### Options

`sysone serve` takes the shared model flags plus the HTTP ones. Defaults are
what a first run needs; every flag is optional.

| Flag | Default / values | Meaning |
|---|---|---|
| `--model {mude,h2o}` | from `--backend` or `--weights` | model family |
| `--backend {split,pooled,reference}` | from `--model` | execution path; `reference` is the test-only semantic judge |
| `--weights PATH` | auto-download | explicit package; read-only, never downloads |
| `--offline` | off | never touch the network |
| `--quant {int8,fp16}` | `int8` | ANE weight storage; compute stays fp16 either way |
| `--attention-mode {causal,noncausal_full,prefix_noncausal}` | family default (`prefix_noncausal` for mude, `causal` for h2o) | mask; a request can override it |
| `--bucket N` | `1024` | ANE program column width |
| `--width N` | `16384` | GPU buffer width, also the max input tokens |
| `--W N` | `4` | resident layers; `0` keeps all of them |
| `--artifacts PATH` | platform cache | compiled artifact root |
| `--descriptor PATH` | — | semantic-judge descriptor (`--backend reference` only) |
| `--debug-timings` | off | add per-stage timings to each answer |

Vision (only for models with a tower):

| Flag | Default / values | Meaning |
|---|---|---|
| `--vision {auto,off}` | `auto` | `off` serves text only; image requests get `422` |
| `--vision-idle S` | `60` | seconds before the tower is unloaded; `0` unloads after every request |
| `--vision-bucket N` | chain bucket | tower tile width; must be a multiple of 4 |

HTTP:

| Flag | Default | Meaning |
|---|---|---|
| `--host HOST` | `127.0.0.1` | bind address |
| `--port N` | `8000` | bind port; `0` picks a free one |
| `--queue N` | `8` | admission waiting slots |
| `--admit-timeout S` | `120` | seconds a request may wait before `429` |
| `--name NAME` | model id | `model` field in responses |

Environment overrides: `SYS1_LOG_LEVEL` (default `info`), `SYS1_WEIGHTS_DIR`,
`SYS1_ARTIFACTS_DIR`, `SYS1_E8_DIR`, and `DECISION_API_KEY`; the matching flags
win over the variables.

In another shell:

```sh
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @examples/request-service.json
```

## Requests

`/v1/systemone` carries `state` plus a `questions` **object** keyed by question
id (`examples/request-service.json`). Each question has `type`, `instructions`,
and `criteria`:

```json
{
  "state": {"battery": 8},
  "questions": {
    "q1": {
      "type": "choice",
      "instructions": "What now?",
      "criteria": {"continue": "keep going", "divert": "go to a charger"}
    }
  }
}
```

`criteria` for `choice` is an object of option keys (an optional description is
the value); for `score` it is an ordered array of levels; for `noul` it is
`{"true": …, "false": …}`. A request may carry up to 255 options per question.
`state` and `instructions` may be strings or arbitrary JSON; non-strings are
serialized as JSON into the prompt.

## Service contract

`sysone serve` exposes:

| Route | Method | Auth | Purpose |
|---|---|---|---|
| `/health` | GET | none | readiness and model info |
| `/metrics` | GET | none | admission, engine, startup, vision, batching counters |
| `/v1/models` | GET | bearer when configured | model name, max input tokens, modalities, vision limits |
| `/v1/systemone` | POST | bearer when configured | run one decision request |

- Authentication is enabled by setting `DECISION_API_KEY` (not `SYS1_API_KEY`).
  When set, `/v1/models` and `/v1/systemone` require
  `Authorization: Bearer $DECISION_API_KEY`; `/health` and `/metrics` stay open.
- While the engine loads, every route returns `503` with `Retry-After`. A load
  failure also returns `503`, with the failure reason.
- When the admission queue is full, `/v1/systemone` returns `429` with
  `Retry-After` and closes the connection without reading the body.
- Malformed requests return the same two `422` shapes as the reference protocol:
  a string `detail`, or `{"invalid_questions": {id: reason}}` with every bad
  question reported at once.
- The response is:

```json
{
  "model": "...",
  "answers": {
    "q1": {"type": "choice", "choice": "continue", "probabilities": {"continue": 0.7, "divert": 0.3}}
  },
  "usage": {"input_tokens": 123, "output_tokens": 0},
  "attention_mode": "prefix_noncausal",
  "attention_trained": true
}
```

- `noul` answers carry `noul` (P(true)); `score` answers carry the expected
  level in `score` and per-level `probabilities`.
- `attention_mode` in the request body overrides the process default set by
  `--attention-mode` for that request. The CLI request file accepts the same
  field.
- `images` accepts a list of `data:image/...;base64,` data URLs. Video is rejected.
  When `--vision off` is set, image requests are rejected with `422`.

## Capability matrix

| | mude | h2o |
|---|---|---|
| Question kinds | `choice`, `noul` | `choice`, `noul`, `score` |
| Attention modes | `causal`, `noncausal_full`, `prefix_noncausal` (trained on `prefix_noncausal`) | `causal` (default), `noncausal_full` |
| Readout | fixed 255-row code table | vendor pooled readout |
| Max options per question | 255 | 255 |
| Max input tokens | `--width` (default 16384) | `--width` (default 16384) |
| Vision | available by default (`--vision auto`) | available by default (`--vision auto`) |

The first start also builds the vision tower's 26 programs, then releases the
tower: text-only processes stay at 0 resident, and the first image request pays
only the load. `--vision off` skips the tower and its programs. The third dylib
(`libe8image.dylib`) is only needed when an image request is actually served.

## Resources and tuning

- Both caches live under the platform cache and honour `SYS1_WEIGHTS_DIR` /
  `SYS1_ARTIFACTS_DIR`; deleting one forces a re-download or a recompile. The
  first compile is minutes-level, later starts report `compile 0.00s`.
- Keep `--quant` stable across starts: it selects the artifact set (`s<bucket>`
  for `int8`, `s<bucket>-fp16` for `fp16`), so switching does not overwrite the
  other set but does compile a second one.
- `--bucket` and `--width` trade tile size against GPU memory. On a 16 GB
  machine start with `--width 8192`; widen the bucket only when the prompts
  need it. A prompt longer than `--width` is rejected with `422`.
- `--W` trades memory for latency: the default `4` (one block) keeps the least
  resident, `--W 0` keeps every layer loaded.
- A geometry outside the measured whitelist still runs, but its numbers are
  experimental.

## Troubleshooting

- `NotReadyError: 找不到 libe8ane.dylib` — run `tools/build_ane.sh`; check
  `SYS1_E8_DIR`. If `clang` is missing, run `xcode-select --install`.
- `ModuleNotFoundError: tokenizers` (or `jinja2`/`numpy`/`huggingface_hub`) —
  reinstall the package in the active venv: `pip install -e ./src`.
- A `download` error with `--offline` — drop `--offline` to let the first run
  fetch the weights; run the printed `hf download …` command and start with
  `--weights <that directory>`; or copy a finished package (including its
  `source.json`) under `SYS1_WEIGHTS_DIR`.
- `权重文件不存在` from an explicit `--weights` — the directory must be the
  published layout: a single `model.safetensors`, or a row-int8 directory with
  `int8_manifest.json`. Do not use `qybaihe/mude-1.0-mlx-8bit`: its MLX q8
  shards are not readable by the ANE chain.
- `422` with `prompt 超过链的缓冲宽度` — raise `--width` (and `--bucket` if
  needed), or shorten the prompt.
- Out-of-memory or a wired-memory warning — lower `--width`, and check that no
  other large model is resident.
- `503` from the service — the engine is still loading (the first start also
  downloads the weights and compiles), or loading failed; read the process log
  and `/metrics`.
- `429` — raise `--queue`/`--admit-timeout` or retry after `Retry-After`.
- h2o fails at `Serve.open` — `serve_config.json` or `h2o_lightning_shim.py` is
  missing; delete the cached package and let the next start re-download it.
- `--backend reference` requires `--descriptor`; it is the semantic-judge path,
  not a model runtime.

## Repository layout

- `src/sysone/` — the engine (Python only; dependencies are numpy, tokenizers,
  and jinja2).
- `src/sysone/devices/apple/shaders/` — vendored Metal kernel sources, compiled
  at process startup.
- `tools/native/` — the three `.m` sources for the dylibs; `tools/build_ane.sh`
  builds them.
- `examples/` — one request for each wire shape.
- `tests/` — unit tests and Apple integration tests; integration tests skip when
  weights or dylibs are absent.

## Licenses

The model weights carry their own licenses: mude is MIT for the changed files
plus Apache-2.0 for the Qwen base; h2o is Apache-2.0. Their `LICENSE`, `NOTICE`,
and `LICENSES/` files are part of the download and must be kept.

The engine source in this repository does not yet carry a root `LICENSE`; that
is an open release item.
