English | [简体中文](README.zh.md)

# SysOne

A fast, local decision inference engine. System-1 speed for edge AI.

SysOne takes a structured state (a JSON object or a piece of text) plus a set of
questions and options, and returns a probability for every option. The shipped
Apple Silicon path is split across two devices: ANE programs for the projection
and FFN stages, and runtime-compiled Metal kernels for attention and the GDN
state. Both halves share IOSurfaces zero-copy.

This README is the complete public path from a fresh clone to a running service
on an Apple Silicon Mac. It does not require any private repository. It is the
master copy; the Chinese translation is `README.zh.md` and both are kept in
sync.

## What is supported

| Backend | Family | Model | Verified path |
|---|---|---|---|
| `pooled` | h2o | [`h2oai/h2o-lightning-4b`](https://huggingface.co/h2oai/h2o-lightning-4b) | ANE + Metal, vendor pooled readout |
| `split` | mude | [`qybaihe/mude-1.0`](https://huggingface.co/qybaihe/mude-1.0) | ANE + Metal, code-row readout |

`--backend reference` is the M0 semantic judge used by tests. It is not a
production path and requires an explicit `--descriptor`.

Both released models support `choice` and `noul` questions. h2o additionally
supports `score` through the vendor `format_answer`; mude's code-row readout
does not.

## Requirements

- Apple Silicon (M1 or newer) on macOS 15 or newer.
- 16 GB of memory or more; h2o's weights alone are about 9.3 GB.
- arm64 Python 3.11 or newer.
- Xcode Command Line Tools (`xcode-select --install`) for `clang`.
- Disk: about 9.3 GB of weights for h2o, about 4 GB for mude, plus several
  GiB of compiled artifacts (h2o is about 3.33 GiB). The weights are downloaded
  on first use.

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

## Build the native libraries

`serve` needs three small native libraries. They are built from the `.m` sources
in `tools/native/`, not shipped in the repository. Run this once per checkout;
the first `serve` will not start without it:

```sh
tools/build_ane.sh
```

The script writes `libe8ane.dylib` (the ANE private-API host), `libe8mtl.dylib`
(the Metal host) and `libe8image.dylib` (the ImageIO decoder used only by the
vision path) into `.tmp/e8`. The engine finds that directory on its own;
`SYS1_E8_DIR` overrides it. Do not use `/tmp`: macOS clears it on reboot.

## Start the service

```sh
sysone serve --model h2o
```

That one command is the whole path. On the first start it downloads the pinned
weights, compiles the missing ANE programs (the vision tower's 26 included) and
then prints `sysone serve: ready in ...s`; later starts read the artifact cache
and report `compile 0.00s`. Run it again at any time: it is idempotent and never
rebuilds what is already on disk.

The port is bound before loading, so `/health` and `/v1/systemone` answer `503`
until the ready line. For mude use `--model mude`. h2o also accepts the
bidirectional `noncausal_full` mode, which requires the whole prompt to fit in
one ANE bucket: long prompts need `--bucket 4096` (the default bucket is 1024),
while the causal default has no such constraint.

Check it from another shell:

```sh
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @examples/request-service.json
```

The full contract — routes, authentication, status codes, request and response
shapes, and per-model capabilities — is in [API.md](API.md).

### Options

`sysone serve` takes the shared model flags plus the HTTP ones. Defaults are
what a first run needs; every flag is optional.

| Flag | Default / values | Meaning |
|---|---|---|
| `--model {h2o,mude}` | from `--backend` or `--weights` | model family |
| `--backend {split,pooled,reference}` | from `--model` | execution path; `reference` is the test-only semantic judge |
| `--weights PATH` | auto-download | explicit package; read-only, never downloads |
| `--offline` | off | never touch the network |
| `--quant {int8,fp16}` | `int8` | ANE weight storage; compute stays fp16 either way |
| `--attention-mode {causal,noncausal_full,prefix_noncausal}` | family default (`causal` for h2o, `prefix_noncausal` for mude) | mask; a request can override it |
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

The model weights carry their own licenses: h2o is Apache-2.0; mude is MIT for
the changed files plus Apache-2.0 for the Qwen base. Their `LICENSE`, `NOTICE`,
and `LICENSES/` files are part of the download and must be kept.

The engine source in this repository does not yet carry a root `LICENSE`; that
is an open release item.
