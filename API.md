English | [简体中文](API.zh.md)

# SysOne service API

`sysone serve --model h2o` exposes one HTTP service on `127.0.0.1:8000`. This
page is the request contract; installation and startup are in
[README.md](README.md).

## Endpoints

| Route | Method | Auth | Purpose |
|---|---|---|---|
| `/health` | GET | none | readiness and model info |
| `/metrics` | GET | none | admission, engine, startup, vision, batching counters |
| `/v1/models` | GET | bearer when configured | model name, max input tokens, modalities, image contract |
| `/v1/systemone` | POST | bearer when configured | run one decision request |

`/health` returns `status`, `model`, `max_input_tokens`, `modalities` and
`authentication`. `/v1/models` adds the image contract when the model has a
tower: `min_pixels`, `max_pixels`, `max_bytes`, `max_source_pixels`,
`placement`, `patch_embedding`, `decoder` and `image_processor`.

## Authentication

Set `DECISION_API_KEY` (not `SYS1_API_KEY`) to enable it. `/v1/models` and
`/v1/systemone` then require `Authorization: Bearer $DECISION_API_KEY`;
`/health` and `/metrics` stay open.

## Status codes

- While the engine loads, every route returns `503` with `Retry-After`. A load
  failure also returns `503`, with the failure reason.
- When the admission queue is full, `/v1/systemone` returns `429` with
  `Retry-After` and closes the connection without reading the body.
- Malformed requests return the same two `422` shapes as the reference
  protocol: a string `detail`, or `{"invalid_questions": {id: reason}}` with
  every bad question reported at once.
- A request this model cannot answer returns `400` with a `detail`: a question
  kind it has no readout for, or more options or `score` levels than its
  readout carries (see the table below). Retrying it unchanged will not help.

## Request

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

`type` is `choice`, `noul` or `score`. Anything else — `locate` is the one a
vendor ships — is a per-question `422` that names the kinds this wire format
carries.

`images` accepts a list of `data:image/...;base64,` data URLs. With
`--vision off` set, image requests are rejected with `422`.

`videos` is refused with `422` and `video inputs are not available: this engine
ships no video path`: d3's own package documents video input, but this engine
has neither a frame decoder nor a time axis.

Every question is answered from the state alone: questions of one request never
attend to each other, so adding, removing or reordering them cannot change
another answer. That is also why there is no `isolated` or `joint` knob — those
are layouts in which the questions, or the state, see each other.

## Response

```json
{
  "model": "...",
  "answers": {
    "q1": {"type": "choice", "choice": "continue", "probabilities": {"continue": 0.7, "divert": 0.3}, "confidence": 0.4}
  },
  "usage": {"input_tokens": 123, "output_tokens": 0},
  "attention_mode": "causal",
  "attention_trained": true
}
```

- `choice` and `score` answers also carry `confidence`: `1` for a point mass,
  `0` for a distribution as flat as the option count allows. The form is the
  model's own — h2o reports its top option's probability rescaled against that
  flat distribution (`0.7`/`0.3` gives `0.4`), mude reports `1 - H/log n`, which
  gives `0.12` for the same pair. The example above is h2o's number.
- `noul` answers carry `noul` (P(true)) and no `confidence`: neither vendor's
  yes/no answer object has one.
- `score` answers carry the expected level in `score` (0-based), the per-level
  `probabilities` keyed `"0"`..`"n-1"`, and `confidence`. mude takes 2 to 10
  levels, h2o up to 255.
- `attention_mode` in the request body overrides the process default set by
  `--attention-mode` for that request.

## Per-model capabilities

| | h2o | mude | d3 | torchcast | pplx12 | pplx13 |
|---|---|---|---|---|---|---|
| Question kinds | `choice`, `noul`, `score` | `choice`, `noul`, `score` | `choice`, `noul`, `score` | `choice`, `noul`, `score` | `choice`, `noul`, `score` | `choice`, `noul`, `score` |
| Attention modes | `causal` (default), `noncausal_full` | `causal`, `noncausal_full`, `prefix_noncausal` (trained) | `noncausal_full` only | `noncausal_full` only | `noncausal_full` only | `noncausal_full` only |
| Readout | vendor pooled readout | 255-row code table | 255-row code table | 26 `lm_head` letter rows | 255-row code table (BF16) | 255-row code table (BF16) |
| Max options per question | 255 (255 levels) | 255 (10 levels) | 255 (10 levels) | 26 (10 levels) | 255 (255 levels) | 255 (255 levels) |
| Max input tokens | `--width` (default 32768) | `--width` (default 32768) | `--width` (default 32768) | `--width` (default 32768) | `--width` (default 32768) | `--width` (default 32768) |
| Vision | available by default (`--vision auto`) | available by default (`--vision auto`) | available by default (`--vision auto`) | available by default (`--vision auto`) | available by default (`--vision auto`) | available by default (`--vision auto`) |

`--width` is both the GPU buffer width and the input limit, so a longer prompt
needs a larger value. Every family with a tower answers images; mude's tower is
inherited from d3-nano and, per its model card, was neither trained nor
evaluated for mu's decisions — it runs, but it is not a validated capability
there.

The first start also builds the vision tower's 26 programs, then releases the
tower: text-only processes stay at 0 resident, and the first image request pays
only the load. `--vision off` skips the tower and its programs. The third dylib
(`libe8image.dylib`) is only needed when an image request is actually served.
