"""CLI：inspect 走家族解析，run 走参考后端。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from helpers import FIXTURES, make_request, make_tiny
from sysone.cli import (
    _artifacts_dir,
    _geometry_report,
    _infer_family,
    _platform_report,
    _resolve_backend,
    main,
)
from sysone.devices.apple import platform as apple_platform
from sysone.errors import ConfigError


def test_inspect_prints_descriptor(capsys) -> None:
    code = main(["inspect", "--model", "mude", "--weights", str(FIXTURES / "mude")])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["family"] == "mude"
    assert payload["num_layers"] == 20
    assert payload["default_mode"] == "prefix_noncausal"


def test_the_model_flag_names_the_parameter_size_of_each_family(capsys) -> None:
    """同一个厂商可能发 2B / 4B 几档，所以 `--model` 的取值表与帮助都要带档位。

    用法行（metavar）与报错各查一遍：认错家族的那一刻正是用户需要看到"这个名字对应多大"。
    """
    with pytest.raises(SystemExit) as helped:
        main(["serve", "--help"])
    assert helped.value.code == 0
    assert ("{d3=4.54B,h2o=4.66B,mude=1.98B,pplx12=4.54B,pplx13=4.54B,singvl=4.54B,"
            "torchcast=5.17B}") in capsys.readouterr().out
    with pytest.raises(SystemExit) as refused:
        main(["serve", "--model", "d3-8b"])
    assert refused.value.code == 2
    assert "mude 1.98B" in capsys.readouterr().err


def test_inspect_prints_the_parameter_size(capsys) -> None:
    """`inspect` 的回包里带 `params`：它是家族名承诺的那一档，不是 config 里的字段。"""
    code = main(["inspect", "--model", "mude", "--weights", str(FIXTURES / "mude")])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["params"] == "1.98B"


def test_run_with_tiny_model(tmp_path: Path, capsys) -> None:
    descriptor, store = make_tiny(tmp_path)
    descriptor_path = tmp_path / "descriptor.json"
    descriptor_path.write_text(json.dumps(descriptor.to_mapping()), encoding="utf-8")
    request = make_request(mode=None, questions=1)
    input_path = tmp_path / "request.json"
    input_path.write_text(
        json.dumps(
            {
                "state": request.state,
                "questions": [{"text": request.questions[0].text, "options": list(request.questions[0].options)}],
                "attention_mode": "causal",
            }
        ),
        encoding="utf-8",
    )
    code = main(
        [
            "run",
            "--backend",
            "reference",
            "--descriptor",
            str(descriptor_path),
            "--weights",
            str(tmp_path / "tiny.safetensors"),
            "--input",
            str(input_path),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["answers"][0]["value"] in {"A", "B", "C", "D"}
    assert payload["provenance"]["attention_mode"] == "causal"
    assert payload["provenance"]["attention_trained"] is False


def test_run_rejects_unknown_mode(tmp_path: Path, capsys) -> None:
    descriptor, _ = make_tiny(tmp_path)
    descriptor_path = tmp_path / "descriptor.json"
    descriptor_path.write_text(json.dumps(descriptor.to_mapping()), encoding="utf-8")
    code = main(
        [
            "run",
            "--backend",
            "reference",
            "--descriptor",
            str(descriptor_path),
            "--weights",
            str(tmp_path / "tiny.safetensors"),
            "--input",
            str(tmp_path / "request.json"),
            "--attention-mode",
            "bogus",
        ]
    )
    assert code == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["code"] == "config"


# ---- doctor：几何白名单的**查询**形态（不装载权重）--------------------------------------


def test_doctor_reports_whether_a_models_geometry_is_measured() -> None:
    """`doctor --model … --weights …` 回答"这套几何在这台机器上验过没有"，不用装权重。"""
    report = _geometry_report("mude", FIXTURES / "mude")
    assert (report["hidden"], report["ffn"], report["num_layers"]) == (2048, 6144, 20)
    assert report["measured"] is True
    assert "note" not in report


def test_doctor_marks_an_unmeasured_geometry_as_an_experimental_reading(monkeypatch) -> None:
    """表外几何不是"不能跑"，是"数值没对拍过"：报告要这么写，与装链时的告警同一判据。"""
    monkeypatch.setattr(apple_platform, "MEASURED_GEOMETRIES", frozenset())
    report = _geometry_report("mude", FIXTURES / "mude")
    assert report["measured"] is False
    assert "实验读数" in report["note"]


def test_doctor_needs_both_the_model_and_its_weights() -> None:
    with pytest.raises(ConfigError):
        _geometry_report("mude", None)
    with pytest.raises(ConfigError):
        _geometry_report(None, FIXTURES / "mude")


def test_doctor_reports_the_artifact_format_even_when_apple_is_unavailable() -> None:
    """产物格式版本是跨机器身份的一部分：苹果不可用时也要能查，别让用户卡在旧产物上。"""
    report = _platform_report("apple")
    assert report["artifact_version"] == 3


# ---- 产物落点：命令行 > 环境变量 > 平台缓存目录 ----------------------------------------


def test_artifacts_dir_precedence(tmp_path: Path, monkeypatch) -> None:
    """落点决定首启要不要重编，所以三档优先级都是契约，不是实现细节。"""
    from sysone.config.storage import default_artifacts_dir

    monkeypatch.delenv("SYS1_ARTIFACTS_DIR", raising=False)
    assert _artifacts_dir() == default_artifacts_dir()

    monkeypatch.setenv("SYS1_ARTIFACTS_DIR", str(tmp_path / "env"))
    assert _artifacts_dir() == tmp_path / "env"
    assert _artifacts_dir(tmp_path / "flag") == tmp_path / "flag"


# ---- compile：与 serve 同一条装载路径，只是不起服务 -------------------------------------


def test_compile_refuses_the_reference_backend(capsys) -> None:
    """参考后端没有产物可编：报错要说清给哪两个后端，而不是让它走到 `--descriptor` 缺失。"""
    code = main(["compile", "--backend", "reference", "--weights", str(FIXTURES / "mude")])
    assert code == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["code"] == "config"
    assert payload["context"]["backend"] == "reference"
    assert payload["context"]["available"] == ["split", "pooled", "vendor"]


# ---- 后端默认值：不给 --backend 时按家族/包身份推断 ------------------------------------


def test_backend_follows_the_model_family() -> None:
    """`--model h2o` 不给 --backend 就是 pooled；reference 不会因为"没给"而被选中。"""
    args = SimpleNamespace(backend=None, model="mude", weights=FIXTURES / "mude", descriptor=None)
    assert _resolve_backend(args) == "split"
    args = SimpleNamespace(backend=None, model="h2o", weights=FIXTURES / "h2o", descriptor=None)
    assert _resolve_backend(args) == "pooled"
    args = SimpleNamespace(backend=None, model="torchcast", weights=FIXTURES / "torchcast",
                           descriptor=None)
    assert _resolve_backend(args) == "pooled"
    # d3 的答案也是按码表读出的：同一个 split 后端，家族只换几何与掩码声明
    for name in ("d3",):
        args = SimpleNamespace(backend=None, model=name, weights=FIXTURES / "torchcast",
                               descriptor=None)
        assert _resolve_backend(args) == "split"


def test_package_markers_name_one_family_or_none(tmp_path: Path) -> None:
    """没给 --model 时按发布包的必需件认家族；认不出**唯一**一家就要求显式指定，不猜。

    d3 与 mude 是同一条 d3 线格式的两份检查点：`decision_config.json` +
    `readout.safetensors` 一模一样，所以这两个目录都落进"认不出唯一家族" —— 猜成 mude
    会拿 `prefix_noncausal` 这个 d3 没训练过的默认掩码读出一份看起来正常的答案。
    torchcast 的 `decision_config.json` 同名，但它多一份 int8 清单、没有码表，于是各认各的。
    """
    def package(name: str, *files: str) -> Path:
        directory = tmp_path / name
        directory.mkdir()
        for file in files:
            (directory / file).write_text("{}", encoding="utf-8")
        return directory

    assert _infer_family(package("h2o", "serve_config.json")) == "h2o"
    assert _infer_family(package("tc", "decision_config.json", "int8_manifest.json")) == "torchcast"
    # SingVL 没有独立码表（读出切绑定 embedding 的行），标记件是它独有的多模态处理器声明。
    assert _infer_family(package("sv", "processor_config.json")) == "singvl"

    # 同一份 d3 线格式的两家：必需件相同，认不出是哪一家
    with pytest.raises(ConfigError) as ambiguous:
        _infer_family(package("wire", "decision_config.json", "readout.safetensors"))
    assert ambiguous.value.context["matches"] == ["d3", "mude"]
    assert "--model" in ambiguous.value.context["hint"]

    # 只带一个共用的文件名：哪一家都不像
    with pytest.raises(ConfigError) as caught:
        _infer_family(package("bare", "decision_config.json"))
    assert "d3" in caught.value.context["hint"]


def test_the_vendor_path_refuses_the_knobs_of_our_chain(monkeypatch, capsys) -> None:
    """厂商后端的前向不在我们的链上：显式给的链形状参数要报错，不能静默丢掉。

    `--W` 的默认值就是 4，所以"给了"只能按"与默认值不同"来认 —— 这里给 0（全装）。
    用一条样例家族而不是 torchcast：它已经改走 pooled，厂商路线本身仍是注册表里可选的一条。
    """
    from sysone import cli
    from sysone.models.recipe import VendorRecipe
    from sysone.models.registry import Family, family
    from sysone.models import registry

    monkeypatch.setitem(registry._REGISTRY, "bare-vendor", Family(
        name="bare-vendor",
        parse_config=family("torchcast").parse_config,
        build_layer_plan=lambda descriptor, config: None,
        strategy="vendor",
        recipe=VendorRecipe(bridge=lambda path: object()),
    ))
    monkeypatch.setattr(cli, "_CLI_FAMILIES", (*cli._CLI_FAMILIES, "bare-vendor"))
    code = main(["run", "--model", "bare-vendor", "--weights", str(FIXTURES / "torchcast"),
                 "--W", "0", "--input", "unused.json"])
    assert code == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["code"] == "config"
    assert payload["context"]["backend"] == "vendor"
    assert payload["context"]["refused"] == ["--W"]


def test_a_backend_that_contradicts_the_family_is_refused(capsys) -> None:
    """`--backend split --model torchcast` 报错并说出两边，不按一条没为它验证过的路跑。"""
    code = main(["run", "--backend", "split", "--model", "torchcast",
                 "--weights", str(FIXTURES / "torchcast"), "--input", "unused.json"])
    assert code == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["context"]["backend"] == "split"
    assert payload["context"]["strategy"] == "pooled"


def test_batch_flag_folds_into_the_scheduler_config() -> None:
    """`--batch` 只是覆盖 `SchedulerConfig.batch` 的默认值，默认值仍只有 dataclass 一个来源。"""
    from sysone.cli import _build_parser, _scheduler

    def scheduler(*argv: str):
        args = _build_parser().parse_args(["run", "--input", "x", *argv])
        return _scheduler(args, None, max_len=4096)

    assert scheduler().batch == 4, "不给就是 scheduler 的默认值"
    assert scheduler().max_len == 4096
    assert scheduler("--batch", "2").batch == 2
    assert scheduler("--batch", "1").batch == 1, "1 = 不攒批"
    assert scheduler("--batch", "0").batch == 0, "0 也当不攒批（引擎会再取 max(1, …)）"


def test_the_tower_is_unloaded_after_every_request_by_default() -> None:
    """非文本的塔不占常态内存：默认窗口 0（每请求后即卸），留着是显式的选择。

    `_chain_knobs_given` 按"与默认值不同"认显式，所以默认值一改，`--vision-idle 0` 就不再是
    vendor 那条路要拒的旋钮 —— 这两件事必须由同一个常量决定，否则默认值只是文档里的一句话。
    """
    from sysone.cli import _VISION_IDLE, _build_parser, _chain_knobs_given

    def knobs(*argv: str) -> list[str]:
        return _chain_knobs_given(_build_parser().parse_args(["run", "--input", "x", *argv]))

    assert _VISION_IDLE == 0.0, "闲时 0 常驻是默认契约"
    assert knobs() == []
    assert knobs("--vision-idle", "0") == [], "与默认值相同就是没给"
    assert knobs("--vision-idle", "60") == ["--vision-idle"]


def test_backend_is_inferred_from_the_weights_package(tmp_path: Path) -> None:
    """没给 --model 时按必需件认家族；只有**一家**像时才推断。

    d3 与 mude 是同一组必需件（d3 线格式的两份检查点），两家同时命中就不猜 ——
    猜成 mude 会拿到另外两家没训练过的 `prefix_noncausal` 默认掩码。
    """
    (tmp_path / "serve_config.json").write_text("{}", encoding="utf-8")
    args = SimpleNamespace(backend=None, model=None, weights=tmp_path, descriptor=None)
    assert _resolve_backend(args) == "pooled"
    assert args.model == "h2o"

    wired = tmp_path / "wire"
    wired.mkdir()
    for name in ("decision_config.json", "readout.safetensors"):
        (wired / name).write_text("{}", encoding="utf-8")
    with pytest.raises(ConfigError) as caught:
        _resolve_backend(SimpleNamespace(backend=None, model=None, weights=wired, descriptor=None))
    assert caught.value.context["matches"] == ["d3", "mude"]


def test_backend_inference_asks_for_an_explicit_model_on_an_unknown_package(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        _resolve_backend(
            SimpleNamespace(backend=None, model=None, weights=tmp_path, descriptor=None)
        )


def test_compile_reports_the_startup_split_and_shuts_down(monkeypatch, capsys) -> None:
    """装完就卸：这条命令是部署时跑一次的，不该留一条链占着 ANE。"""
    from sysone import cli
    from sysone.config import StorageConfig
    from sysone.domain.work import CompilationTimes

    calls: list[str] = []

    class FakeEngine:
        config = SimpleNamespace(storage=StorageConfig(artifacts_dir=Path("/var/art")))

        def start(self) -> CompilationTimes:
            calls.append("start")
            return CompilationTimes(load_s=1.0, compile_s=12.5, warmup_s=0.5)

        def shutdown(self) -> None:
            calls.append("shutdown")

    monkeypatch.setattr(cli, "_build_engine", lambda args, mode: FakeEngine())
    code = cli.main(["compile", "--backend", "split", "--weights", str(FIXTURES / "mude")])
    assert code == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "sysone compile: 14.00s (load 1.00s, compile 12.50s, warmup 0.50s)"
    assert lines[1] == "artifacts: /var/art"
    assert calls == ["start", "shutdown"]


# ---- doctor 的产物一栏：与装载同一份判据 -----------------------------------------------


def test_doctor_reports_every_layer_as_outstanding_on_a_cold_cache(tmp_path: Path) -> None:
    from helpers import make_mude_checkpoint
    from sysone.cli import _artifact_report

    root = make_mude_checkpoint(tmp_path / "model")
    report = _artifact_report("mude", root, bucket=256, artifacts=tmp_path / "art")

    assert report["layers"] == {"reusable": 0, "total": 2}
    assert report["missing_layers"] == [0, 1]
    assert report["s"] == 256
    # 目录键把家族名与桶宽都编进去：mude 与 h2o 的权重目录常常都叫 `model`
    assert report["directory"] == str(tmp_path / "art" / f"mude-{root.name}" / "s256")
    assert "sysone compile" in report["note"]


def test_doctor_says_when_every_layer_can_be_reused(tmp_path: Path, monkeypatch) -> None:
    """全命中就必然是编译段 0 秒 —— 报告直接这么说，别让人自己去推。"""
    from helpers import make_mude_checkpoint
    from sysone.cli import _artifact_report
    from sysone.devices.apple import chain

    root = make_mude_checkpoint(tmp_path / "model")
    monkeypatch.setattr(chain, "cached_program", lambda directory, *, source, s: object())

    report = _artifact_report("mude", root, bucket=256, artifacts=tmp_path / "art")
    assert report["layers"] == {"reusable": 2, "total": 2}
    assert "missing_layers" not in report
    assert "0 秒" in report["note"]


def test_doctor_reports_artifacts_alongside_geometry(tmp_path: Path, capsys) -> None:
    from helpers import make_mude_checkpoint

    root = make_mude_checkpoint(tmp_path / "model")
    code = main([
        "doctor", "--platform", "reference", "--model", "mude",
        "--weights", str(root), "--artifacts", str(tmp_path / "art"),
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["artifacts"]["layers"]["total"] == 2
    assert payload["geometry"]["family"] == "mude"


# ---- 首启下载：显式 --weights 只读，省略才按发布来源下载 --------------------------------


def test_weights_flag_is_optional_for_model_commands() -> None:
    from sysone.cli import _build_parser

    args = _build_parser().parse_args(["compile", "--model", "mude", "--offline"])
    assert args.weights is None
    assert args.offline is True
    assert args.quant == "int8"


def test_weights_dir_precedence(tmp_path: Path, monkeypatch) -> None:
    from sysone.cli import _weights_dir
    from sysone.config.storage import default_weights_dir

    monkeypatch.delenv("SYS1_WEIGHTS_DIR", raising=False)
    assert _weights_dir() == default_weights_dir()

    monkeypatch.setenv("SYS1_WEIGHTS_DIR", str(tmp_path / "env"))
    assert _weights_dir() == tmp_path / "env"


def test_explicit_missing_weights_is_a_config_error_and_never_downloads(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    from sysone.weights import fetch

    def never(*_args, **_kwargs):
        raise AssertionError("explicit --weights must not download")

    monkeypatch.setattr(fetch, "ensure_weights", never)
    code = main(["compile", "--model", "mude", "--weights", str(tmp_path / "absent")])
    assert code == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["code"] == "config"
    assert payload["context"]["path"].endswith("absent")


def test_offline_without_local_weights_names_the_manual_command(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    monkeypatch.setenv("SYS1_WEIGHTS_DIR", str(tmp_path / "weights"))
    code = main(["compile", "--model", "mude", "--offline", "--artifacts", str(tmp_path / "art")])
    assert code == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["code"] == "download"
    assert "hf download qybaihe/mude-1.0" in payload["context"]["command"]
    assert payload["context"]["revision"] == "0b6a5c4ea9ca2267707a28ff568737f54ae219c5"


def test_omitted_weights_resolves_the_published_source(tmp_path: Path, monkeypatch) -> None:
    import sysone.weights.fetch as fetch
    from sysone.cli import _resolve_weights

    seen: dict[str, object] = {}

    def fake(source, root, *, offline=False):
        seen.update(source=source, root=Path(root), offline=offline)
        return FIXTURES / "mude"

    monkeypatch.setattr(fetch, "ensure_weights", fake)
    resolved = _resolve_weights(None, "mude", offline=False, root=tmp_path)
    assert resolved == FIXTURES / "mude"
    assert seen["source"].repo_id == "qybaihe/mude-1.0"
    assert seen["root"] == tmp_path
    assert seen["offline"] is False


def test_quant_modes_get_separate_artifact_directories(tmp_path: Path) -> None:
    """两档共用目录会让每切一次就互相顶掉 sidecar；doctor 与装载必须同一条路径规则。"""
    from helpers import make_mude_checkpoint
    from sysone.cli import _artifact_report

    root = make_mude_checkpoint(tmp_path / "model")
    int8 = _artifact_report("mude", root, bucket=256, artifacts=tmp_path / "art", quant="int8")
    fp16 = _artifact_report("mude", root, bucket=256, artifacts=tmp_path / "art", quant="fp16")

    assert int8["quant"] == "int8"
    assert fp16["quant"] == "fp16"
    assert str(int8["directory"]).endswith("s256")
    assert str(fp16["directory"]).endswith("s256-fp16")
    assert int8["directory"] != fp16["directory"]
