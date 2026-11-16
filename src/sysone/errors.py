"""错误层次：所有引擎异常都带稳定的 `code` 与 HTTP `status`。

服务层只按 `code` 与 `status` 决定回包，不靠字符串匹配；错误上下文统一放进 `context`，
便于日志与 `/metrics` 聚合（见 docs/sysone/interfaces.md 第 9 节）。
"""


class SysOneError(Exception):
    """引擎错误的基类。"""

    code = "internal"
    status = 500

    def __init__(self, message: str, **context: object) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    def __str__(self) -> str:
        if not self.context:
            return self.message
        detail = ", ".join(f"{key}={value!r}" for key, value in self.context.items())
        return f"{self.message} ({detail})"


class ConfigError(SysOneError):
    """配置非法：启动期错误，不服务。"""

    code = "config"


class UnknownModelError(SysOneError):
    """`--model` 未知，或家族未注册。"""

    code = "unknown_model"
    status = 400


class UnsupportedModeError(SysOneError):
    """请求的注意力模式不在该模型的能力集合里。"""

    code = "unsupported_mode"
    status = 400


class UnsupportedKindError(SysOneError):
    """请求的题型不在该模型的读出能力里（线格式认、这个模型答不了）。

    与 `ConfigError` 分开：这是**坏请求**不是我们的配置坏了，客户端据此知道不要重试。
    今天只有 mude 会抛（码表读出没有 score 的档位模板）。
    """

    code = "unsupported_kind"
    status = 400


class UnsupportedPrecisionError(SysOneError):
    """精度 / 设备 / 权重组合不可用。"""

    code = "unsupported_precision"


class MissingTensorError(SysOneError):
    """权重缺键，或文件不存在。"""

    code = "missing_tensor"


class BudgetExceededError(SysOneError):
    """内存预算不足。"""

    code = "budget_exceeded"


class ArtifactStaleError(SysOneError):
    """编译产物过期且不允许重编译。"""

    code = "artifact_stale"


class NoKernelError(SysOneError):
    """selector 找不到可用的算子实现。"""

    code = "no_kernel"


class MaskPlanError(SysOneError):
    """掩码计划不满足不变量（构造期错误）。"""

    code = "mask_plan"


class QueueFullError(SysOneError):
    """准入队列满。"""

    code = "queue_full"
    status = 429


class NotReadyError(SysOneError):
    """编译 / 预热尚未完成。"""

    code = "not_ready"
    status = 503


class RequestTooLongError(SysOneError):
    """序列超过 `max_len`。"""

    code = "request_too_long"
    status = 422


class UnsupportedMediaError(SysOneError):
    """模态不支持，或视觉塔未启用。"""

    code = "unsupported_media"
    status = 422


class ReadoutError(SysOneError):
    """读不出合法答案（标签缺失 / 概率退化）。"""

    code = "readout"
    status = 422


class RegistrationError(SysOneError):
    """模型家族重复注册，或注册冲突。"""

    code = "registration"
