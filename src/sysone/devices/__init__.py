"""设备层：平台能力、算子实现选择、参考实现（reference）。

依赖方向是 `devices ← domain`（+ `weights`），不反向依赖 `engine`；平台实现只通过
`Platform` Protocol 暴露能力与实现类，不持有任何张量。
"""
