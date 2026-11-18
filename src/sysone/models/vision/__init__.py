"""视觉前端：像素与合流（M4 的 P1 / P3）。

两条**不碰设备**的纯函数链：

* `preprocess` —— 图片字节进来（`devices/apple/image.py` 解码）、归一图出去：几何
  （`smart_resize`）、一次可分离双三次、一条仿射。出口是塔的 `encode(img)` 直接吃的
  `(H, W, 3)` fp32，**不含 patchify**（patchify 在塔里，见 `devices/apple/vision.py`）。
* `merge`（P3）—— 占位符展开、M-RoPE 位置、行落点。

放在 `models/` 而不是 `devices/apple/`，因为它们是两颗设备之间的**契约**：塔的 `encode` 与
前门都只认这里定义的形状，换一台机器（甚至换回 CPU 参考实现）也不该改这两件事。
"""
