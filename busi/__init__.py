"""busi — Attention U-Net extended to BUSI breast-ultrasound segmentation.

Submodules (data, model, losses, metrics, train, viz) are imported lazily by
the caller so that `import busi` and `import busi.config` work even before the
deep-learning stack (torch, albumentations, ...) is installed.
"""

__version__ = "0.1.0"
