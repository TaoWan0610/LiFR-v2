from .m3ed import M3ED


class SHFEmergy(M3ED):
    """SHF-Emerge MoGe-style RGB/depth/semantic samples."""

    def __str__(self):
        return f'SHFEmergy(split={self.split})'
