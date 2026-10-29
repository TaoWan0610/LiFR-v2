from .m3ed_anytime import M3EDAnytime


class SHFEmergyAnytime(M3EDAnytime):
    """SHF-Emerge 50 ms causal RGB/event samples."""

    frame_interval_ms = 50

    def __str__(self):
        return f'SHFEmergyAnytime(split={self.split}, steps={self.time_window_steps})'
