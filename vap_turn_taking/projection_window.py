import torch
import torch.nn as nn

from typing import List

from vap_turn_taking.functional import bin_times_to_frames


class ProjectionWindow(nn.Module):
    def __init__(
        self,
        bin_times: List = [0.2, 0.4, 0.6, 0.8],
        frame_hz: int = 50,
        threshold_ratio: float = 0.5,
    ):
        super().__init__()
        self.bin_times = bin_times
        self.frame_hz = frame_hz
        self.threshold_ratio = threshold_ratio

        self.bin_frames = bin_times_to_frames(bin_times, frame_hz)
        self.n_bins = len(self.bin_frames)
        self.total_bins = self.n_bins * 2
        self.horizon = sum(self.bin_frames)

    def __repr__(self) -> str:
        s = "VAPLabel(\n"
        s += f"  bin_times: {self.bin_times}\n"
        s += f"  bin_frames: {self.bin_frames}\n"
        s += f"  frame_hz: {self.frame_hz}\n"
        s += f"  thresh: {self.threshold_ratio}\n"
        s += ")\n"
        return s

    def projection(self, va: torch.Tensor) -> torch.Tensor:
        """
        Extract projection (bins)
        (b, n, c) -> (b, N, c, M), M=horizon window size, N=valid frames

        Arguments:
            va:         torch.Tensor (B, N, C)

        Returns:
            vaps:       torch.Tensor (B, m, C, M)

        """
        # Shift to get next frame projections
        return va[..., 1:, :].unfold(dimension=-2, size=sum(self.bin_frames), step=1)

    def projection_bins(self, projection_window: torch.Tensor) -> torch.Tensor:
        """
        Iterate over the bin boundaries and sum the activity
        for each channel/speaker.
        divide by the number of frames to get activity ratio.
        If ratio is greater than or equal to the threshold_ratio
        the bin is considered active
        """

        start = 0
        v_bins = []
        for b in self.bin_frames:
            end = start + b
            m = projection_window[..., start:end].sum(dim=-1) / b
            m = (m >= self.threshold_ratio).float()
            v_bins.append(m)
            start = end
        return torch.stack(v_bins, dim=-1)  # (*, t, c, n_bins)

    def __call__(self, va: torch.Tensor) -> torch.Tensor:
        projection_windows = self.projection(va)
        return self.projection_bins(projection_windows)


if __name__ == "__main__":

    data = torch.load("example/vap_data.pt")
    labeler = ProjectionWindow()

    vad = data["shift"]["vad"]
    proj_win = labeler(vad)
    print("vad: ", tuple(vad.shape))
    print("proj_win: ", tuple(proj_win.shape))
