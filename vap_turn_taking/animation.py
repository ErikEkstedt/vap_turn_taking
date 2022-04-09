from os.path import dirname
from os import makedirs

import torch
import torchaudio
import subprocess
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle

from vap_turn_taking.plot_utils import plot_vad_oh

mpl.use("agg")

"""
This works if FFMPEG is installed correctly. The below line should work.

conda install -c conda-forge ffmpeg

"""


class VAPanimation:
    def __init__(
        self,
        p,
        p_bc,
        p_class,
        vap_bins,
        x,
        va,
        events=None,
        window_duration=10,
        frame_hz=100,
        sample_rate=16000,
        fps=20,
        dpi=200,
        bin_frames=[20, 40, 60, 80],
    ) -> None:
        """"""
        # Model output
        self.p = p
        self.p_bc = p_bc
        self.p_class = p_class

        self.vap_bins = vap_bins
        self.weighted_oh = self._weighted_oh(p_class, vap_bins)
        self.best_p, self.best_idx = p_class.max(dim=-1)

        # Model input
        self.x = x  # Waveform
        self.va = va  # Voice Activity
        self.events = events  # events

        # Parameters
        self.frame_hz = frame_hz
        self.sample_rate = sample_rate
        self.window_duration = window_duration
        self.window_frames = self.window_duration * self.frame_hz
        self.center = self.window_frames // 2
        self.bin_frames = bin_frames

        # Animation params
        self.dpi = dpi
        self.fps = fps
        self.frame_step = int(100.0 / self.fps)

        self.plot_kwargs = {
            "A": {"color": "b"},
            "B": {"color": "orange"},
            "va": {"alpha": 0.6},
            "bc": {"color": "darkgreen", "alpha": 0.6},
            "vap": {"ylim": [-0.5, 0.5], "width": 3},
            "current": {"width": 5},
        }

        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 4))
        self.pred_ax = self.ax.twinx()
        self.vap_ax = self.ax.twinx()
        self.vap_patches = []
        self.draw_vap_patches = True

        self.draw_static()
        self.started = False

    def _weighted_oh(self, p_class, vap_bins):
        weighted_oh = p_class.unsqueeze(-1).unsqueeze(-1) * vap_bins
        weighted_oh = weighted_oh.sum(dim=1)  # sum all class onehot
        return weighted_oh

    def set_axis_lim(self):
        self.ax.set_xlim([0, self.window_frames])
        # PROBS
        self.pred_ax.set_xlim([0, self.window_frames])
        self.pred_ax.set_yticks([])
        # VAP
        self.vap_ax.set_ylim([-1, 1])
        self.vap_ax.set_yticks([])

    def draw_static(self):
        self.current_line = self.pred_ax.vlines(
            self.center,
            ymin=-1,
            ymax=1,
            color="r",
            linewidth=self.plot_kwargs["current"]["width"],
        )

        # VAP BOX
        s = (torch.tensor(self.bin_frames).cumsum(0) + self.center).tolist()
        ymin, ymax = self.plot_kwargs["vap"]["ylim"]
        w = s[-1] - self.center
        h = ymax - ymin
        # white background

        vap_background = Rectangle(
            xy=[self.center, ymin], width=w, height=h, color="w", alpha=1
        )
        self.vap_ax.add_patch(vap_background)
        self.vap_ax.vlines(
            s,
            ymin=ymin,
            ymax=ymax,
            color="k",
            linewidth=self.plot_kwargs["vap"]["width"],
        )
        self.vap_ax.plot(
            [self.center, s[-1]],
            [ymin, ymin],
            color="k",
            linewidth=self.plot_kwargs["vap"]["width"],
        )
        self.vap_ax.plot(
            [self.center, s[-1]],
            [ymax, ymax],
            color="k",
            linewidth=self.plot_kwargs["vap"]["width"],
        )

    def clear_ax(self):
        self.ax.cla()
        self.pa.remove()
        self.pb.remove()
        self.p_bc_a.remove()
        self.p_bc_b.remove()

        for i in range(len(self.vap_patches)):
            self.vap_patches[i].remove()

    def draw_step(self, step=0):
        if not self.started:
            self.started = True
        else:
            self.clear_ax()

        end = step + self.window_frames

        _ = plot_vad_oh(
            self.va[step:end], ax=self.ax, alpha=self.plot_kwargs["va"]["alpha"]
        )

        # Draw probalitiy curves
        (self.pa,) = self.pred_ax.plot(
            self.p[step:end, 0], color=self.plot_kwargs["A"]["color"]
        )
        (self.p_bc_a,) = self.pred_ax.plot(
            self.p_bc[step:end, 0], color=self.plot_kwargs["bc"]["color"]
        )
        (self.pb,) = self.pred_ax.plot(
            self.p[step:, 1] - 1, color=self.plot_kwargs["B"]["color"]
        )
        (self.p_bc_b,) = self.pred_ax.plot(
            self.p_bc[step:end, 1] - 1, color=self.plot_kwargs["bc"]["color"]
        )

        # draw weighted oh projection

        h = self.plot_kwargs["vap"]["ylim"][-1]

        jj = 0
        for speaker, sp_color in zip(
            [0, 1], [self.plot_kwargs["A"]["color"], self.plot_kwargs["B"]["color"]]
        ):
            bf_cum = 0
            for bin, bf in enumerate(self.bin_frames):

                alpha = self.weighted_oh[step + self.center, speaker, bin].item()

                if self.draw_vap_patches:
                    start = self.center + bf_cum
                    vap_patch = Rectangle(
                        xy=[start, -h * speaker],
                        width=bf,
                        height=h,
                        color=sp_color,
                        alpha=alpha,
                    )
                    # self.vap_patches.append(vap_patch)
                    self.vap_ax.add_patch(vap_patch)
                else:
                    self.vap_ax.patches[jj + 1].set_alpha(alpha)
                # self.vap_patches.append(p)
                bf_cum += bf
                jj += 1

        self.draw_vap_patches = False

    def update(self, step):
        self.draw_step(step)
        self.set_axis_lim()
        return []

    def ffmpeg_call(self, out_path, vid_path, wav_path):
        """
        Overlay the static image on top of the video (saved with transparency) and
        adding the audio.

        Arguments:
            vid_path:  path to temporary dynamic video file
            wav_path:  path to temporary audio file
            img_path:  path to temporary static image
            out_path:  path to save final video to
        """
        cmd = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            vid_path,
            "-i",
            wav_path,
            "-vcodec",
            "libopenh264",
            out_path,
        ]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        p.communicate()

    def save_video(self, path="test.mp4"):
        tmp_video_path = "/tmp/vap_video_ani.mp4"
        tmp_wav_path = "/tmp/vap_video_audio.wav"
        n_frames = self.p.shape[0] - self.center

        if len(dirname(path)) > 0:
            makedirs(dirname(path), exist_ok=True)

        sample_offset = int(self.sample_rate * self.center / self.frame_hz)

        # SAVE tmp waveform
        torchaudio.save(
            tmp_wav_path,
            self.x[sample_offset:].unsqueeze(0),
            sample_rate=self.sample_rate,
        )

        # Save matplot video
        moviewriter = animation.FFMpegWriter(
            fps=self.fps  # , codec="libopenh264", extra_args=["-threads", "16"]
        )

        with moviewriter.saving(self.fig, tmp_video_path, dpi=self.dpi):
            for step in tqdm(range(0, n_frames, self.frame_step)):
                _ = self.update(step)
                moviewriter.grab_frame()

        self.ffmpeg_call(path, tmp_video_path, tmp_wav_path)


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--video_data", type=str)
    parser.add_argument("--filename", type=str, default="video.mp4")
    args = parser.parse_args()

    if not args.filename.endswith(".mp4"):
        args.filename += ".mp4"

    data = torch.load(args.video_data)
    # x = torch.rand(1, 16000)  # Waveform
    # events = {"shift", "hold", "backchannel"}  # Events
    # va = torch.rand(1, 100, 2)  # Voice Activity
    # p = torch.rand(1, 100, 2)  # Next speaker probs
    # p_bc = torch.rand(1, 100, 2)  # Backchannel probs
    # p_class = torch.rand(1, 100)  # Backchannel probs
    # vap_bins = torch.rand(256, 2, 4)  # the binary representation of the bin window

    # Save video
    ani = VAPanimation(
        p=data["p"],
        p_bc=data["p_bc"],
        p_class=data["logits"].softmax(-1),
        vap_bins=data["vap_bins"],
        x=data["waveform"],
        va=data["va"],
        events=None,
        fps=20,
    )
    ani.save_video(args.filename)
