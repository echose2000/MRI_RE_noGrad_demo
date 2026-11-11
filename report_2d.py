"""Plots and saved results for the 2D reconstruction experiment."""

from __future__ import annotations

import csv
import json
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


class StageMetrics(NamedTuple):
    psnr: float
    ssim: float


class ReconstructionReport:
    def __init__(self, reference, zero_filled, steps, *, show):
        self.reference = self._image_array(reference)
        self.zero_filled = self._image_array(zero_filled)
        self.stage_names = [f"stage_{depth}" for depth in range(1, steps)] + ["final"]
        self.show = show
        self.history = []
        self.latest_images = []
        if show:
            self.overview, self.image_handles = self._plot_overview()
            plt.show(block=False)

    def record(self, iteration, images, scores):
        self.history.append((iteration, scores))
        self.latest_images = images
        if self.show:
            self._update_overview(self.image_handles)
            self.overview.canvas.draw_idle()
            plt.pause(0.001)

    def finish(self, model_name, sampling_rate, result_dir):
        iterations, scores = zip(*self.history)
        final_psnrs = [stages[-1].psnr for stages in scores]
        final_ssims = [stages[-1].ssim for stages in scores]
        best_psnr = int(np.argmax(final_psnrs))
        best_ssim = int(np.argmax(final_ssims))
        print(f"Final PSNR: {final_psnrs[-1]:.4f} dB, SSIM: {final_ssims[-1]:.4f}")
        print(f"Best PSNR: {final_psnrs[best_psnr]:.4f} dB at iteration {iterations[best_psnr]}")
        print(f"Best SSIM: {final_ssims[best_ssim]:.4f} at iteration {iterations[best_ssim]}")
        for depth, score in enumerate(scores[-1][:-1], start=1):
            print(f"Stage {depth}: PSNR {score.psnr:.4f} dB, SSIM {score.ssim:.4f}")

        if not self.show and result_dir is None:
            return

        if self.show:
            overview = self.overview
        else:
            overview, handles = self._plot_overview()
            self._update_overview(handles)
        curves = self._plot_metric_curves(iterations, scores)
        recurrent = self._plot_recurrent_metrics(scores[-1])

        if result_dir is not None:
            result_dir.mkdir(parents=True, exist_ok=True)
            for filename, figure in (
                ("reconstruction_overview.png", overview),
                ("metric_curves.png", curves),
                ("recurrent_steps_metrics.png", recurrent),
            ):
                figure.savefig(result_dir / filename, dpi=150, bbox_inches="tight")
            self._save_metrics(result_dir, model_name, sampling_rate)

        if self.show:
            plt.show(block=True)

    @staticmethod
    def _image_array(image):
        return image[0, 0].detach().cpu().numpy()

    def _figure(self, figsize):
        if self.show:
            return plt.figure(figsize=figsize)
        return Figure(figsize=figsize)

    def _plot_overview(self):
        stage_titles = [f"Recursive stage {depth}" for depth in range(1, len(self.stage_names))]
        titles = ["GT", "Undersampled", *stage_titles, "Final reconstruction", "Error map"]
        columns = min(7, len(titles))
        rows = (len(titles) + columns - 1) // columns
        figure = self._figure((4 * columns, 4 * rows))
        axes = figure.subplots(rows, columns, squeeze=False)
        for axis in axes.flat:
            axis.set_axis_off()

        handles = []
        blank = np.zeros_like(self.reference)
        panels = [self.reference, self.zero_filled, *([blank] * (len(titles) - 2))]
        for axis, title, panel in zip(axes.flat, titles, panels):
            axis.set_title(title)
            if title == "Error map":
                handles.append(axis.imshow(panel, cmap="viridis", vmin=0, vmax=0.1))
            else:
                handles.append(axis.imshow(panel, cmap="gray", vmin=0, vmax=1))
        figure.tight_layout()
        return figure, handles[2:]

    def _update_overview(self, handles):
        images = [self._image_array(image) for image in self.latest_images]
        for handle, image in zip(handles, images):
            handle.set_data(image)
        handles[-1].set_data(np.abs(images[-1] - self.reference))

    def _plot_metric_curves(self, iterations, scores):
        figure = self._figure((13, 5))
        ax_psnr, ax_ssim = figure.subplots(1, 2)
        for name, stage_scores in zip(self.stage_names, zip(*scores)):
            style = {"linewidth": 2} if name == "final" else {"linestyle": "--"}
            ax_psnr.plot(iterations, [score.psnr for score in stage_scores], label=name, **style)
            ax_ssim.plot(iterations, [score.ssim for score in stage_scores], label=name, **style)

        ax_psnr.set(xlabel="training iteration", ylabel="PSNR (dB)", title="PSNR during recursion")
        ax_ssim.set(xlabel="training iteration", ylabel="SSIM", title="SSIM during recursion")
        legend_columns = (len(self.stage_names) + 9) // 10
        for axis in (ax_psnr, ax_ssim):
            axis.grid(alpha=0.25)
            axis.legend(ncol=legend_columns, fontsize="small")
        figure.tight_layout()
        return figure

    def _plot_recurrent_metrics(self, scores):
        figure = self._figure((8, 5))
        ax_psnr = figure.subplots()
        ax_ssim = ax_psnr.twinx()
        depths = np.arange(1, len(scores) + 1)
        ticks = np.unique(np.linspace(0, len(scores) - 1, min(10, len(scores)), dtype=int))
        marker_stride = max(1, (len(scores) + 11) // 12)
        markers = sorted(set(range(0, len(scores), marker_stride)) | {len(scores) - 1})
        lines = []
        for axis, values, name, color, marker, precision, offset in (
            (ax_psnr, [score.psnr for score in scores], "PSNR", "tab:blue", "o", 2, 7),
            (ax_ssim, [score.ssim for score in scores], "SSIM", "tab:red", "s", 4, -15),
        ):
            lines.extend(axis.plot(
                depths, values, label=name, color=color, marker=marker,
                markevery=markers, linewidth=2,
            ))
            for index in ticks:
                value = values[index]
                axis.annotate(
                    f"{value:.{precision}f}", (depths[index], value), textcoords="offset points",
                    xytext=(0, offset), ha="center", color=color,
                )

        ax_psnr.legend(handles=lines, loc="best")
        ax_psnr.set_xlabel("recurrent steps")
        ax_psnr.set_ylabel("PSNR (dB)", color="tab:blue")
        ax_ssim.set_ylabel("SSIM", color="tab:red")
        ax_psnr.set_xticks(depths[ticks])
        ax_psnr.set_xlim(1, max(1.5, len(scores)))
        ax_psnr.grid(alpha=0.25)
        figure.suptitle("Final reconstruction quality vs. recurrent steps")
        figure.tight_layout()
        return figure

    def _save_metrics(self, result_dir, model_name, sampling_rate):
        with (result_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["iteration", "stage", "psnr", "ssim"])
            for iteration, scores in self.history:
                for name, score in zip(self.stage_names, scores):
                    writer.writerow([iteration, name, score.psnr, score.ssim])

        _, scores = self.history[-1]
        final = scores[-1]
        summary = {
            "model": model_name,
            "sampling_rate": sampling_rate,
            "final_psnr": final.psnr,
            "final_ssim": final.ssim,
            "recurrent_steps": list(range(1, len(scores) + 1)),
            "recurrent_psnr": [score.psnr for score in scores],
            "recurrent_ssim": [score.ssim for score in scores],
        }
        with (result_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
