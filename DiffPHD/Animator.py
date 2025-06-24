from .RadarNetwork import RadarNetwork
from .DiffPHD import DiffPHD
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from .helpers import plot_cov_ellipse
from datetime import datetime
import os


class Animator:
    def __init__(
        self,
        radar_network: RadarNetwork,
        diff_phd: DiffPHD,
        save: bool = True,
        fps: int = 5,
        loop: bool = True,
        figsize: tuple[int, int] = (15, 15),
    ):
        """
        Initialize the animator.
        """
        self.radar_network = radar_network
        self.diff_phd = diff_phd
        self.save = save
        self.fps = fps
        self.loop = loop
        self.figsize = figsize

        self.radar_snaps: list[RadarNetwork] = []
        self.phd_snaps: list[DiffPHD] = []

        self.colors = self.assign_radar_colors()
        self.text_offsets = self.assign_radar_text_offsets()

        self.snap()

    def assign_radar_colors(self):
        """
        Assign colors to the radars.
        """
        colors = [
            "b",
            "r",
            "g",
            "c",
            "m",
            "orange",
            "purple",
            "brown",
            "pink",
            "olive",
        ]
        color_dict = {}
        for i, radar in enumerate(self.radar_network.radars):
            color_dict[radar.id] = colors[i % len(colors)]

        return color_dict

    def assign_radar_text_offsets(self):
        """
        Assign text offsets to the radars.
        """
        n_radars = len(self.radar_network.radars)
        base = 0.005
        step = 0.02
        offsets = [base + i * step for i in range(-(n_radars // 2), n_radars // 2 + 1)]
        offset_dict = {}
        for i, radar in enumerate(self.radar_network.radars):
            offset_dict[radar.id] = offsets[i % len(offsets)]

        return offset_dict

    def snap(self):
        """
        Take a snapshot of the current state of the simulation.
        """
        self.radar_snaps.append(self.radar_network.copy())
        self.phd_snaps.append(self.diff_phd.copy())

    def animate(self, show: bool = True) -> FuncAnimation:
        """
        Animate the simulation.

        :param show: whether to show the animation
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        plt.tight_layout()

        def update(frame):
            ax.clear()

            radar_network: RadarNetwork = self.radar_snaps[frame]
            diff_phd: DiffPHD = self.phd_snaps[frame]

            # plot global clutter
            clutter = radar_network.global_clutter
            clutter_x = [point[0] for point in clutter]
            clutter_y = [point[1] for point in clutter]
            ax.scatter(clutter_x, clutter_y, color="grey", marker="x", s=10)

            # plot airports
            for airport in radar_network.birth_intensity:
                mean = airport.means[0]
                cov = airport.covariances[0]
                ax.scatter(mean[0], mean[1], color="k", s=10, marker="+")
                ellipse = plot_cov_ellipse(
                    mean[:2], cov[:2, :2], n_std=3, color="k", linestyle="--"
                )
                ax.add_artist(ellipse)

            # plot trajectories
            for trajectory in radar_network.target_trajectories:
                trajectory_x = [point[0] for point in trajectory]
                trajectory_y = [point[1] for point in trajectory]
                ax.plot(trajectory_x, trajectory_y, color="gray", alpha=0.5)

            # plot radars neighborhoods
            plotted_edges = set()
            for radar in radar_network.radars:
                color = self.colors[radar.id]
                for neighbor_id in diff_phd.graph[radar.id]:
                    edge = tuple(sorted([radar.id, neighbor_id]))
                    if edge in plotted_edges:
                        continue
                    plotted_edges.add(edge)
                    neighbor_fov = diff_phd.fovs[neighbor_id]
                    neighbor_center = neighbor_fov[0]
                    ax.plot(
                        [radar.center[0], neighbor_center[0]],
                        [radar.center[1], neighbor_center[1]],
                        color="lightgrey",
                        linestyle="--",
                    )

            # plot radars ranges
            for radar in radar_network.radars:
                color = self.colors[radar.id]
                circle = plt.Circle(radar.center, radar.radius, fill=False, color=color)
                ax.add_artist(circle)
                ax.text(radar.center[0], radar.center[1], radar.id, color=color)

            # plot radars clutter
            for radar in radar_network.radars:
                clutter = radar.local_clutter
                clutter_x = [point[0] for point in clutter]
                clutter_y = [point[1] for point in clutter]
                ax.scatter(clutter_x, clutter_y, color="lightgrey", marker="x", s=10)

            # plot targets
            targets = radar_network.targets
            targets_x = [target[0] for target in targets]
            targets_y = [target[1] for target in targets]
            ax.scatter(targets_x, targets_y, color="black", s=100, marker="+")

            # plot radars target measurements
            for radar in radar_network.radars:
                measurements = radar.detected_targets
                measurements_x = [point[0] for point in measurements]
                measurements_y = [point[1] for point in measurements]
                color = self.colors[radar.id]
                ax.scatter(
                    measurements_x,
                    measurements_y,
                    color=color,
                    s=10,
                    marker="x",
                    alpha=0.5,
                )

            # plot estimates
            for id, phd in diff_phd.phds.items():
                color = self.colors[id]
                # plot means
                estimates = phd.estimates
                estimates_x = [mean[0] for mean in estimates.means]
                estimates_y = [mean[1] for mean in estimates.means]
                ax.scatter(estimates_x, estimates_y, color=color, s=10)

                # add ID labels
                width = radar_network.rectangle[1, 0] - radar_network.rectangle[0, 0]
                height = radar_network.rectangle[1, 1] - radar_network.rectangle[0, 1]
                offset_x = 0.02 * width
                offset_y = self.text_offsets[id] * height
                labels_x = [x + offset_x for x in estimates_x]
                labels_y = [y + offset_y for y in estimates_y]
                for i, label in enumerate(estimates.weights):
                    ax.text(labels_x[i], labels_y[i], f"{id}: {label:.2f}", color=color)

                # plot ellipses
                for i, mean in enumerate(estimates.means):
                    cov = estimates.covariances[i][:2, :2] + phd.model.R
                    ellipse = plot_cov_ellipse(mean[:2], cov, n_std=3, color=color)
                    ax.add_artist(ellipse)

            # set plot parameters
            margin = 0.08
            margin_width = margin * width
            margin_height = margin * height
            ax.set_xlim(
                radar_network.rectangle[0, 0] - margin_width,
                radar_network.rectangle[1, 0] + margin_width,
            )
            ax.set_ylim(
                radar_network.rectangle[0, 1] - margin_height,
                radar_network.rectangle[1, 1] + margin_height,
            )
            ax.set_aspect("equal", adjustable="box")
            ax.set_axis_off()
            ax.set_title(f"Step {frame}")
            plt.tight_layout()

        ani = FuncAnimation(
            fig,
            update,
            frames=len(self.radar_snaps),
            repeat=self.loop,
            interval=1000 / self.fps,
        )

        if self.save:
            dir_name = "saved-gifs"
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{dir_name}/{timestamp}.gif"
            writer = PillowWriter(fps=self.fps)
            ani.save(filename, writer=writer)

        if show:
            plt.show()

        return ani
