from .Radar import Radar
from .StateSpaceModel import StateSpaceModel
from .CVM import CVM
from .GaussianMixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from .helpers import plot_cov_ellipse


class RadarNetwork:

    def __init__(
        self,
        radars: list[Radar],
        model: StateSpaceModel = CVM(),
        birth_intensity: GaussianMixture = None,
        survival_probability: float = 1.0,
        global_clutter_intensity_per_unit: float = 0,
    ):
        """
        Initialize the radar network.

        :param radars: The radars in the network.
        :param model: The state space model used by the radars.
        :param birth_intensity: The birth intensity of the targets.
        :param survival_probability: The probability of a target's survival.
        :param global_clutter_intensity_per_unit: The global clutter intensity per unit
        area.
        """
        self.radars = radars
        self.model = model
        self.birth_intensity = birth_intensity
        self.survival_probability = survival_probability
        self.global_clutter_intensity_per_unit = global_clutter_intensity_per_unit
        self.rectangle: np.ndarray = self.calculate_rectangle()
        self.targets: list[np.ndarray] = []
        self.target_trajectories: list[list[np.ndarray]] = []
        self.global_clutter: list[np.ndarray] = []

    def copy(self):
        """
        Create a deep copy of the radar network.
        """
        new_net = RadarNetwork(
            radars=[radar.copy() for radar in self.radars],
            model=self.model,
            birth_intensity=self.birth_intensity,
            survival_probability=self.survival_probability,
            global_clutter_intensity_per_unit=self.global_clutter_intensity_per_unit,
        )

        new_net.targets = [target.copy() for target in self.targets]
        new_net.target_trajectories = [
            [point.copy() for point in trajectory]
            for trajectory in self.target_trajectories
        ]
        new_net.global_clutter = [point.copy() for point in self.global_clutter]

        return new_net

    def calculate_rectangle(self) -> np.ndarray:
        """
        Calculate the rectangle that bounds all radars.
        """
        min_x = min(radar.center[0] - radar.radius for radar in self.radars)
        max_x = max(radar.center[0] + radar.radius for radar in self.radars)
        min_y = min(radar.center[1] - radar.radius for radar in self.radars)
        max_y = max(radar.center[1] + radar.radius for radar in self.radars)

        return np.array([[min_x, max_y], [max_x, min_y]])

    def scan(self) -> dict[str, list[np.ndarray]]:
        """
        Perform the radar scans.
        """
        self.update_environment()
        self.individual_scans()
        return self.get_measurements()

    def update_environment(self):
        """
        Update the environment.
        """
        self.generate_global_clutter()
        self.evolve_targets()
        self.birth_targets()
        self.remove_out_of_bounds()

    def generate_global_clutter(self):
        """
        Generate global clutter.
        """
        self.global_clutter = []

        # sample the number of clutter points from a Poisson distribution
        area = (self.rectangle[1, 0] - self.rectangle[0, 0]) * (
            self.rectangle[0, 1] - self.rectangle[1, 1]
        )
        clutter_intensity = area * self.global_clutter_intensity_per_unit
        num_clutter = np.random.poisson(clutter_intensity)

        # sample the clutter points from a uniform distribution
        for _ in range(num_clutter):
            x = np.random.uniform(self.rectangle[0, 0], self.rectangle[1, 0])
            y = np.random.uniform(self.rectangle[1, 1], self.rectangle[0, 1])
            self.global_clutter.append(np.array([x, y]))

    def evolve_targets(self):
        """
        Evolve targets.
        """
        evolved_targets = []
        evolved_trajectories = []
        for target, trajectories in zip(self.targets, self.target_trajectories):
            target = self.model.evolve(target)
            trajectories.append(target)
            if np.random.rand() < self.survival_probability:
                evolved_targets.append(target)
                evolved_trajectories.append(trajectories.copy())
        self.targets = evolved_targets
        self.target_trajectories = evolved_trajectories

    def birth_targets(self):
        """
        Generate new targets.
        """
        if self.birth_intensity is None:
            return

        new_targets = self.birth_intensity.sample()
        self.targets += new_targets
        self.target_trajectories.extend([target] for target in new_targets)

    def remove_out_of_bounds(self):
        """
        Remove targets that are out of bounds.
        """
        for i, target in enumerate(self.targets):
            if (
                target[0] < self.rectangle[0, 0]
                or target[0] > self.rectangle[1, 0]
                or target[1] < self.rectangle[1, 1]
                or target[1] > self.rectangle[0, 1]
            ):
                self.targets.pop(i)
                self.target_trajectories.pop(i)

    def individual_scans(self):
        """
        Perform individual scans.
        """
        for radar in self.radars:
            radar.scan(self.targets, self.global_clutter)

    def get_measurements(self) -> dict[str, list[np.ndarray]]:
        """
        Get the measurements from all radars.
        """
        return {radar.id: radar.measurements for radar in self.radars}

    def plot_radars(self):
        """
        Plot the radars.
        """
        plt.figure()
        for radar in self.radars:
            circle = plt.Circle(radar.center, radar.radius, color="k", fill=False)
            plt.gca().add_artist(circle)
            plt.text(radar.center[0], radar.center[1], radar.id)
        plt.xlim(self.rectangle[0, 0], self.rectangle[1, 0])
        plt.ylim(self.rectangle[1, 1], self.rectangle[0, 1])
        plt.gca().set_aspect("equal", adjustable="box")
        plt.show()

    def plot_state(self):
        """
        Plot the state of the radar net.
        """
        plt.figure(figsize=(15, 15))

        # plot global clutter
        clutter = self.global_clutter
        clutter_x = [point[0] for point in clutter]
        clutter_y = [point[1] for point in clutter]
        plt.scatter(clutter_x, clutter_y, color="grey", s=10, marker="x")

        # plot airports
        for airport in self.birth_intensity:
            mean = airport.means[0]
            cov = airport.covariances[0]
            plt.scatter(mean[0], mean[1], color="b", s=10)
            ellipse = plot_cov_ellipse(mean[:2], cov[:2, :2], n_std=3, color="b")
            plt.gca().add_artist(ellipse)

        # plot trajectories
        for trajectory in self.target_trajectories:
            trajectory_x = [point[0] for point in trajectory]
            trajectory_y = [point[1] for point in trajectory]
            plt.plot(trajectory_x, trajectory_y, color="g", alpha=0.5)

        # plot radars ranges
        for radar in self.radars:
            circle = plt.Circle(radar.center, radar.radius, color="k", fill=False)
            plt.gca().add_artist(circle)
            plt.text(radar.center[0], radar.center[1], radar.id)

        # plot radars clutter
        for radar in self.radars:
            clutter = radar.local_clutter
            clutter_x = [point[0] for point in clutter]
            clutter_y = [point[1] for point in clutter]
            plt.scatter(clutter_x, clutter_y, color="lightgrey", s=10, marker="x")

        # plot targets
        targets = self.targets
        targets_x = [target[0] for target in targets]
        targets_y = [target[1] for target in targets]
        plt.scatter(targets_x, targets_y, color="g", s=10)

        # plot radars target measurements
        for radar in self.radars:
            measurements = radar.detected_targets
            measurements_x = [point[0] for point in measurements]
            measurements_y = [point[1] for point in measurements]
            plt.scatter(measurements_x, measurements_y, color="r", s=10)

        plt.xlim(self.rectangle[0, 0], self.rectangle[1, 0])
        plt.ylim(self.rectangle[1, 1], self.rectangle[0, 1])
        plt.gca().set_aspect("equal", adjustable="box")
        plt.show()
