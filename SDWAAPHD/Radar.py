import numpy as np
from .StateSpaceModel import StateSpaceModel
from .CVM import CVM


class Radar:
    def __init__(
        self,
        id: str,
        center: np.ndarray,
        radius: float,
        model: StateSpaceModel = CVM(),
        clutter_intensity_per_unit: float = 0,
        detection_probability: float = 1.0,
    ):
        """
        Initialize the radar.

        :param id: The unique identifier of the radar.
        :param center: The center of the radar.
        :param radius: The radius of the radar.
        :param model: The state space model used by the radar.
        :param clutter_intensity_per_unit: The clutter intensity per unit area.
        :param detection_probability: The probability of a target's detection.
        """
        self.id = id
        self.center = center
        self.radius = radius
        self.model = model
        self.clutter_intensity_per_unit = clutter_intensity_per_unit
        self.detection_probability = detection_probability
        self.detected_targets: list[np.ndarray] = []
        self.global_clutter: list[np.ndarray] = []
        self.local_clutter: list[np.ndarray] = []
        self.clutter: list[np.ndarray] = []
        self.measurements: list[np.ndarray] = []

    def copy(self):
        """
        Create a deep copy of the radar.
        """
        new_radar = Radar(
            id=self.id,
            center=self.center.copy(),
            radius=self.radius,
            model=self.model,
            clutter_intensity_per_unit=self.clutter_intensity_per_unit,
            detection_probability=self.detection_probability,
        )

        new_radar.detected_targets = [target.copy() for target in self.detected_targets]
        new_radar.global_clutter = [point.copy() for point in self.global_clutter]
        new_radar.local_clutter = [point.copy() for point in self.local_clutter]
        new_radar.clutter = [point.copy() for point in self.clutter]
        new_radar.measurements = [point.copy() for point in self.measurements]

        return new_radar

    def __repr__(self):
        """
        Return a string representation of the Radar object.
        """
        return f"Radar {self.id} at {self.center} with radius {self.radius}"

    def scan(self, targets, global_clutter):
        """
        Perform one radar scan.
        """
        # filter clutter points that are within the radar's range
        self.global_clutter = [
            point for point in global_clutter if self.is_in_range(point)
        ]

        # detect targets
        self.detect_targets(targets)

        # generate clutter
        self.generate_clutter()
        self.clutter = self.local_clutter + self.global_clutter

        # combine the measurements
        self.measurements = self.detected_targets + self.clutter

    def is_in_range(self, point):
        """
        Check if a point is within the radar's range.
        """
        return np.linalg.norm(point - self.center) <= self.radius

    def generate_clutter(self):
        """
        Generate clutter.
        """
        self.local_clutter = []

        # sample the number of clutter points from a Poisson distribution
        area = np.pi * self.radius**2
        clutter_intensity = area * self.clutter_intensity_per_unit
        num_clutter = np.random.poisson(clutter_intensity)

        # sample the clutter points from a uniform distribution
        for _ in range(num_clutter):
            angle = np.random.uniform(0, 2 * np.pi)
            radius = self.radius * np.sqrt(np.random.uniform(0, 1))
            x = self.center[0] + radius * np.cos(angle)
            y = self.center[1] + radius * np.sin(angle)
            self.local_clutter.append(np.array([x, y]))

    def detect_targets(self, targets):
        """
        Detect and measure targets.
        """
        self.detected_targets = []
        for target in targets:
            if not self.is_in_range(target[:2]):
                continue
            if np.random.rand() < self.detection_probability:
                measurement = self.model.measure(target)
                if not self.is_in_range(measurement):
                    continue
                self.detected_targets.append(measurement)
