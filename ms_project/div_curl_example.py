from manim import *
import numpy as np
from electromagnetic_animator import ElectrostaticVisualizer

class DivCurlExample(ElectrostaticVisualizer):
    def construct(self):
        # Add a positive and negative charge
        self.add_charge((2, 0, 0), +1, radius=0.3)
        self.add_charge((-2, 0, 0), -1, radius=0.3)

        # Solve the fields
        self.solve_fields()

        # Render a single frame of the field visualization
        self.render_fields(stream_density=1.5, equipotential_steps=15)

# To visualize one frame, we can use Manim's CLI or run this script directly
if __name__ == "__main__":
    scene = DivCurlExample()
    scene.render()