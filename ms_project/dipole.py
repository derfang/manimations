from manim import *
from finiteSolver import ElectrostaticFDSolver as ElectrostaticFDM
import numpy as np

class SolveAndVisualize(Scene):
    def construct(self):
        # Initialize the solver with default grid and domain size
        solver = ElectrostaticFDM()

        # Define charge density (point charge at center)
        rho = np.zeros(solver.grid_size)
        center_x, center_y = solver.grid_size[0] // 2, solver.grid_size[1] // 2
        rho[center_x, center_y] = 1.0  # Point charge

        # Solve the problem
        solver.solve(rho)

        # Get potential and field data
        potential = solver.get_potential_data()
        field_func = lambda p: solver.get_field_vector_interpolated(p[0], p[1])

        # Visualize potential as a heatmap
        potential_title = Text("Electric Potential").to_edge(UP)
        self.add(potential_title)
        self.play(FadeIn(potential_title))

        # Visualize electric field as a vector field
        vector_field = ArrowVectorField(field_func)
        stream_lines = StreamLines(field_func, stroke_width=2, max_anchors_per_line=30)
        stream_lines.shuffle()
        field_title = Text("Electric Field").to_edge(UP)

        self.add(vector_field)
        self.play(
            AnimationGroup(*[Create(line) for line in stream_lines], lag_ratio=0.1),
            Write(field_title)
        )
        self.wait()