from manim import *

class ElectricField(Scene):
    CONFIG = {
        "production_quality_flow": True,
        "vector_field_func": lambda p: UP,
    }

    def construct(self):
        self.add_plane()
        self.add_title()
        self.setup_warped_grid()
        self.show_uniform_field()
        self.show_moving_charges()
        self.show_field_lines()

    def add_plane(self):
        self.plane = NumberPlane()
        self.plane.add_coordinates()
        self.plane.coordinate_labels.submobjects.pop(-1)
        self.add(self.plane)

    def add_title(self):
        title = Tex("Electric Field")
        title.to_edge(UP, buff=MED_SMALL_BUFF)
        title.add_background_rectangle()
        self.title = title
        self.add(title)

    def setup_warped_grid(self):
        warped_grid = self.warped_grid = self.get_warpable_grid()
        warped_grid.save_state()
        func_label = self.get_func_label()
        # Removed the unit circle from the scene setup
        # self.unit_circle = Circle(
        #     radius=self.plane.x_axis.unit_size,  # Updated to use x_axis.unit_size
        #     stroke_color=YELLOW,
        #     fill_color=BLACK,
        #     fill_opacity=1
        # )

        self.add_foreground_mobjects(self.title, func_label)  # Removed unit_circle from foreground objects
        self.remove(self.plane)
        self.play(
            warped_grid.animate.apply_complex_function(lambda z: z),
        )
        self.wait()

    def show_uniform_field(self):
        vector_field = self.vector_field = VectorField(
            lambda p: UP,
            colors=[BLUE_E, WHITE, RED]
        )
        self.add(vector_field)  # Add the vector field directly to the scene
        self.wait()

    def show_moving_charges(self):
        protons = VGroup(*[Dot(color=RED).move_to(2 * RIGHT * i) for i in range(-3, 4)])
        electrons = VGroup(*[Dot(color=BLUE).move_to(2 * LEFT * i) for i in range(-3, 4)])
        self.play(FadeIn(protons), FadeIn(electrons))
        self.wait()

    def show_field_lines(self):
        stream_lines = StreamLines(
            lambda p: np.array([p[1], -p[0], 0]),
            stroke_width=2,
            max_anchors_per_line=30
        )
        self.add(stream_lines)
        self.play(ShowPassingFlash(stream_lines))
        self.wait()

    def get_func_label(self):
        func_label = MathTex("f(z) = z + 1 / z")
        func_label.add_background_rectangle()
        func_label.next_to(self.title, DOWN, MED_SMALL_BUFF)
        return func_label

    def get_warpable_grid(self):
        return NumberPlane()