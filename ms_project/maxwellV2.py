from manim import *
import numpy as np
from scipy.fftpack import fft

class MaxwellEquations3D(ThreeDScene):  # Changed from Scene to ThreeDScene
    def construct(self):
        # Title Slide
        title = Text("Maxwell's Equations", font_size=48)
        self.add_fixed_in_frame_mobjects(title)  # Ensures the text remains static
        self.play(Write(title))
        self.wait(2)
        self.play(FadeOut(title))

        # Gauss's Law for Electricity
        gauss_law = MathTex("\\nabla \\cdot \\mathbf{E} = \\frac{\\rho}{\\varepsilon_0}")
        gauss_law_title = Text("Gauss's Law for Electricity", font_size=36).next_to(gauss_law, UP)
        self.add_fixed_in_frame_mobjects(gauss_law_title, gauss_law)  # Ensures the text remains static
        self.play(Write(gauss_law_title), Write(gauss_law))
        self.wait(2)
        self.play(FadeOut(gauss_law_title), FadeOut(gauss_law))

        # 3D Vector Field for Gauss's Law
        axes = ThreeDAxes()
        self.set_camera_orientation(phi=0 * DEGREES, theta=0 * DEGREES)
        self.add(axes)

        def electric_field(pos):
            r = np.linalg.norm(pos)
            return pos / (r**3 + 1e-6)  # Avoid division by zero

        field = ArrowVectorField(electric_field)
        self.play(Create(field))
        self.wait(2)
        self.play(FadeOut(field))

        # Gauss's Law for Magnetism
        gauss_magnetism = MathTex("\\nabla \\cdot \\mathbf{B} = 0")
        gauss_magnetism_title = Text("Gauss's Law for Magnetism", font_size=36).next_to(gauss_magnetism, UP)
        self.add_fixed_in_frame_mobjects(gauss_magnetism_title, gauss_magnetism)  # Ensures the text remains static
        self.play(Write(gauss_magnetism_title), Write(gauss_magnetism))
        self.wait(2)
        self.play(FadeOut(gauss_magnetism_title), FadeOut(gauss_magnetism))

        # 3D Vector Field for Gauss's Law for Magnetism
        def magnetic_field(pos):
            x, y, z = pos
            return np.array([-y, x, 0])  # Circular field

        field = ArrowVectorField(magnetic_field)
        self.play(Create(field))
        self.wait(2)
        self.play(FadeOut(field))

        # Faraday's Law
        faraday_law = MathTex("\\nabla \\times \\mathbf{E} = -\\frac{\\partial \\mathbf{B}}{\\partial t}")
        faraday_law_title = Text("Faraday's Law", font_size=36).next_to(faraday_law, UP)
        self.add_fixed_in_frame_mobjects(faraday_law_title, faraday_law)  # Ensures the text remains static
        self.play(Write(faraday_law_title), Write(faraday_law))
        self.wait(2)
        self.play(FadeOut(faraday_law_title), FadeOut(faraday_law))

        # Ampere's Law with Maxwell's Correction
        ampere_law = MathTex("\\nabla \\times \\mathbf{B} = \\mu_0 \\mathbf{J} + \\mu_0 \\varepsilon_0 \\frac{\\partial \\mathbf{E}}{\\partial t}")
        ampere_law_title = Text("Ampere's Law", font_size=36).next_to(ampere_law, UP)
        self.add_fixed_in_frame_mobjects(ampere_law_title, ampere_law)  # Ensures the text remains static
        self.play(Write(ampere_law_title), Write(ampere_law))
        self.wait(2)
        self.play(FadeOut(ampere_law_title), FadeOut(ampere_law))

        # Conclusion Slide
        conclusion = Text("These are Maxwell's Equations", font_size=36)
        self.play(Write(conclusion))
        self.wait(3)
        self.play(FadeOut(conclusion))