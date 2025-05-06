from manim import *

class MaxwellEquations(Scene):
    def construct(self):
        # Title
        title = Text("Maxwell's Equations", font_size=48)
        self.play(Write(title))
        self.wait(2)
        self.play(FadeOut(title))

        # Equation 1: Gauss's Law
        gauss_law = MathTex("\\nabla \\cdot \\mathbf{E} = \\frac{\\rho}{\\varepsilon_0}")
        gauss_law_title = Text("Gauss's Law", font_size=36).next_to(gauss_law, UP)
        self.play(Write(gauss_law_title), Write(gauss_law))
        self.wait(2)
        self.play(FadeOut(gauss_law_title), FadeOut(gauss_law))

        # Numerical Example: Gauss's Law
        gauss_example = Text("Numerical Example: Gauss's Law", font_size=24).to_edge(UP)
        gauss_code = Code(
            code="""import numpy as np
rho = 1e-6  # Charge density (C/m^3)
epsilon_0 = 8.85e-12  # Permittivity of free space (F/m)
L = 1.0  # Length of the domain (m)
N = 100  # Number of grid points
dx = L / (N - 1)  # Grid spacing
x = np.linspace(0, L, N)
E = np.cumsum(rho / epsilon_0 * dx)  # Numerical integration
print(E)""",
            tab_width=4,
            language="Python",
            font_size=16
        )
        self.play(Write(gauss_example), Create(gauss_code))
        self.wait(4)
        self.play(FadeOut(gauss_example), FadeOut(gauss_code))

        # Equation 2: Gauss's Law for Magnetism
        gauss_magnetism = MathTex("\\nabla \\cdot \\mathbf{B} = 0")
        gauss_magnetism_title = Text("Gauss's Law for Magnetism", font_size=36).next_to(gauss_magnetism, UP)
        self.play(Write(gauss_magnetism_title), Write(gauss_magnetism))
        self.wait(2)
        self.play(FadeOut(gauss_magnetism_title), FadeOut(gauss_magnetism))

        # Numerical Example: Gauss's Law for Magnetism
        magnetism_example = Text("Numerical Example: Gauss's Law for Magnetism", font_size=24).to_edge(UP)
        magnetism_code = Code(
            code="""import numpy as np
B = np.ones(100)  # Uniform magnetic field
dx = 0.01  # Grid spacing
div_B = np.gradient(B, dx)  # Numerical divergence
print(div_B)""",
            tab_width=4,
            language="Python",
            font_size=16
        )
        self.play(Write(magnetism_example), Create(magnetism_code))
        self.wait(4)
        self.play(FadeOut(magnetism_example), FadeOut(magnetism_code))

        # Equation 3: Faraday's Law of Induction
        faraday_law = MathTex("\\nabla \\times \\mathbf{E} = -\\frac{\\partial \\mathbf{B}}{\\partial t}")
        faraday_law_title = Text("Faraday's Law of Induction", font_size=36).next_to(faraday_law, UP)
        self.play(Write(faraday_law_title), Write(faraday_law))
        self.wait(2)
        self.play(FadeOut(faraday_law_title), FadeOut(faraday_law))

        # Numerical Example: Faraday's Law
        faraday_example = Text("Numerical Example: Faraday's Law", font_size=24).to_edge(UP)
        faraday_code = Code(
            code="""import numpy as np
t = np.linspace(0, 1, 100)
B_t = np.sin(2 * np.pi * t)
E = -np.gradient(B_t, t)
print(E)""",
            tab_width=4,
            language="Python",
            font_size=16
        )
        self.play(Write(faraday_example), Create(faraday_code))
        self.wait(4)
        self.play(FadeOut(faraday_example), FadeOut(faraday_code))

        # Equation 4: Ampere's Law (with Maxwell's correction)
        ampere_law = MathTex("\\nabla \\times \\mathbf{B} = \\mu_0 \\mathbf{J} + \\mu_0 \\varepsilon_0 \\frac{\\partial \\mathbf{E}}{\\partial t}")
        ampere_law_title = Text("Ampere's Law (with Maxwell's correction)", font_size=36).next_to(ampere_law, UP)
        self.play(Write(ampere_law_title), Write(ampere_law))
        self.wait(2)
        self.play(FadeOut(ampere_law_title), FadeOut(ampere_law))

        # Numerical Example: Ampere's Law
        ampere_example = Text("Numerical Example: Ampere's Law", font_size=24).to_edge(UP)
        ampere_code = Code(
            code="""import numpy as np
mu_0 = 4 * np.pi * 1e-7
epsilon_0 = 8.85e-12
J = np.zeros(100)
t = np.linspace(0, 1, 100)
E_t = np.sin(2 * np.pi * t)
dE_dt = np.gradient(E_t, t)
B = mu_0 * (J + epsilon_0 * dE_dt)
print(B)""",
            tab_width=4,
            language="Python",
            font_size=16
        )
        self.play(Write(ampere_example), Create(ampere_code))
        self.wait(4)
        self.play(FadeOut(ampere_example), FadeOut(ampere_code))

        # End Scene
        end_text = Text("These are Maxwell's Equations", font_size=36)
        self.play(Write(end_text))
        self.wait(3)
        self.play(FadeOut(end_text))