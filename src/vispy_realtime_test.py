"""
Multiple real-time digital signals with GLSL-based clipping.
"""

from vispy import gloo
from vispy import app
import numpy as np


# Number of cols and rows in the table.
# nrows = 1
# ncols = 1

# Number of signals.
# m = nrows*ncols

# Number of samples per signal.
# n = 1000

# Various signal amplitudes.
# amplitudes = .1 + .2 * np.random.rand(m, 1).astype(np.float32)
# amplitudes = 0

# Generate the signals as a (m, n) array.
# y = amplitudes * np.random.randn(m, n).astype(np.float32)
# print(y.shape)
# Dummy data
# y = amplitudes + np.random.randint(2, size=n).reshape(m, n)
# print(y.shape)

# y = amplitudes * np.random.randn(m, n).astype(np.float32)

# # Color of each vertex (TODO: make it more efficient by using a GLSL-based
# # color map and the index).
# color = np.repeat(np.random.uniform(size=(m, 3), low=.5, high=.9),
#                   n, axis=0).astype(np.float32)
#
# # Signal 2D index of each vertex (row and col) and x-index (sample index
# # within each signal).
# index = np.c_[np.repeat(np.repeat(np.arange(ncols), nrows), n),
#               np.repeat(np.tile(np.arange(nrows), ncols), n),
#               np.tile(np.arange(n), m)].astype(np.float32)

# print(amplitudes.shape)
# print(y.shape)
# print(color.shape)
# print(index.shape)

VERT_SHADER = """
#version 120
// y coordinate of the position.
attribute float a_position;
// row, col, and time index.
attribute vec3 a_index;
varying vec3 v_index;
// 2D scaling factor (zooming).
uniform vec2 u_scale;
// Size of the table.
uniform vec2 u_size;
// Number of samples per signal.
uniform float u_n;
// Color.
attribute vec3 a_color;
varying vec4 v_color;
// Varying variables used for clipping in the fragment shader.
varying vec2 v_position;
varying vec4 v_ab;
void main() {
    float nrows = u_size.x;
    float ncols = u_size.y;
    // Compute the x coordinate from the time index.
    float x = -1 + 2*a_index.z / (u_n-1);
    vec2 position = vec2(x - (1 - 1 / u_scale.x), a_position);
    // Find the affine transformation for the subplots.
    vec2 a = vec2(1./ncols, 1./nrows)*.9;
    vec2 b = vec2(-1 + 2*(a_index.x+.5) / ncols,
                  -1 + 2*(a_index.y+.5) / nrows);
    // Apply the static subplot transformation + scaling.
    gl_Position = vec4(a*u_scale*position+b, 0.0, 1.0);
    v_color = vec4(a_color, 1.);
    v_index = a_index;
    // For clipping test in the fragment shader.
    v_position = gl_Position.xy;
    v_ab = vec4(a, b);
}
"""

FRAG_SHADER = """
#version 120
varying vec4 v_color;
varying vec3 v_index;
varying vec2 v_position;
varying vec4 v_ab;
void main() {
    gl_FragColor = v_color;
    // Discard the fragments between the signals (emulate glMultiDrawArrays).
    if ((fract(v_index.x) > 0.) || (fract(v_index.y) > 0.))
        discard;
    // Clipping test.
    vec2 test = abs((v_position.xy-v_ab.zw)/v_ab.xy);
    if ((test.x > 1) || (test.y > 1))
        discard;
}
"""


class Canvas(app.Canvas):
    def __init__(self, nrows, ncols, n, initial_amplitude):
        app.Canvas.__init__(self, title='Use your wheel to zoom!',
                            keys='interactive')

        m = nrows*ncols
        self.m = m
        y = initial_amplitude + np.random.randint(2, size=n).reshape(m, n)
        self.y = y
        self.amplitude = initial_amplitude

        # Color of each vertex (TODO: make it more efficient by using a GLSL-based
        # color map and the index).
        color = np.repeat(np.random.uniform(size=(m, 3), low=.5, high=.9),
                          n, axis=0).astype(np.float32)

        # Signal 2D index of each vertex (row and col) and x-index (sample index
        # within each signal).
        index = np.c_[np.repeat(np.repeat(np.arange(ncols), nrows), n),
                      np.repeat(np.tile(np.arange(nrows), ncols), n),
                      np.tile(np.arange(n), m)].astype(np.float32)

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.program['a_position'] = y.reshape(-1, 1)
        self.program['a_color'] = color
        self.program['a_index'] = index
        self.program['u_scale'] = (1., 1.)
        self.program['u_size'] = (nrows, ncols)
        self.program['u_n'] = n

        gloo.set_viewport(0, 0, *self.physical_size)

        self._timer = app.Timer('auto', connect=self.on_timer, start=True)

        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

        self.show()

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.physical_size)

    def set_data(self, x):
        self.y += 1
        self.program['a_position'].set_data(self.y.ravel().astype(np.float32))
        self.update()
        print("Yawn data updated in canvas")

    def on_timer(self, event):
        """Add some data at the end of each signal (real-time signals)."""
        k = 10

        self.y[:, :-k] = self.y[:, k:]
        self.y[:, -k:] = self.amplitude * np.random.randn(self.m, k)

        self.program['a_position'].set_data(self.y.ravel().astype(np.float32))
        self.update()

    def on_draw(self, event):
        gloo.clear()
        self.program.draw('line_strip')


# def main():
#     c = Canvas()
#     app.run()
#
#
# if __name__ == '__main__':
#     main()

if __name__ == '__main__':
    c = Canvas(1, 1, 1000, 0)
    app.run()