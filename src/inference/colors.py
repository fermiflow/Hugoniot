import numpy as np
from typing import Union, Tuple
from matplotlib.colors import to_rgb

default_colors = ['C0', 'C1','C2', 'C3', 'C4', 'C6', 'C5', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13']

def gen_gradient_colors(start_color: Union[str, Tuple],
                        end_color: Union[str, Tuple],
                        steps: int,
    ):
    """
        Generate a list of colors that gradually change from start_color to end_color.
    Input:
        start_color: string or Tuple of 3 integers, representing RGB values of the start color
        end_color: string or Tuple of 3 integers, representing RGB values of the end color
        steps: number of colors to generate
    """
    if isinstance(start_color, str):
        start_color = to_rgb(start_color)
    if isinstance(end_color, str):
        end_color = to_rgb(end_color)
    color_R = np.linspace(start_color[0], end_color[0], steps)
    color_G = np.linspace(start_color[1], end_color[1], steps)
    color_B = np.linspace(start_color[2], end_color[2], steps)
    colors = [(color_R[i], color_G[i], color_B[i]) for i in range(steps)]
    return colors
