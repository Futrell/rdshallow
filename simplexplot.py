import sys
import itertools
import functools

import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

@functools.cache
def simplex_grid(k, num_points, edge_distance=0):
    """ Generate all arrays a of length d such that sum(a) == 1. """
    assert k > 1
    assert num_points > 1
    # To do a fixed distance from edges, do a weighted mean of the initial
    # grid with the center point, where the weight is however much you need
    # to get the right distance.
    middle = 1 / k # center of simplex
    weight = 1 - edge_distance * k
    numbers = np.linspace(0, 1, num_points)
    numbers = weight * numbers + (1 - weight) * middle
    def gen():
        for xs in itertools.product(numbers, repeat=k):
            if np.isclose(sum(xs), 1):
                yield np.array(xs)
    return np.array(list(gen()))

def project_to_simplex(p):
    # Project a 3D point (x, y, z) onto a 2D triangle for simplex visualization
    return np.array([p[0] + p[1] / 2, np.sqrt(3) / 2 * p[1]])

def simplex_plot(points,
                 labels=None,
                 colors=None,
                 point_cmap='magma',
                 with_arrows=False,
                 animate=True,
                 background_values=None,
                 grid_cmap='viridis',
                 grid_density=50,
                 grid_alpha=1,
                 show=True,
                 save=False,
                 interval=250):
    # Check if the points are valid (i.e., they lie on the simplex)
    if not np.allclose(np.sum(points, axis=1), 1):
        raise ValueError("All points must lie on the 3D probability simplex (i.e., sum to 1).")
    
    # Set default labels if not provided
    if labels and len(labels) != 3:
        raise ValueError("The 'labels' argument must contain exactly three strings.")
    
    # Define the vertices of the 2D simplex (rotated)
    # Mapping x -> top, y -> bottom right, z -> bottom left (clockwise)
    vertices = np.eye(3)
    projected_vertices = np.array([project_to_simplex(v) for v in vertices])
    permutation = [2,0,1]

    # Project all the input points
    projected_points = np.array([project_to_simplex(p) for p in points[:, permutation]])

    # Create a plot
    fig, ax = plt.subplots()

    # Plot the simplex (triangle)
    triangle = plt.Polygon(projected_vertices, color='lightblue', alpha=0.5)
    ax.add_patch(triangle)

    if background_values is not None:
        grid = simplex_grid(3, grid_density)
        values = grid @ background_values
        projected_grid = np.array([project_to_simplex(line) for line in grid[:, permutation]])
        triang = tri.Triangulation(projected_grid[:, 0], projected_grid[:, 1])
        color_plot = ax.tripcolor(triang, values, shading='gouraud', cmap=grid_cmap, alpha=grid_alpha)

        plt.colorbar(color_plot, ax=ax)

    # Plot the input points
    if colors is None:
        colors = ['red'] * len(points)
    else:
        cmap = cm.get_cmap(point_cmap)
        colors = cmap(colors / colors.max())

    if not animate:
        if with_arrows:
            for i in range(len(projected_points) - 1):
                ax.arrow(projected_points[i, 0], projected_points[i, 1],
                         projected_points[i+1, 0] - projected_points[i, 0],
                         projected_points[i+1, 1] - projected_points[i, 1],
                         head_width=0.02, length_includes_head=True, color=colors[i])
        else:
            ax.scatter(projected_points[:, 0], projected_points[:, 1], color=colors, s=50)

    # Add custom labels for the rotated vertices
    if labels:
        ax.text(projected_vertices[0, 0], projected_vertices[0, 1], labels[permutation[0]] + "\n [0,0,1]", ha='center', va='bottom')
        ax.text(projected_vertices[1, 0], projected_vertices[1, 1], labels[permutation[1]] + "\n [0,1,0]", ha='right', va='top')
        ax.text(projected_vertices[2, 0], projected_vertices[2, 1], labels[permutation[2]] + "\n [1,0,0]", ha='left', va='top')

    # Set limits and labels
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, np.sqrt(3) / 2 + 0.1)

    # Add a legend
    ax.legend()

    # Set equal aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')

    if animate:
        scatter = ax.scatter([], [], color=[], edgecolors='k', s=100)  # Set an initial color
        text_display = ax.text(0, 0, '', fontsize=10, ha='center', va='bottom')

        def init():
            scatter.set_offsets(np.empty((0, 2)))
            scatter.set_color([])
            text_display.set_text('')
            return scatter, text_display

        # Function to update the plot for each frame
        def update(frame):
            # Get the point and color at the current frame
            new_offsets = [projected_points[frame]]  # Only the current point
            new_color = [colors[frame]]  # Only the current color
            scatter.set_offsets(new_offsets)  # Update the scatter plot data
            scatter.set_facecolors(new_color)  # Update the color of the point
            scatter.set_color(new_color)  # Update the color of the point
            scatter.set_edgecolor('k')

            # Update the text display with the original 3D point [x, y, z], formatted to two decimals
            point_value = points[frame, permutation]
            formatted_text = f"[{point_value[0]:.2f}, {point_value[1]:.2f}, {point_value[2]:.2f}]"
            text_display.set_position((new_offsets[0][0], new_offsets[0][1] + 0.05))  # Position above the point
            text_display.set_text(formatted_text)  # Update the text
            
            return scatter, text_display

        # Create the animation
        ani = FuncAnimation(fig, update, frames=len(points), init_func=init, interval=interval, blit=False)
        if save:
            ani.save(save, writer="pillow", fps=1000/interval)

    if show:
        plt.show()

        
    

def plot(d, item, condition, alternatives, distortion=None, save=False):
    da = d[d['_item'] == int(item)]
    daf = da[da['_condition'] == condition][alternatives]
    Z = daf.sum(axis=1).to_numpy()
    p = daf[alternatives].to_numpy() / Z[:, None]
    d_kl = da[da['_condition'] == condition]['d_kl_div']
    simplex_plot(p, labels=alternatives, colors=d_kl, background_values=-distortion, save=save)

def main(filename, item_str, condition, alternatives_str, distortion_str=None):
    d = pd.read_csv(filename)
    item = int(item_str)
    alternatives = alternatives_str.split()
    distortion = None if distortion_str is None else eval(distortion_str)
    return plot(d, item, condition, alternatives, distortion=distortion)
    
if __name__ == '__main__':
    main(*sys.argv[1:])
    
    
