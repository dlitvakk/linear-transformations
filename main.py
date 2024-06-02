import numpy as np
import matplotlib.pyplot as plt

star = np.array([
    [0, 0],
    [0.75, 0.25],
    [1, 1],
    [1.25, 0.25],
    [2, 0],
    [1.25, -0.25],
    [1, -1],
    [0.75, -0.25],
    [0, 0]
])

tree = np.array([
    [1, 0.75],
    [1.5, 0.25],
    [1.25, 0.25],
    [1.75, -0.25],
    [1.5, -0.25],
    [2, -0.75],
    [0, -0.75],
    [0.5, -0.25],
    [0.25, -0.25],
    [0.75, 0.25],
    [0.5, 0.25],
    [1, 0.75]
])


def plot_object(obj, title):
    plt.figure()
    plt.plot(obj[:, 0], obj[:, 1], marker='o')
    plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid()
    plt.show()


plot_object(star, "Star")
plot_object(tree, "Tree")


def rotate(obj, angle):
    rad = np.deg2rad(angle)
    rotation_matrix = np.array([
        [np.cos(rad), -np.sin(rad)],
        [np.sin(rad), np.cos(rad)]
    ])
    transformed_obj = obj @ rotation_matrix.T
    print("Rotation Matrix:\n", rotation_matrix)
    return transformed_obj


def scale(obj, scope):
    scaling_matrix = np.array([
        [scope, 0],
        [0, scope]
    ])
    transformed_obj = obj @ scaling_matrix.T
    print("Scaling Matrix:\n", scaling_matrix)
    return transformed_obj


def reflect(obj, axis):
    if axis == 'x':
        reflection_matrix = np.array([
            [1, 0],
            [0, -1]
        ])
    elif axis == 'y':
        reflection_matrix = np.array([
            [-1, 0],
            [0, 1]
        ])

    transformed_obj = obj @ reflection_matrix.T
    print("Reflection Matrix:\n", reflection_matrix)
    return transformed_obj


def shear(obj, shear_factor, axis):
    if axis == 'x':
        shear_matrix = np.array([
            [1, shear_factor],
            [0, 1]
        ])
    elif axis == 'y':
        shear_matrix = np.array([
            [1, 0],
            [shear_factor, 1]
        ])
    else:
        raise ValueError("Axis must be 'x' or 'y'")

    transformed_obj = obj @ shear_matrix.T
    print("Shear Matrix:\n", shear_matrix)
    return transformed_obj



rotated_star = rotate(star, 78)
plot_object(rotated_star, "Rotated Star by 78 degrees")

rotated_tree = rotate(tree, 100)
plot_object(rotated_tree, "Rotated Tree by 100 degrees")


scaled_star = scale(star, 5)
plot_object(scaled_star, "Scaled Star by 5 degrees")

scaled_tree = scale(tree, 2)
plot_object(scaled_tree, "Scaled Tree by factor of 2")


reflected_star = reflect(star, 'x')
plot_object(reflected_star, "Reflected Star over x-axis")

reflected_tree = reflect(tree, 'y')
plot_object(reflected_tree, "Reflected Tree over y-axis")


sheared_star = shear(star, 2, 'x')
plot_object(sheared_star, "Sheared Star over x-axis")

sheared_tree = shear(tree, 3, 'y')
plot_object(sheared_tree, "Sheared Tree over y-axis")
