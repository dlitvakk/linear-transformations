import numpy as np
import matplotlib.pyplot as plt
import cv2
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

cube_nodes = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1]
])

cube_edges = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7]
])

def plot_3d_object(nodes, edges, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter3D(nodes[:, 0], nodes[:, 1], nodes[:, 2])

    for edge in edges:
        points = nodes[edge]
        ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 'black')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_title(title)
    plt.show()


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
plot_3d_object(cube_nodes, cube_edges, "Cube in 3D")


def rotate(object, angle):
    rad = np.deg2rad(angle)
    rotation_matrix = np.array([
        [np.cos(rad), -np.sin(rad)],
        [np.sin(rad), np.cos(rad)]
    ])
    transformed = np.dot(object, rotation_matrix)
    print("Rotation Matrix:\n", rotation_matrix)
    return transformed


def scale(object, scope):
    scaling_matrix = np.array([
        [scope, 0],
        [0, scope]
    ])
    transformed = np.dot(object, scaling_matrix)
    print("Scaling Matrix:\n", scaling_matrix)
    return transformed


def reflect(object, axis):
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

    transformed = np.dot(object, reflection_matrix)
    print("Reflection Matrix:\n", reflection_matrix)
    return transformed


def shear(object, shear, axis):
    if axis == 'x':
        shear_matrix = np.array([
            [1, shear],
            [0, 1]
        ])
    elif axis == 'y':
        shear_matrix = np.array([
            [1, 0],
            [shear, 1]
        ])

    transformed = object.dot(shear_matrix.T)
    print("Shear Matrix:\n", shear_matrix)
    return transformed

def custom(object, matrix):
    transformed = np.dot(object, matrix)
    print("Custom Matrix:\n", matrix)
    return transformed

rotated_star = rotate(star, 50)
plot_object(rotated_star, "Rotated Star by 50 degrees")

rotated_tree = rotate(tree, 150)
plot_object(rotated_tree, "Rotated Tree by 150 degrees")


scaled_star = scale(star, 5)
plot_object(scaled_star, "Scaled Star by 5 degrees")

scaled_tree = scale(tree, 2)
plot_object(scaled_tree, "Scaled Tree by factor of 2")


reflected_star = reflect(star, 'y')
plot_object(reflected_star, "Reflected Star over y-axis")

reflected_tree = reflect(tree, 'x')
plot_object(reflected_tree, "Reflected Tree over x-axis")


sheared_star = shear(star, 2, 'x')
plot_object(sheared_star, "Sheared Star over x-axis")

sheared_tree = shear(tree, 3, 'y')
plot_object(sheared_tree, "Sheared Tree over y-axis")

custom_star = custom(star, [[2, 3], [6, 0]])
plot_object(custom_star, "Custom Star")
custom_tree = custom(tree, [[-5, 1], [3, -2]])
plot_object(custom_tree, "Custom Tree")
###

def rotate_3d(object, angle, axis):
    rad = np.deg2rad(angle)
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(rad), -np.sin(rad)],
            [0, np.sin(rad), np.cos(rad)]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(rad), 0, np.sin(rad)],
            [0, 1, 0],
            [-np.sin(rad), 0, np.cos(rad)]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [np.cos(rad), -np.sin(rad), 0],
            [np.sin(rad), np.cos(rad), 0],
            [0, 0, 1]
        ])
    transformed = np.dot(object, rotation_matrix.T)
    print("Rotation Matrix:\n", rotation_matrix)
    return transformed

def shear_3d(object, shear, axis):
    if axis == 'x':
        shear_matrix = np.array([
            [1, shear, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
    elif axis == 'y':
        shear_matrix = np.array([
            [1, 0, 0],
            [shear, 1, 0],
            [0, 0, 1]
        ])
    elif axis == 'z':
        shear_matrix = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, shear, 1]
        ])

    transformed = object.dot(shear_matrix.T)
    print("Shear Matrix:\n", shear_matrix)
    return transformed

rotated_cube = rotate_3d(cube_nodes, 50, 'x')
plot_3d_object(rotated_cube, cube_edges , "Cube Rotated by 45 Degrees (x-axis)")

sheared_cube = shear_3d(cube_nodes, 15, 'y')
plot_3d_object(sheared_cube, cube_edges , "Cube Sheared by 15 (y-axis)")
####
def rotate_cv(object, angle):
    M = cv2.getRotationMatrix2D((0, 0), angle, 1)

    ones = np.ones((object.shape[0], 1))
    points = np.hstack([object, ones])
    rotated_points = M.dot(points.T).T
    print("Rotation Matrix:\n", M)
    return rotated_points


def scale_cv(object, scope):
    M = np.array([
        [scope, 0, 0],
        [0, scope, 0]
    ])
    ones = np.ones(shape=(len(object), 1))
    points = np.hstack([object, ones])

    scaled_points = M.dot(points.T).T
    scaled_points = scaled_points.T
    print("Scaling Matrix:\n", M)
    return scaled_points[:, :2]

def reflect_cv(object, axis):
    if axis == 'x':
        reflection_matrix = np.array([
            [1, 0, 0],
            [0, -1, 0]
        ], dtype=np.float32)
    elif axis == 'y':
        reflection_matrix = np.array([
            [-1, 0, 0],
            [0, 1, 0]
        ], dtype=np.float32)

    reflected_points = cv2.transform(np.array([object]), reflection_matrix)[0]
    print("Reflection Matrix:\n", reflection_matrix)
    return reflected_points

def shear_cv(object, shear, axis):
    if axis == 'x':
        shear_matrix = np.array([
            [1, shear, 0],
            [0, 1, 0]
        ], dtype=np.float32)
    elif axis == 'y':
        shear_matrix = np.array([
            [1, 0, 0],
            [shear, 1, 0]
        ], dtype=np.float32)

    sheared_points = cv2.transform(np.array([object]), shear_matrix)[0]
    print("Shear Matrix:\n", shear_matrix)
    return sheared_points

rotated_star_cv = rotate_cv(star, 50)
rotated_tree_cv = rotate_cv(tree, 150)
plot_object(rotated_star_cv, "Rotated Star by 50 degrees (OpenCV)")
plot_object(rotated_tree_cv, "Rotated Tree by 150 degrees (OpenCV)")

scaled_star_cv = scale_cv(star, 5)
plot_object(scaled_star_cv, "Scaled Star by Factor of 5 (OpenCV)")
scaled_tree_cv = scale_cv(tree, 2)
plot_object(scaled_tree_cv, "Scaled Tree by Factor of 2 (OpenCV)")

reflected_star_cv = reflect_cv(star, 'y')
plot_object(reflected_star_cv, "Reflected Star over y-axis (OpenCV)")
reflected_tree_cv = reflect_cv(tree, 'x')
plot_object(reflected_tree_cv, "Reflected Tree over x-axis (OpenCV)")

sheared_star_cv = shear_cv(star, 2, 'x')
plot_object(sheared_star_cv, "Sheared Star over x-axis (OpenCV)")
sheared_tree_cv = shear_cv(tree, 3, 'y')
plot_object(sheared_tree_cv, "Sheared Tree over y-axis (OpenCV)")

###
path = "/Users/dlitvakk21/Downloads/Cat_August_2010-4.jpg"
image = cv2.imread(path)

def rotate_image(image, angle):
    rotation_matrix = cv2.getRotationMatrix2D((0,0), angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return rotated_image

def shear_image(image, shear_x, shear_y):
    shear_matrix = np.array([
        [1, shear_x, 0],
        [shear_y, 1, 0]
    ], dtype=np.float32)
    sheared_image = cv2.warpAffine(image, shear_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    return sheared_image

rotated_image = rotate_image(image, 45)
plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
sheared_image = shear_image(rotated_image, 15, 20)
plt.imshow(cv2.cvtColor(sheared_image, cv2.COLOR_BGR2RGB))
