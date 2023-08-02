import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.models import load_model

def load_digit_model():
    saved_model = load_model("model/digits_model.h5")
    if saved_model:
        return saved_model

    else:
        return None


def grid_preprocessing(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    image = cv2.bitwise_not(image)

    # Blur the image
    blurred = cv2.GaussianBlur(image, (7, 7), 1)

    # Apply edge detection
    edged = cv2.Canny(blurred, 50, 100)

    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by area in descending order and keep the largest one
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

    # Get the sudoku puzzle contour
    puzzle_contour = contours[0]

    # Apply a perspective transform to get a top-down view of the puzzle
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in puzzle_contour]), key=lambda x: x[1])
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in puzzle_contour]), key=lambda x: x[1])
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in puzzle_contour]), key=lambda x: x[1])
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in puzzle_contour]), key=lambda x: x[1])

    corners = puzzle_contour[top_left][0], puzzle_contour[top_right][0], puzzle_contour[bottom_right][0], puzzle_contour[bottom_left][0]
    top_left, top_right, bottom_right, bottom_left = corners

    width_A = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
    width_B = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
    height_A = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
    height_B = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))

    max_width = max(int(width_A), int(width_B))
    max_height = max(int(height_A), int(height_B))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype='float32')

    # Changing the perspective of the image to be top-view
    perspective_transformed = cv2.getPerspectiveTransform(np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32'), dst)
    warped = cv2.warpPerspective(image, perspective_transformed, (max_width, max_height))

    warped = cv2.resize(warped, (316,316))

    return warped




def binarize_image(image, threshold=0.4):
    # Rounding the value of each pixle to be white or black
    binary_image = np.where(image < threshold, 0, 1)

    return binary_image




def extract_cells(image):
    # Initialize 2D list
    cells = [[] for _ in range(9)]
    grid_size = 9
    cell_size = image.shape[0] // 9

    for i in range(grid_size):
        for j in range(grid_size):
            # Extract each cell
            cell = image[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            cell = cell.astype('float32')
            # cell = cv2.resize(cell, (28,28))

            # Cutting off the grid lines out of the cells
            cell = cv2.resize(cell, (34,34))  # Resize to 34x34 to allow cropping the line seperating the cells
            cell = cell[3:31, 3:31]  # Crop 3 pixels off the image from each side

            cell /= 255
            # cell = binarize_image(cell)
            cells[i].append(cell)
    return np.array(cells)


# ------------------------------ #


def is_valid(board, row, col, num):
    # Check duplicate in the row
    for x in range(9):
        if board[row][x] == num:
            return False

    # Check duplicate in the column
    for x in range(9):
        if board[x][col] == num:
            return False

    # Check duplicate in the box (3x3)
    start_row = row - row % 3
    start_col = col - col % 3
    for i in range(3):
        for j in range(3):
            if board[i + start_row][j + start_col] == num:
                return False
    return True



def solve_sudoku(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                for num in range(1, 10):
                    if is_valid(board, i, j, num):
                        board[i][j] = num
                        if solve_sudoku(board):
                            return True
                        board[i][j] = 0
                return False
    return True


# ------------------------------ #


def generate_solution(matrix):

    # Creating plain white board
    solution = Image.new('RGB', (450, 450), 'white')
    draw = ImageDraw.Draw(solution)


    for i in range(9):
    # Draw the Sudoku grid
        draw.line([(i * 50, 0), (i * 50, 500)], fill='black', width=2)
        draw.line([(0, i * 50), (500, i * 50)], fill='black', width=2)


    for i in range(0, 500, 150):
        # Draw the Sudoku border
        draw.line([(i, 0), (i, 500)], fill='black', width=5)
        draw.line([(0, i), (500, i)], fill='black', width=5)


    font = ImageFont.truetype("arial.ttf", 35)

    # Fill in the digits
    for i in range(9):      
        for j in range(9):
            digit = matrix[i][j]
            x = j * 50 + 17
            y = i * 50 + 7
            draw.text((x, y), str(digit), fill='black', font=font)


    return solution

def make_grid(cells):
    grid = []
    row = []
    for i in range(81):
        if i % 9 == 0 and i != 0:
            grid.append(row)
            row = []

        idx = i
        index = np.argmax(cells[idx])
        row.append(index)

        if i == 80:
            grid.append(row)
    
    return grid