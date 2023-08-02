import funcs
import tkinter as tk
from tkinter import *
from tkinter import filedialog
import ttkbootstrap as ttk
from PIL import Image, ImageTk


def select_image():
    # File opener button command to import an image.
    global file_path
    file_path = filedialog.askopenfilename()

    image = Image.open(file_path)
    display_img = image.resize((450,450))
    
    # Updating image label of root window to display the image.
    display_img = ImageTk.PhotoImage(display_img)
    dsp_label.config(image= display_img)
    dsp_label.image = display_img
    solve_button.config(state="enabled")



def solve_sudoku():
    # Solving algorithm button command.

    global file_path
    grid = None

    progress_value.set(value=1) # ProgressBar updating
    
    # Grid image preprocessing (more detail in funcs.py file)
    input_grid = funcs.grid_preprocessing(file_path)

    progress_value.set(value=2) # ProgressBar updating

    # Extracting the cells out of the grid.
    cells = funcs.extract_cells(input_grid)

    progress_value.set(value=3) # ProgressBar updating

    # --- #
    
    # fig, ax = plt.subplots(9, 9, figsize=(10, 10))

    # for i in range(9):
    #     for j in range(9):
    #         # Display each cell
    #         ax[i, j].imshow(cells[i][j], cmap='gray')

    #         # Remove the axis
    #         ax[i, j].axis('off')

    # plt.show()

    # --- #


    # Reshaping the cells to prepare for digits predictions.
    cells = cells.reshape((81, 28, 28))
    cell_preds = model.predict(cells)

    progress_value.set(value=4) # ProgressBar updating

    # Saving the predictions in a sudoku grid structure.
    grid = funcs.make_grid(cell_preds)
    print(grid)

    progress_value.set(value=5) # ProgressBar updating

    # Display an error message if can't be solved.
    if not funcs.solve_sudoku(grid):
        # Creating new window to display the solution.
        window = tk.Toplevel(root)
        window.title("Solution")
        label = ttk.Label(master=window, text="No solution exists")
        label.pack()
        return
    
    # Generating the solution as image.
    solution = funcs.generate_solution(grid)
    progress_value.set(value=6) # ProgressBar updating
    solved_img = ImageTk.PhotoImage(solution)
    sol_label.config(image= solved_img)
    dsp_label.image = solved_img


    progress_value.set(value=7) # ProgressBar updating



# Loading digits-recognition model.
model = funcs.load_digit_model()
if model is None:
    print("Error loading model")
    exit()


file_path=""

# Root window.
root = ttk.Window(themename="superhero")
root.title('Sudoku Auto Solver')
root.geometry('1100x800')

# Root window title.
title_label = ttk.Label(master=root, text="Sudoku Auto Solver App", font="Calibri 24")
title_label.pack(pady=10)

# Images frame to display user input and solution
img_frame = ttk.Frame(master=root, width=1000, height=500)
img_frame.pack(pady=20)

display_image = ImageTk.PhotoImage(Image.new('RGB', (450, 450), 'white'))
solved_image = ImageTk.PhotoImage(Image.new('RGB', (450, 450), 'white'))
dsp_label = tk.Label(master=img_frame, image=display_image, borderwidth=2, relief="solid")
sol_label = tk.Label(master=img_frame, image=solved_image, borderwidth=2, relief="solid")
dsp_label.pack(side=LEFT, padx=25, pady=25)
sol_label.pack(side=RIGHT, padx=25, pady=25)

# Solving progress bar
progress_value = tk.IntVar(value=0)
progress_bar = ttk.Progressbar(master=root, variable=progress_value, maximum=7, length=700)
progress_bar.pack()

# Buttons frame to manage the placement
btn_frame = ttk.Frame(master=root, width=1000, height=100)
btn_frame.pack(pady=20)

# Image file-opener button.
import_button = ttk.Button(master=btn_frame, text="Import image", width=15, command=select_image)
import_button.place(relx=0.2, rely=0.2)

# Button to execute solving algorithm.
solve_button = ttk.Button(master=btn_frame, text="Solve", state="disabled", width=15, command=solve_sudoku)
solve_button.place(relx=0.7, rely=0.2)

# About label
about_label = ttk.Label(master=root, text="This app created by Yarden Dali, Software engineering student at SCE.\nThis app might not work at 100% rate and may face some issues.", font="Calibri 24")
about_label.pack()

root.mainloop()
