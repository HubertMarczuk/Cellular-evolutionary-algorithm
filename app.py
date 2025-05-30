import tkinter as tk
from tkinter import messagebox
import numpy as np
import random
import time
from threading import Thread

points = []  # List of city coordinates selected by the user

# Handles canvas clicks to add a city point
def on_canvas_click(event):
    x, y = event.x, event.y
    canvas.create_oval(x-3, y-3, x+3, y+3, fill='red')
    canvas_anim.create_oval(x-3, y-3, x+3, y+3, fill='red')
    points.append((x, y))

# Computes total length of the route (closed loop)
def distance(route, cities):
    d = 0
    for i in range(len(route)):
        d += np.linalg.norm(np.array(cities[route[i]]) - np.array(cities[route[(i+1)%len(route)]]))
    return d

# Returns negative distance so higher is better (for selection)
def fitness(route, cities):
    return -distance(route, cities)

# Generates a random permutation of city indices
def create_individual(n):
    route = list(range(n))
    random.shuffle(route)
    return route

# Order crossover (OX) operator between two parent routes
def crossover(p1, p2):
    n = len(p1)
    start, end = sorted(random.sample(range(n), 2))
    child = [None]*n
    child[start:end] = p1[start:end]
    p2_iter = [x for x in p2 if x not in child]
    idx = 0
    for i in range(n):
        if child[i] is None:
            child[i] = p2_iter[idx]
            idx += 1
    return child

# Swaps two cities in the route (simple mutation)
def mutate(route):
    i, j = random.sample(range(len(route)), 2)
    route[i], route[j] = route[j], route[i]
    return route

# Selects the best neighbor in a toroidal grid with variable neighborhood
def select_best_neighbor(pop, x, y, grid_size, cities, neigh_type, neigh_radius):
    neighbors = []
    for dx in range(-neigh_radius, neigh_radius+1):
        for dy in range(-neigh_radius, neigh_radius+1):
            if dx == 0 and dy == 0:
                continue
            if neigh_type == 'liniowe' and abs(dx) + abs(dy) > neigh_radius:
                continue
            if neigh_type == 'kompaktowe' and max(abs(dx), abs(dy)) > neigh_radius:
                continue
            nx, ny = (x + dx) % grid_size, (y + dy) % grid_size
            neighbors.append(pop[nx][ny])
    return max(neighbors, key=lambda r: fitness(r, cities))

# Runs the cellular evolutionary algorithm with animation support
def run_cEA_TSP_animated(cities, grid_size, generations, update_callback, delay_func, mutation_prob_func, neigh_type_func, neigh_radius_func):
    n = len(cities)
    pop = [[create_individual(n) for _ in range(grid_size)] for _ in range(grid_size)]

    for gen in range(generations):
        new_pop = [[None]*grid_size for _ in range(grid_size)]
        for x in range(grid_size):
            for y in range(grid_size):
                parent1 = pop[x][y]
                parent2 = select_best_neighbor(pop, x, y, grid_size, cities, neigh_type_func(), neigh_radius_func())
                child = crossover(parent1, parent2)
                if random.random() < mutation_prob_func():
                    child = mutate(child)
                if fitness(child, cities) > fitness(parent1, cities):
                    new_pop[x][y] = child
                else:
                    new_pop[x][y] = parent1
        pop = new_pop
        best = max([pop[x][y] for x in range(grid_size) for y in range(grid_size)], key=lambda r: fitness(r, cities))
        update_callback(best, gen)

        delay = delay_func()
        if delay > 0:
            time.sleep(delay)

    return best

# Draws a closed route path between cities on a given canvas
def draw_path(route, cities, canvas_ref, tag="path"):
    canvas_ref.delete(tag)
    for i in range(len(route)):
        x1, y1 = cities[route[i]]
        x2, y2 = cities[route[(i+1)%len(route)]]
        canvas_ref.create_line(x1, y1, x2, y2, fill='blue', width=2, tags=tag)

# Draws the red city points
def draw_points(canvas_ref):
    for x, y in points:
        canvas_ref.create_oval(x-3, y-3, x+3, y+3, fill='red')

# Starts the algorithm in a separate thread
def start_algorithm():
    if len(points) < 3:
        messagebox.showwarning("Błąd", "Dodaj przynajmniej 3 miasta.")
        return

    generations = scale_generations.get()
    grid_size = scale_grid.get()

    canvas.delete("path")
    canvas_anim.delete("path")
    draw_points(canvas_anim)

    def update_animation(best_route, generation):
        draw_path(best_route, points, canvas_anim, tag="path")
        label_result.config(text=f"Generacja: {generation+1}  |  Długość: {round(distance(best_route, points), 2)}")

    def run_thread():
        best_route = run_cEA_TSP_animated(
            points,
            grid_size,
            generations,
            update_callback=update_animation,
            delay_func=lambda: scale_speed.get(),
            mutation_prob_func=lambda: scale_mutation.get(),
            neigh_type_func=lambda: neighborhood_type.get(),
            neigh_radius_func=lambda: neighborhood_radius.get()
        )
        draw_path(best_route, points, canvas, tag="path")

    Thread(target=run_thread).start()

# Clears everything from both canvases and resets state
def clear_canvas():
    global points
    points.clear()
    canvas.delete("all")
    canvas_anim.delete("all")
    label_result.config(text="")

# --- GUI setup ---

root = tk.Tk()
root.title("Algorytm Ewolucyjny Komórkowy - Problem Komiwojażera")

canvas = tk.Canvas(root, width=500, height=500, bg="white")
canvas.grid(row=0, column=0, rowspan=20, padx=10, pady=10)
canvas.bind("<Button-1>", on_canvas_click)

canvas_anim = tk.Canvas(root, width=500, height=500, bg="lightgray")
canvas_anim.grid(row=0, column=1, rowspan=20, padx=10, pady=10)

tk.Label(root, text="Liczba generacji").grid(row=0, column=2, sticky="w")
scale_generations = tk.Scale(root, from_=10, to=500, orient="horizontal")
scale_generations.set(100)
scale_generations.grid(row=1, column=2, padx=10)

tk.Label(root, text="Rozmiar siatki").grid(row=2, column=2, sticky="w")
scale_grid = tk.Scale(root, from_=2, to=10, orient="horizontal")
scale_grid.set(5)
scale_grid.grid(row=3, column=2, padx=10)

tk.Label(root, text="Szybkość animacji (s)").grid(row=4, column=2, sticky="w")
scale_speed = tk.Scale(root, from_=0.0, to=1.0, resolution=0.01, orient="horizontal")
scale_speed.set(0.3)
scale_speed.grid(row=5, column=2, padx=10)

tk.Label(root, text="Prawdopodobieństwo mutacji").grid(row=6, column=2, sticky="w")
scale_mutation = tk.Scale(root, from_=0.0, to=1.0, resolution=0.01, orient="horizontal")
scale_mutation.set(0.1)
scale_mutation.grid(row=7, column=2, padx=10)

tk.Label(root, text="Typ sąsiedztwa").grid(row=8, column=2, sticky="w")
neighborhood_type = tk.StringVar()
type_menu = tk.OptionMenu(root, neighborhood_type, "kompaktowe", "liniowe")
type_menu.grid(row=9, column=2, padx=10)
neighborhood_type.set("kompaktowe")

tk.Label(root, text="Rozmiar sąsiedztwa").grid(row=10, column=2, sticky="w")
neighborhood_radius = tk.Scale(root, from_=1, to=5, orient="horizontal")
neighborhood_radius.set(1)
neighborhood_radius.grid(row=11, column=2, padx=10)

start_button = tk.Button(root, text="Uruchom algorytm", command=start_algorithm)
start_button.grid(row=12, column=2, pady=10)

clear_button = tk.Button(root, text="Wyczyść planszę", command=clear_canvas)
clear_button.grid(row=13, column=2, pady=5)

label_result = tk.Label(root, text="")
label_result.grid(row=14, column=2)

root.mainloop()