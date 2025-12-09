import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import subprocess
import os

class GameOfLifeLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("Game of Life Launcher (CPU vs GPU)")
        self.root.geometry("600x700")

        # --- Sekcja Konfiguracji ---
        config_frame = ttk.LabelFrame(root, text="Konfiguracja symulacji", padding="10")
        config_frame.pack(fill="x", padx=10, pady=5)

        # Wybór wersji (CPU / GPU / MPI)
        self.mode_var = tk.StringVar(value="CPU")
        ttk.Label(config_frame, text="Wersja obliczen:").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Radiobutton(config_frame, text="CPU (OpenMP)", variable=self.mode_var, value="CPU").grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(config_frame, text="GPU (CUDA)", variable=self.mode_var, value="GPU").grid(row=0, column=2, sticky="w")
        ttk.Radiobutton(config_frame, text="MPI", variable=self.mode_var, value="MPI").grid(row=0, column=3, sticky="w")

        # Parametry: Wysokosc (H), Szerokosc (W), Kroki
        ttk.Label(config_frame, text="Wysokosc (H):").grid(row=1, column=0, sticky="w", pady=2)
        self.entry_h = ttk.Entry(config_frame, width=10)
        self.entry_h.insert(0, "20")
        self.entry_h.grid(row=1, column=1, sticky="w")

        ttk.Label(config_frame, text="Szerokosc (W):").grid(row=2, column=0, sticky="w", pady=2)
        self.entry_w = ttk.Entry(config_frame, width=10)
        self.entry_w.insert(0, "40")
        self.entry_w.grid(row=2, column=1, sticky="w")

        ttk.Label(config_frame, text="Liczba krokow:").grid(row=3, column=0, sticky="w", pady=2)
        self.entry_steps = ttk.Entry(config_frame, width=10)
        self.entry_steps.insert(0, "10")
        self.entry_steps.grid(row=3, column=1, sticky="w")

        ttk.Label(config_frame, text="Procesy MPI:").grid(row=4, column=0, sticky="w", pady=2)
        self.entry_mpi_procs = ttk.Entry(config_frame, width=10)
        self.entry_mpi_procs.insert(0, "2")
        self.entry_mpi_procs.grid(row=4, column=1, sticky="w")

        # --- Sekcja Akcji ---
        action_frame = ttk.Frame(root, padding="10")
        action_frame.pack(fill="x", padx=10)

        # Przycisk kompilacji (uruchamia make)
        self.btn_compile = ttk.Button(action_frame, text="Skompiluj (make)", command=self.run_make)
        self.btn_compile.pack(side="left", padx=5)

        # Przycisk uruchomienia
        self.btn_run = ttk.Button(action_frame, text="URUCHOM SYMULACJE", command=self.run_simulation)
        self.btn_run.pack(side="left", padx=5, expand=True, fill="x")

        # --- Sekcja Wyników ---
        output_frame = ttk.LabelFrame(root, text="Wyjscie z konsoli", padding="5")
        output_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Pole tekstowe (ScrolledText)
        self.output_area = scrolledtext.ScrolledText(output_frame, state='disabled', font=("fixed", 10))
        self.output_area.pack(fill="both", expand=True)

    def log(self, text, clear=False):
        self.output_area.config(state='normal')
        if clear:
            self.output_area.delete(1.0, tk.END)
        self.output_area.insert(tk.END, text + "\n")
        self.output_area.see(tk.END)
        self.output_area.config(state='disabled')

    def run_make(self):
        self.log(">>> Uruchamianie 'make'...", clear=True)
        try:
            result = subprocess.run(["make"], capture_output=True, text=True)
            if result.stdout: self.log(result.stdout)
            if result.stderr: self.log(result.stderr)
            if result.returncode == 0:
                self.log(">>> Kompilacja zakonczona sukcesem.")
            else:
                self.log(">>> Blad kompilacji.")
        except FileNotFoundError:
            self.log(">>> Blad: Nie znaleziono polecenia 'make'. Upewnij sie, ze jest zainstalowane.")

    def run_simulation(self):
        # Pobieranie danych
        mode = self.mode_var.get()
        h_str = self.entry_h.get()
        w_str = self.entry_w.get()
        steps_str = self.entry_steps.get()
        mpi_procs_str = self.entry_mpi_procs.get()

        # Walidacja
        if not (h_str.isdigit() and w_str.isdigit() and steps_str.isdigit() and mpi_procs_str.isdigit()):
            messagebox.showerror("Blad", "Wymiary, kroki i liczba procesów muszą być liczbami całkowitymi!")
            return

        if mode == "CPU":
            executable = "./gol-cpu"
            cmd = [executable, h_str, w_str, steps_str]
        elif mode == "GPU":
            executable = "./gol-gpu"
            cmd = [executable, h_str, w_str, steps_str]
        elif mode == "MPI":
            executable = "./gol-mpi"
            mpi_procs = int(mpi_procs_str)
            cmd = ["mpirun", "-np", str(mpi_procs), executable, h_str, w_str, steps_str, str(mpi_procs)]
        else:
            messagebox.showerror("Blad", "Nieznana wersja!")
            return

        if not os.path.exists(executable):
            messagebox.showwarning("Brak pliku", f"Nie znaleziono pliku {executable}.\nKliknij najpierw 'Skompiluj (make)'.")
            return

        self.log(f"\n>>> Uruchamianie: {' '.join(cmd)}", clear=True)
        self.root.update()

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)

            self.log(result.stdout)
            if result.stderr:
                self.log("BLEDY/OSTRZEZENIA:")
                self.log(result.stderr)

        except Exception as e:
            self.log(f"Wystapil nieoczekiwany blad: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use('clam')
    app = GameOfLifeLauncher(root)
    root.mainloop()
