import csv
import os
import re
import subprocess

IMAGES = [
    ("800x800", "src/leon800x800.jpg", "leon"),
    ("2000x2000", "src/napoli-2000x2000.jpg", "napoli"),
    ("5000x5000", "src/ciudad-5000x5000.jpg", "ciudad"),
]

TIME_RE = re.compile(r"Time:\s*([0-9.]+)\s*s")


def run_cmd(cmd):
    if os.name == "nt":
        cmd = ["wsl", "--"] + cmd
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    line = res.stdout.strip()
    print(line)
    return float(TIME_RE.search(line).group(1))


def add_row(rows, image, ksize, poster, mode, config, workers, t, t_seq):
    sp = t_seq / t
    rows.append({
        "image": image,
        "filter": f"{ksize}x{ksize}",
        "poster": poster,
        "mode": mode,
        "config": config,
        "workers": workers,
        "time_s": f"{t:.6f}",
        "speedup": f"{sp:.4f}",
        "efficiency": f"{(sp / workers):.4f}",
    })


def main():
    os.makedirs("output", exist_ok=True)
    rows = []

    for res_label, img_path, prefix in IMAGES:
        for ksize in [3, 5]:
            for poster in [3, 9]:
                save_img = (ksize == 3 and poster == 9)

                out_seq = f"output/{prefix}_cartoon_SEQ.jpg" if save_img else "-"
                t_seq = run_cmd(["./bin/cartoon_seq", img_path, out_seq, str(ksize), str(poster)])
                add_row(rows, res_label, ksize, poster, "Secuencial", "1 thread", 1, t_seq, t_seq)

                for th in [2, 4, 8, 12]:
                    out_omp = f"output/{prefix}_cartoon_OMP.jpg" if (save_img and th == 4) else "-"
                    t = run_cmd(["./bin/cartoon_omp", img_path, out_omp, str(ksize), str(poster), str(th)])
                    add_row(rows, res_label, ksize, poster, "OpenMP", f"{th} threads", th, t, t_seq)

                for np in [2, 4, 8, 12]:
                    out_mpi = f"output/{prefix}_cartoon_MPI.jpg" if (save_img and np == 4) else "-"
                    t = run_cmd(["mpirun", "--oversubscribe", "-np", str(np),
                                 "./bin/cartoon_mpi", img_path, out_mpi, str(ksize), str(poster)])
                    add_row(rows, res_label, ksize, poster, "MPI", f"{np} procs", np, t, t_seq)

                for np, th in [(2, 2), (2, 4), (4, 2), (3, 4)]:
                    out_hyb = f"output/{prefix}_cartoon_HYBRID.jpg" if (save_img and (np, th) == (2, 2)) else "-"
                    t = run_cmd(["mpirun", "--oversubscribe", "-np", str(np),
                                 "./bin/cartoon_hybrid", img_path, out_hyb, str(ksize), str(poster), str(th)])
                    add_row(rows, res_label, ksize, poster, "Hibrido", f"{np}p x {th}t", np * th, t, t_seq)

    csv_path = "output/resultados_benchmark.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} benchmark records to {csv_path}")


if __name__ == "__main__":
    if os.name == "nt":
        subprocess.run(["wsl", "--", "python3", "-u", "benchmark.py"], check=True)
    else:
        main()
