import os, sys, yaml, subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml", help="Master config (globals + profiles)")
    ap.add_argument("--procs",  type=int, default=0, help="Paralel process sayısı (0 → cpu_count()-1)")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    master_cfg_path = (project_root / args.config).resolve()
    if not master_cfg_path.exists():
        print(f"Config bulunamadı: {master_cfg_path}"); sys.exit(1)

    master = yaml.safe_load(open(master_cfg_path, "r", encoding="utf-8")) or {}
    profiles = (master.get("meta", {}) or {}).get("profiles_order") or list((master.get("profiles", {}) or {}).keys())
    if not profiles:
        print("Profil bulunamadı (master config -> profiles)."); sys.exit(1)

    max_workers = args.procs if args.procs > 0 else max(1, min(len(profiles), (mp.cpu_count() or 2) - 1))
    print(f"Paralel eğitim başlıyor. profiller={profiles}, proc={max_workers}")

    env = os.environ.copy()
    # Proje kökünü PYTHONPATH'e ekle (importlar için)
    env["PYTHONPATH"] = str(project_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    # 🔸 YENİ: Her run için ortak run_id oluştur
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"RUN_ID = {run_id}")

    def spawn(profile):
        # 🔸 YENİ: --run-id argümanını geçir
        return subprocess.Popen(
            [sys.executable, "-m", "scripts.train",
             "--config", str(master_cfg_path),
             "--profile", profile,
             "--run-id", run_id],
            cwd=str(project_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

    procs = {p: spawn(p) for p in profiles}

    def stream(tag, proc):
        assert proc.stdout is not None
        for line in proc.stdout:
            print(f"[{tag}] {line.rstrip()}")
        proc.stdout.close()

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(stream, p, procs[p]) for p in profiles]
        for f in as_completed(futs):
            _ = f.result()

    print("\nÖzet (exit code):")
    for p, pr in procs.items():
        pr.wait()
        print(f"  - {p}: {pr.returncode}")

if __name__ == "__main__":
    main()
