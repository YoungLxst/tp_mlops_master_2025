import os
import subprocess
from pathlib import Path

# Aller dans le dossier du script
root = Path(__file__).resolve().parent
os.chdir(root)

# Créer dossier artifacts si absent
(root / "artifacts").mkdir(exist_ok=True)

cmd = [
    "mlflow", "server",
    "--backend-store-uri", "sqlite:///mlflow.db",
    "--default-artifact-root", "./artifacts",
    "--host", "0.0.0.0",
    "--port", "5000"
]

print(">>> Starting MLflow server...")
subprocess.run(cmd)