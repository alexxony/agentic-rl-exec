import subprocess

def run(cmd, timeout=None):
    print(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if r.stdout:
        print(r.stdout[-2500:])
    if r.stderr:
        print("STDERR:", r.stderr[-2500:])
    print(f"(exit {r.returncode})")
    return r.returncode

# Patch gsm8k.py's import to bypass verl/__init__.py (which pulls in ray, a heavy dep
# unneeded for plain preprocessing). hdfs_io itself only needs os/shutil/logging.
patch = r'''
import sys, types, importlib.util

# stub the 'verl' and 'verl.utils' packages so `from verl.utils.hdfs_io import ...`
# resolves without executing verl/__init__.py (which imports ray).
verl_pkg = types.ModuleType("verl")
verl_pkg.__path__ = ["/content/verl/verl"]
sys.modules["verl"] = verl_pkg

verl_utils_pkg = types.ModuleType("verl.utils")
verl_utils_pkg.__path__ = ["/content/verl/verl/utils"]
sys.modules["verl.utils"] = verl_utils_pkg

spec = importlib.util.spec_from_file_location("verl.utils.hdfs_io", "/content/verl/verl/utils/hdfs_io.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["verl.utils.hdfs_io"] = mod
spec.loader.exec_module(mod)

sys.argv = ["gsm8k.py", "--local_save_dir", "/content/data/gsm8k"]
exec(open("/content/verl/examples/data_preprocess/gsm8k.py").read())
'''

with open("/tmp/run_gsm8k_preprocess.py", "w") as f:
    f.write(patch)

run("python3 /tmp/run_gsm8k_preprocess.py", timeout=180)
run("ls -la /content/data/gsm8k/")
run("python3 -c \"import pandas as pd; df = pd.read_parquet('/content/data/gsm8k/train.parquet'); print(df.shape); print(df.iloc[0].to_dict())\"")
