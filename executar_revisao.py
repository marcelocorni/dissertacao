# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#   "duckdb==1.5.5", "numpy==2.5.3", "pyarrow==23.0.1",
#   "scikit-learn==1.9.1", "matplotlib==3.11.2", "joblib==1.6.0",
#   "torch==2.14.0+cu130",
#   "pandas==2.3.3", "pypdf==6.10.0", "threadpoolctl==3.7.0", "scipy==1.18.1",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cu130" }
# [[tool.uv.index]]
# name = "pytorch-cu130"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
# ///
"""Revisão opt-in; a opção padrão planeja, não processa ou consulta RPC."""
from revisao_experimental.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
