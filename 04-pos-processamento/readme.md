# Pós-processamento

Esta etapa aplica duas fontes externas aos escores de anomalia:

1. Label Cloud histórico, para comparação por endereço;
2. regras estruturais de front-running, seguidas de validação semântica on-chain.

Nenhuma dessas fontes participa do treinamento do Autoencoder ou do Isolation
Forest.

## Label Cloud

### Normalização

```powershell
uv run ".\04-pos-processamento\02_preparar_labelcloud_historico.py" `
  --input-csv ".\04-pos-processamento\resultados-labelcloud\etherscan_labl_cloud_202609172241.csv" `
  --output-dir ".\04-pos-processamento\resultados-labelcloud"
```

### Vínculo com candidatos AE/IF

```powershell
uv run ".\04-pos-processamento\03_vincular_labelcloud_candidatos.py" `
  --tags ".\04-pos-processamento\resultados-labelcloud\01_labelcloud_historico_normalizado.parquet" `
  --candidates-dir ".\05-analise-resultados\resultados\candidatos" `
  --output-dir ".\04-pos-processamento\resultados-labelcloud\vinculos-candidatos" `
  --temp-dir ".\.tmp\duckdb"
```

### Rótulo binário hipotético

```powershell
uv run ".\04-pos-processamento\04_gerar_rotulos_hipoteticos_labelcloud.py" `
  --tags ".\04-pos-processamento\resultados-labelcloud\01_labelcloud_historico_normalizado.parquet" `
  --split-manifest ".\03-machine-learning\01-features\resultados-splits\00_manifest_splits_temporais.json" `
  --output-dir ".\04-pos-processamento\resultados-labelcloud" `
  --temp-dir ".\.tmp\duckdb"
```

Resultados em `.\04-pos-processamento\resultados-labelcloud`:

| Artefato | Conteúdo |
|---|---|
| `01_labelcloud_historico_normalizado.csv/.parquet` | endereços, categorias e tags normalizadas |
| `02_resumo_labelcloud_historico.csv` | contagens por categoria |
| `03_pendencias_name_tag.csv` | registros sem tag descritiva |
| `04_manifest_labelcloud_historico.json` | integridade e proveniência |
| `05_rotulos_binarios_hipoteticos_labelcloud.parquet` | classe hipotética por transação |
| `06_resumo_rotulos_hipoteticos.csv` | positivos e negativos por janela |
| `07_manifest_rotulos_hipoteticos.json` | definição do experimento |
| `vinculos-candidatos\` | candidatos AE/IF enriquecidos com tags |

## Front-running

- Candidatos estruturais:
  `.\04-pos-processamento\05-rotulador-front-running-csharp`.
- Validação semântica:
  `.\04-pos-processamento\06-validacao-semantica-front-running`.

Os candidatos C# somente se tornam rótulos após o enriquecimento RPC, a
validação das regras e a adjudicação assistida.
