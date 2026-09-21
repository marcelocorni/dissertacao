# Features, recortes e pré-processamento

## Configuração da amostragem

Os artefatos `01`, `02` e `03` leem
`.\configuracao\pipeline.json`. O campo
`amostragem.percentual_blocos` controla a fração de blocos usada em toda a
geração de features e recortes. A execução original utiliza `1.0%`.

A seleção mantém o bloco completo e é determinística: `1%` equivale a módulo
`100`, `5%` a módulo `20`, `10%` a módulo `10` e `100%` a módulo `1`. Dessa
forma, todas as transações de cada bloco selecionado permanecem juntas. Alterar
o percentual não modifica resultados existentes; ele só afeta novas execuções.
O parâmetro `--block-modulus` existe apenas para uma substituição pontual.

## 1. Auditoria e correlações

Código:
`.\03-machine-learning\01-features\01_auditoria_selecao_features.py`.

```powershell
uv run ".\03-machine-learning\01-features\01_auditoria_selecao_features.py" `
  --output-dir ".\03-machine-learning\01-features\resultados" `
  --temp-dir ".\.tmp\duckdb"

uv run ".\03-machine-learning\01-features\04_gerar_figuras_correlacao_pdf.py" `
  --results-dir ".\03-machine-learning\01-features\resultados"
```

Resultados em
`.\03-machine-learning\01-features\resultados`:

| Artefato | Conteúdo |
|---|---|
| `01_schema_parquet.csv` | esquema dos Parquets de origem |
| `02_resumo_dataset.json` | período, arquivos, linhas e blocos |
| `03_nulidade_colunas_originais.csv` | nulos por coluna original |
| `04_distribuicao_transaction_type.csv` | distribuição dos tipos de transação |
| `05_distribuicao_receipt_status.csv` | sucesso e falha dos receipts |
| `06_perfil_features.csv` | estatísticas das features candidatas |
| `07_amostra_features.parquet` | amostra on-chain auditada |
| `08_features_constantes_na_amostra.csv` | features sem variância |
| `09_correlacao_pearson.csv` | matriz de Pearson |
| `10_correlacao_spearman.csv` | matriz de Spearman |
| `11_correlacao_pearson.png` | figura raster de Pearson |
| `12_correlacao_spearman.png` | figura raster de Spearman |
| `13_pares_alta_correlacao.csv` | pares acima do limiar |
| `14_metodologia_execucao.md` | parâmetros metodológicos |
| `15_resultado_execucao.json` | manifesto da auditoria |
| `16_correlacao_pearson_vetorial.pdf` | figura vetorial de Pearson |
| `17_correlacao_spearman_vetorial.pdf` | figura vetorial de Spearman |
| `18_matrizes_correlacao_vetorial.pdf` | PDF combinado |

## 2. Matrizes A, B e C

Código:
`.\03-machine-learning\01-features\02_gerar_matrizes_features.py`.

```powershell
uv run ".\03-machine-learning\01-features\02_gerar_matrizes_features.py" `
  --start-date 2024-04-01 --end-date 2024-12-31 `
  --dataset-name treino_2024_pos_dencun `
  --output-dir ".\03-machine-learning\01-features\resultados-matrizes" `
  --temp-dir ".\.tmp\duckdb"
```

- Matriz A: features transacionais.
- Matriz B: matriz A mais contexto intrabloco e temporal.
- Matriz C: matriz B mais limite de gás e taxa total paga.

Resultados em
`.\03-machine-learning\01-features\resultados-matrizes`:

| Artefato | Conteúdo |
|---|---|
| `01_resumo_geracao.json` | contagens e período |
| `02_dicionario_features.csv` | definição e matriz de cada feature |
| `03_validacao_features.csv` | nulos, infinitos e variância |
| `04_manifest_matrizes.json` | contrato das matrizes |
| `05_metadados.parquet` | `row_id`, hash, endereço, bloco e timestamp |
| `06_matriz_A_baseline.parquet` | matriz A |
| `07_matriz_B_contexto.parquet` | matriz B |
| `08_matriz_C_estendida.parquet` | matriz C |
| `09_metodologia_matrizes.md` | método de geração |
| `10_manifest_execucao.json` | proveniência e parâmetros |
| `11_arquivos_origem.txt` | Parquets lidos |

## 3. Recortes temporais

Código:
`.\03-machine-learning\01-features\03_gerar_splits_temporais.py`.

```powershell
uv run ".\03-machine-learning\01-features\03_gerar_splits_temporais.py" `
  --output-root ".\03-machine-learning\01-features\resultados-splits" `
  --training-dir ".\03-machine-learning\01-features\resultados-matrizes" `
  --temp-dir ".\.tmp\duckdb"
```

O manifesto global fica em
`.\03-machine-learning\01-features\resultados-splits\00_manifest_splits_temporais.json`.
Cada pasta de recorte possui os mesmos artefatos `01` a `11` descritos para as
matrizes de treinamento.

## 4. Pré-processamento

Código:
`.\03-machine-learning\01-features\05_preprocessar_matrizes.py`.

```powershell
uv run ".\03-machine-learning\01-features\05_preprocessar_matrizes.py" `
  --split-manifest ".\03-machine-learning\01-features\resultados-splits\00_manifest_splits_temporais.json" `
  --output-dir ".\03-machine-learning\01-features\resultados-preprocessamento" `
  --temp-dir ".\.tmp\duckdb"
```

O RobustScaler é ajustado somente no treino e aplicado sem reajuste às demais
janelas. Cada recorte recebe:

- `12_matriz_A_preprocessada.parquet`;
- `13_matriz_B_preprocessada.parquet`;
- `14_matriz_C_preprocessada.parquet`;
- `15_manifest_preprocessamento.json`.

Resultados centrais em
`.\03-machine-learning\01-features\resultados-preprocessamento`:

| Artefato | Conteúdo |
|---|---|
| `01_parametros_robust_scaler.json` | medianas e escalas do treino |
| `02_dicionario_transformacoes.csv` | transformação por feature |
| `03_perfil_treino_matriz_B_preprocessada.csv` | perfil da matriz B |
| `04_validacao_recortes.csv` | cobertura e valores inválidos |
| `05_manifest_preprocessamento.json` | assinatura do pré-processamento |

## 5. Drift temporal

Código:
`.\03-machine-learning\01-features\06_analisar_drift_temporal.py`.

```powershell
uv run ".\03-machine-learning\01-features\06_analisar_drift_temporal.py" `
  --split-manifest ".\03-machine-learning\01-features\resultados-splits\00_manifest_splits_temporais.json" `
  --preprocessing-manifest ".\03-machine-learning\01-features\resultados-preprocessamento\05_manifest_preprocessamento.json" `
  --transformation-dictionary ".\03-machine-learning\01-features\resultados-preprocessamento\02_dicionario_transformacoes.csv" `
  --output-dir ".\03-machine-learning\01-features\resultados-drift" `
  --temp-dir ".\.tmp\duckdb"
```

Resultados em
`.\03-machine-learning\01-features\resultados-drift`:

| Artefato | Conteúdo |
|---|---|
| `01_drift_features_recortes.csv` | PSI e KS aproximado por feature e janela |
| `02_resumo_drift_recortes.csv` | resumo por janela |
| `03_ranking_features_drift.csv` | ranking das features |
| `04_drift_por_matriz.csv` | drift agregado por matriz |
| `05_drift_transacoes_tipo_4.csv` | efeito das transações tipo 4 |
| `06_distribuicao_tipos_transacao.csv` | tipos por janela |
| `07_heatmap_psi.pdf/.png` | mapa de PSI |
| `08_heatmap_ks_aproximado.pdf/.png` | mapa de KS |
| `09_metodologia_drift.md` | método de cálculo |
| `10_manifest_drift.json` | manifesto da execução |
