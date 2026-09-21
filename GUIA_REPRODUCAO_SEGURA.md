# Guia de execução do experimento

Este guia apresenta a ordem completa. Os comandos abaixo gravam em uma raiz de
execução separada e leem os Parquets do caminho definido em
`.\configuracao\pipeline.json`.

## Configuração dos dados

Antes da primeira execução, ajuste o único caminho externo do pipeline:

```json
{
  "dados": {
    "root_template": "F:\\ethereum-{year}"
  },
  "amostragem": {
    "percentual_blocos": 1.0
  }
}
```

O marcador `{year}` é substituído automaticamente por `2024` ou `2025`.
O percentual `1.0` reproduz a amostra original. Valores maiores processam uma
fatia maior, sempre por blocos completos, e aumentam proporcionalmente o custo
computacional e o volume dos artefatos.

## Variáveis da execução

```powershell
$run = ".\execucoes\execucao-2026-09-20"
$tmp = "$run\temporarios\duckdb"
$features = "$run\03-machine-learning\01-features"
$ifResult = "$run\03-machine-learning\03-if\resultados"
$aeResult = "$run\03-machine-learning\02-ae\resultados"
$analysis = "$run\05-analise-resultados\resultados"
$labels = "$run\04-pos-processamento\resultados-labelcloud"
$semantic = "$run\04-pos-processamento\06-validacao-semantica-front-running"
$semanticResults = "$semantic\resultados"
$semanticCache = "$semantic\cache"
New-Item -ItemType Directory -Force -Path $run, $tmp | Out-Null
```

## 1. Auditoria das features

```powershell
uv run ".\03-machine-learning\01-features\01_auditoria_selecao_features.py" `
  --output-dir "$features\resultados" --temp-dir $tmp

uv run ".\03-machine-learning\01-features\04_gerar_figuras_correlacao_pdf.py" `
  --results-dir "$features\resultados"
```

Resultados: `.\execucoes\execucao-2026-09-20\03-machine-learning\01-features\resultados`.

## 2. Matrizes e recortes temporais

```powershell
uv run ".\03-machine-learning\01-features\02_gerar_matrizes_features.py" `
  --start-date 2024-04-01 --end-date 2024-12-31 `
  --dataset-name treino_2024_pos_dencun `
  --output-dir "$features\resultados-matrizes" --temp-dir $tmp

uv run ".\03-machine-learning\01-features\03_gerar_splits_temporais.py" `
  --output-root "$features\resultados-splits" `
  --training-dir "$features\resultados-matrizes" --temp-dir $tmp
```

Resultados: `.\execucoes\execucao-2026-09-20\03-machine-learning\01-features\resultados-matrizes`
e `.\execucoes\execucao-2026-09-20\03-machine-learning\01-features\resultados-splits`.

## 3. Pré-processamento e drift

```powershell
uv run ".\03-machine-learning\01-features\05_preprocessar_matrizes.py" `
  --split-manifest "$features\resultados-splits\00_manifest_splits_temporais.json" `
  --output-dir "$features\resultados-preprocessamento" --temp-dir $tmp

uv run ".\03-machine-learning\01-features\06_analisar_drift_temporal.py" `
  --split-manifest "$features\resultados-splits\00_manifest_splits_temporais.json" `
  --preprocessing-manifest "$features\resultados-preprocessamento\05_manifest_preprocessamento.json" `
  --transformation-dictionary "$features\resultados-preprocessamento\02_dicionario_transformacoes.csv" `
  --output-dir "$features\resultados-drift" --temp-dir $tmp
```

## 4. Isolation Forest e Autoencoder

```powershell
uv run ".\03-machine-learning\03-if\01_treinar_isolation_forest.py" `
  --split-manifest "$features\resultados-splits\00_manifest_splits_temporais.json" `
  --preprocessing-manifest "$features\resultados-preprocessamento\05_manifest_preprocessamento.json" `
  --output-dir $ifResult --temp-dir $tmp

uv run ".\03-machine-learning\02-ae\01_treinar_autoencoder.py" `
  --split-manifest "$features\resultados-splits\00_manifest_splits_temporais.json" `
  --preprocessing-manifest "$features\resultados-preprocessamento\05_manifest_preprocessamento.json" `
  --if-results-dir $ifResult --output-dir $aeResult --temp-dir $tmp --device auto
```

Execute o IF antes do AE para gerar a concordância entre os modelos.

## 5. Análise integrada AE/IF

```powershell
uv run ".\05-analise-resultados\01_analisar_ae_if.py" `
  --split-manifest "$features\resultados-splits\00_manifest_splits_temporais.json" `
  --ae-results-dir $aeResult --if-results-dir $ifResult `
  --output-dir $analysis --temp-dir $tmp
```

## 6. Comparação opcional com Label Cloud

```powershell
uv run ".\04-pos-processamento\02_preparar_labelcloud_historico.py" `
  --input-csv ".\04-pos-processamento\resultados-labelcloud\etherscan_labl_cloud_202609172241.csv" `
  --output-dir $labels

uv run ".\04-pos-processamento\03_vincular_labelcloud_candidatos.py" `
  --tags "$labels\01_labelcloud_historico_normalizado.parquet" `
  --candidates-dir "$analysis\candidatos" `
  --output-dir "$labels\vinculos-candidatos" --temp-dir $tmp

uv run ".\04-pos-processamento\04_gerar_rotulos_hipoteticos_labelcloud.py" `
  --tags "$labels\01_labelcloud_historico_normalizado.parquet" `
  --split-manifest "$features\resultados-splits\00_manifest_splits_temporais.json" `
  --output-dir $labels --temp-dir $tmp

uv run ".\05-analise-resultados\03_avaliar_labelcloud_hipotetico.py" `
  --labels "$labels\05_rotulos_binarios_hipoteticos_labelcloud.parquet" `
  --ae-results-dir $aeResult --if-results-dir $ifResult `
  --split-manifest "$features\resultados-splits\00_manifest_splits_temporais.json" `
  --output-dir "$run\05-analise-resultados\resultados-labelcloud-hipotetico" `
  --temp-dir $tmp
```

## 7. Candidatos estruturais em C#

```powershell
$csharpProject = ".\04-pos-processamento\05-rotulador-front-running-csharp"
$csharpOut = "$run\04-pos-processamento\05-rotulador-front-running-csharp\resultados-v3"
$csharpWindows = "$csharpOut\sampled\independent_sampled"

dotnet run --project $csharpProject --configuration Release -- `
  --input $features --input-mode sampled `
  --output-dir $csharpOut --detectors all --memory-limit 12GB --threads 8
```

## 8. Auditoria das saídas C#

```powershell
uv run ".\04-pos-processamento\06-validacao-semantica-front-running\07_auditar_saidas_csharp.py" `
  --csharp-root $csharpWindows `
  --output-dir "$semanticResults\auditoria-csharp"
```

## 9. Validação semântica e adjudicação

Para reutilizar o cache RPC disponível:

```powershell
Copy-Item `
  -LiteralPath ".\04-pos-processamento\06-validacao-semantica-front-running\cache" `
  -Destination $semanticCache -Recurse

$env:ETH_RPC_URL = Read-Host "Endpoint RPC Ethereum"

uv run ".\04-pos-processamento\06-validacao-semantica-front-running\11_executar_pipeline_janelas.py" `
  --csharp-root $csharpWindows --results-root $semanticResults `
  --cache-root $semanticCache `
  --audit-manifest "$semanticResults\auditoria-csharp\02_manifest_auditoria_csharp.json" `
  --regression-manifest ".\04-pos-processamento\06-validacao-semantica-front-running\resultados-blocos\transicao_dencun_2024\04_regressao_rpc.json"
```

## 10. Consolidação dos rótulos

```powershell
uv run ".\04-pos-processamento\06-validacao-semantica-front-running\12_consolidar_rotulos_semanticos.py" `
  --results-root $semanticResults --output-dir "$semanticResults\consolidado"
```

## 11. Avaliação final com rótulos semânticos

```powershell
uv run ".\05-analise-resultados\02_avaliar_rotulos_semanticos.py" `
  --labels "$semanticResults\consolidado\03_rotulos_transacao.parquet" `
  --event-roles "$semanticResults\consolidado\02_papeis_evento.parquet" `
  --ae-results-dir $aeResult --if-results-dir $ifResult `
  --split-manifest "$features\resultados-splits\00_manifest_splits_temporais.json" `
  --output-dir "$run\05-analise-resultados\resultados-rotulos-semanticos" `
  --temp-dir $tmp
```

## 12. Interface de consulta

```powershell
$env:SEMANTIC_RESULTS_ROOT = $semanticResults
uv run --with streamlit --with duckdb streamlit run `
  ".\04-pos-processamento\06-validacao-semantica-front-running\05_interface_dossie_semantico.py"
```

A descrição de cada artefato está no README da etapa que o produz.
