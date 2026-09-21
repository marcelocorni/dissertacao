# Autoencoder

## Objetivo

Treinar cinco configurações de Autoencoder nas matrizes preprocessadas. O
modelo aprende somente no recorte de treino. A validação interna do treino é
usada para *early stopping* e a janela de validação de 2025 calibra os limiares
Q99, Q99,5 e Q99,9.

Código:
`.\03-machine-learning\02-ae\01_treinar_autoencoder.py`.

## Execução

```powershell
uv run ".\03-machine-learning\02-ae\01_treinar_autoencoder.py" `
  --split-manifest ".\03-machine-learning\01-features\resultados-splits\00_manifest_splits_temporais.json" `
  --preprocessing-manifest ".\03-machine-learning\01-features\resultados-preprocessamento\05_manifest_preprocessamento.json" `
  --if-results-dir ".\03-machine-learning\03-if\resultados" `
  --output-dir ".\03-machine-learning\02-ae\resultados" `
  --temp-dir ".\.tmp\duckdb" `
  --device auto
```

## Resultados

Diretório: `.\03-machine-learning\02-ae\resultados`.

| Artefato | Conteúdo |
|---|---|
| `01_configuracoes_experimento.json` | arquitetura, sementes e assinatura |
| `02_resumo_treinamento.csv` | época ótima, duração e memória |
| `03_historico_treinamento.csv` | perdas por época |
| `04_limiares_validacao.csv` | Q99, Q99,5 e Q99,9 da validação |
| `05_estatisticas_escores.csv` | distribuição do erro por janela |
| `06_taxas_anomalias_recortes.csv` | anomalias por limiar e janela |
| `07_taxas_anomalias_tipo4.csv` | análise estratificada do tipo 4 |
| `08_concordancia_ae_validacao.csv` | concordância entre configurações AE |
| `09_concordancia_ae_if.csv` | concordância entre AE e IF |
| `10_curvas_treinamento.pdf/.png` | perdas de treino e validação interna |
| `11_taxas_anomalias_recortes.pdf/.png` | taxas por janela |
| `12_metodologia_autoencoder.md` | método executado |
| `13_manifest_autoencoder.json` | arquivos, dispositivo e assinaturas |
| `modelos\*.pt` | pesos e contrato de features |
| `escores\<configuração>\*.parquet` | erro e flags por transação |
