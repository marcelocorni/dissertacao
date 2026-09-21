# Isolation Forest

## Objetivo

Treinar cinco configurações de Isolation Forest nas matrizes preprocessadas. O
ajuste ocorre apenas no treino e os limiares Q99, Q99,5 e Q99,9 são calculados
na validação temporal.

Código:
`.\03-machine-learning\03-if\01_treinar_isolation_forest.py`.

## Execução

```powershell
uv run ".\03-machine-learning\03-if\01_treinar_isolation_forest.py" `
  --split-manifest ".\03-machine-learning\01-features\resultados-splits\00_manifest_splits_temporais.json" `
  --preprocessing-manifest ".\03-machine-learning\01-features\resultados-preprocessamento\05_manifest_preprocessamento.json" `
  --output-dir ".\03-machine-learning\03-if\resultados" `
  --temp-dir ".\.tmp\duckdb"
```

## Resultados

Diretório: `.\03-machine-learning\03-if\resultados`.

| Artefato | Conteúdo |
|---|---|
| `01_configuracoes_experimento.json` | parâmetros e assinatura |
| `02_resumo_treinamento.csv` | amostra, árvores, duração e modelos |
| `03_limiares_validacao.csv` | Q99, Q99,5 e Q99,9 da validação |
| `04_estatisticas_escores.csv` | distribuição dos escores por janela |
| `05_taxas_anomalias_recortes.csv` | anomalias por limiar e janela |
| `06_taxas_anomalias_tipo4.csv` | análise estratificada do tipo 4 |
| `07_concordancia_validacao.csv` | concordância entre configurações IF |
| `08_taxas_anomalias_recortes.pdf/.png` | taxas por janela |
| `09_metodologia_isolation_forest.md` | método executado |
| `10_manifest_isolation_forest.json` | arquivos e assinaturas |
| `modelos\*.joblib` | estimadores e contrato de features |
| `escores\<configuração>\*.parquet` | escore e flags por transação |
