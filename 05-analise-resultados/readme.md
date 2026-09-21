# Análise dos resultados

## 1. Análise integrada AE/IF

O script combina os escores do Autoencoder e do Isolation Forest, calcula
taxas, interseção, união e Jaccard e materializa candidatos rastreáveis.

Código:
`.\05-analise-resultados\01_analisar_ae_if.py`.

```powershell
uv run ".\05-analise-resultados\01_analisar_ae_if.py" `
  --split-manifest ".\03-machine-learning\01-features\resultados-splits\00_manifest_splits_temporais.json" `
  --ae-results-dir ".\03-machine-learning\02-ae\resultados" `
  --if-results-dir ".\03-machine-learning\03-if\resultados" `
  --output-dir ".\05-analise-resultados\resultados" `
  --temp-dir ".\.tmp\duckdb"
```

Resultados em `.\05-analise-resultados\resultados`:

| Artefato | Conteúdo |
|---|---|
| `01_resumo_integrado_ae_if.csv` | taxas, interseção, união e Jaccard |
| `02_resumo_candidatos.csv` | quantidade de candidatos por configuração |
| `03_top_candidatos.csv` | candidatos com maior consenso |
| `04_if_quantis_escores.pdf/.png` | evolução dos quantis do IF |
| `05_if_curvas_sobrevivencia.pdf/.png` | distribuição acumulada dos escores IF |
| `06_heatmaps_taxas_ae_if.pdf/.png` | taxas dos dois modelos |
| `07_heatmap_concordancia_ae_if.pdf/.png` | concordância AE/IF |
| `08_metodologia_analise_integrada.md` | método da integração |
| `09_manifest_analise_integrada.json` | entradas, parâmetros e saídas |
| `candidatos\<configuração>\*.parquet` | união de candidatos no limiar Q99,5 |

## 2. Avaliação hipotética com Label Cloud

Código:
`.\05-analise-resultados\03_avaliar_labelcloud_hipotetico.py`.

```powershell
uv run ".\05-analise-resultados\03_avaliar_labelcloud_hipotetico.py" `
  --labels ".\04-pos-processamento\resultados-labelcloud\05_rotulos_binarios_hipoteticos_labelcloud.parquet" `
  --ae-results-dir ".\03-machine-learning\02-ae\resultados" `
  --if-results-dir ".\03-machine-learning\03-if\resultados" `
  --split-manifest ".\03-machine-learning\01-features\resultados-splits\00_manifest_splits_temporais.json" `
  --output-dir ".\05-analise-resultados\resultados-labelcloud-hipotetico" `
  --temp-dir ".\.tmp\duckdb"
```

Resultados em
`.\05-analise-resultados\resultados-labelcloud-hipotetico`:

| Artefato | Conteúdo |
|---|---|
| `01_metricas_discriminacao.csv` | ROC-AUC e PR-AUC |
| `02_metricas_limiares.csv` | matriz de confusão e métricas em Q99, Q99,5 e Q99,9 |
| `03_metricas_top_k.csv` | precision, recall e lift no top-k |
| `04_manifest_avaliacao.json` | fontes e definição dos rótulos |
| `05_curvas_roc_pr_<janela>.pdf/.png` | curvas ROC e precisão–recall |
| `06_heatmap_roc_auc.pdf/.png` | comparação de ROC-AUC |
| `07_heatmap_pr_auc.pdf/.png` | comparação de PR-AUC |
| `08_matrizes_confusao_teste_<limiar>.pdf/.png` | matrizes de confusão do teste |
| `09_distribuicao_escores_teste.pdf/.png` | escores por classe hipotética |
| `10_lift_top_k_teste.pdf/.png` | lift no teste final |
| `11_sintese_resultados.md` | síntese das métricas |

## 3. Avaliação final com rótulos semânticos

Código:
`.\05-analise-resultados\02_avaliar_rotulos_semanticos.py`.

```powershell
uv run ".\05-analise-resultados\02_avaliar_rotulos_semanticos.py"
```

A avaliação usa atacantes como positivos e vítimas dos mesmos eventos como
negativos contextuais. Transações sem papel adjudicado não são presumidas
negativas. A política estrita considera somente eventos confirmados; a análise
de sensibilidade inclui confirmados ou prováveis.

Resultados em
`.\05-analise-resultados\resultados-rotulos-semanticos`:

| Artefato | Conteúdo |
|---|---|
| `01_cobertura_rotulos.csv` | positivos, vítimas, conflitos e cobertura por janela e política |
| `02_metricas_discriminacao.csv` | ROC-AUC, PR-AUC e IC95% por modelo, janela, política e tipo de ataque |
| `03_metricas_limiares.csv` | matrizes de confusão e métricas em q990, q995 e q999 |
| `04_metricas_top_k.csv` | precisão, recall e lift nos top 1%, 5% e 10% |
| `05_metricas_pareadas_eventos.csv` | comparação dos escores de atacantes e vítimas do mesmo evento |
| `06_selecao_validacao_teste.csv` | configuração e limiar escolhidos na validação, com resultado no teste |
| `07_curvas_roc_pr_*.pdf/.png` | curvas da validação e do teste final |
| `08_heatmap_roc_auc_*.pdf/.png` | ROC-AUC nas sete janelas |
| `09_heatmap_pr_auc_*.pdf/.png` | PR-AUC nas sete janelas |
| `10_matrizes_confusao_*.pdf/.png` | matrizes no teste final |
| `11_heatmap_superioridade_pareada_*.pdf/.png` | frequência em que atacante supera vítima |
| `12_modelos_selecionados_teste.pdf/.png` | síntese dos modelos selecionados na validação |
| `13_sintese_resultados.md` | principais resultados e limitações |
| `14_metodologia_avaliacao.md` | protocolo completo da avaliação |
| `15_manifest_avaliacao.json` | fontes, parâmetros, hashes e contagens |
