# Candidatos estruturais de front-running

## Objetivo

Aplicar regras de `insertion`, `displacement` e `suppression` diretamente aos
Parquets Ethereum. O projeto usa DuckDB.NET e .NET 8.

Os resultados desta etapa são candidatos. A confirmação ocorre em
`.\04-pos-processamento\06-validacao-semantica-front-running`.

## Regras

- `insertion`: procura três transações ordenadas e atribui os papéis
  `attacker_front`, `victim` e `attacker_back`.
- `displacement`: procura uma transação anterior com mesmo destino, valor
  compatível e preço de gás superior ao da vítima.
- `suppression`: identifica sequências competitivas anteriores à âncora; exige
  evidência externa de mempool e não gera positivo apenas com blocos minerados.

## Execução

```powershell
dotnet run `
  --project ".\04-pos-processamento\05-rotulador-front-running-csharp" `
  --configuration Release -- `
  --input ".\03-machine-learning\01-features" `
  --input-mode sampled `
  --output-dir ".\04-pos-processamento\05-rotulador-front-running-csharp\resultados-v3" `
  --detectors all --memory-limit 12GB --threads 8
```

O modo `sampled` aplica as regras a todas as transações dos blocos selecionados
pelos recortes temporais, sem depender da classificação do AE ou do IF.
O caminho dos Parquets é lido de `.\configuracao\pipeline.json`.

## Resultados

Raiz:
`.\04-pos-processamento\05-rotulador-front-running-csharp\resultados-v3\sampled\independent_sampled`.

Cada uma das sete janelas contém:

| Artefato | Conteúdo |
|---|---|
| `01_detection_events.parquet` | eventos candidatos, ordem, hashes e papéis |
| `02_transaction_labels.parquet` | papéis candidatos por transação |
| `03_summary.csv` | contagens por detector |
| `04_manifest_run.json` | parâmetros, fontes e cobertura |
| `05_enrichment_queue.parquet` | hashes que exigem calldata, receipt e logs |

O caminho de uma janela segue o padrão:
`.\04-pos-processamento\05-rotulador-front-running-csharp\resultados-v3\sampled\independent_sampled\<janela>`.
