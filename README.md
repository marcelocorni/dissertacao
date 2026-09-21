# Pipeline de detecção de anomalias e front-running em Ethereum

Este repositório implementa o experimento completo com transações Ethereum em
Parquet, Autoencoder, Isolation Forest, regras estruturais de front-running e
validação semântica por dados on-chain.

Todos os caminhos desta documentação são relativos à raiz `src`. A localização
externa dos Parquets é definida uma única vez em
`.\configuracao\pipeline.json`. O mesmo arquivo define o percentual de blocos
processado; o experimento original utiliza `1.0%`.

## Fluxo

```text
Parquets Ethereum
  -> auditoria e engenharia de features
  -> recortes temporais
  -> pré-processamento e drift
  -> Isolation Forest e Autoencoder
  -> análise integrada dos escores
  -> candidatos estruturais em C#
  -> enriquecimento RPC
  -> validação e adjudicação semântica
  -> consolidação e avaliação final
```

O Label Cloud constitui uma comparação externa opcional. Seus dados não entram
nas features nem no treinamento dos modelos.

## Entradas principais

| Entrada | Caminho |
|---|---|
| Configuração dos Parquets | `.\configuracao\pipeline.json` |
| Código-fonte | `.` |
| CSV histórico do Label Cloud | `.\04-pos-processamento\resultados-labelcloud\etherscan_labl_cloud_202609172241.csv` |

## Etapas

| Ordem | Etapa | Diretório |
|---:|---|---|
| 1 | Coleta dos Parquets | `.\01-coleta-dados` |
| 2 | Remoção opcional de `input` | `.\02-pre-processamento` |
| 3 | Features, recortes, scaler e drift | `.\03-machine-learning\01-features` |
| 4 | Isolation Forest | `.\03-machine-learning\03-if` |
| 5 | Autoencoder | `.\03-machine-learning\02-ae` |
| 6 | Pós-processamento e fontes externas | `.\04-pos-processamento` |
| 7 | Análise e gráficos | `.\05-analise-resultados` |

## Documentação

- Ordem de execução: [GUIA_REPRODUCAO_SEGURA.md](./GUIA_REPRODUCAO_SEGURA.md)
- Códigos e resultados: [INVENTARIO_CODIGOS.md](./INVENTARIO_CODIGOS.md)
- Estrutura de pastas: [ORGANIZACAO_REPOSITORIO.md](./ORGANIZACAO_REPOSITORIO.md)

Cada diretório possui um README com finalidade, comando e descrição dos
artefatos gerados pela respectiva etapa.

## Execução

Os scripts Python usam UV e dependências PEP 723:

```powershell
uv run ".\caminho\script.py" [parâmetros]
```

Antes da primeira execução, ajuste `dados.root_template` em
`.\configuracao\pipeline.json` e confira `amostragem.percentual_blocos`. O
marcador `{year}` é obrigatório. Os
parâmetros `--root-template` e `--raw-root-template` permanecem disponíveis
somente para substituições pontuais.

O projeto C# exige .NET 8. A validação semântica exige um endpoint Ethereum
JSON-RPC informado apenas na sessão:

```powershell
$env:ETH_RPC_URL = Read-Host "Endpoint RPC Ethereum"
```

Para preservar os resultados atuais, cada reprodução deve usar uma nova raiz
de saída, conforme o guia de execução.
