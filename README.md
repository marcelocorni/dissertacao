# Pipeline de detecção de anomalias e front-running em Ethereum

Este repositório implementa o experimento completo com transações Ethereum em
Parquet, Autoencoder, Isolation Forest, regras estruturais de front-running e
validação semântica por dados on-chain.

Todos os caminhos desta documentação são relativos à raiz do repositório. A localização
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
- Painel comparativo de resultados: [06-painel-resultados/README.md](./06-painel-resultados/README.md)

Cada diretório possui um README com finalidade, comando e descrição dos
artefatos gerados pela respectiva etapa.

## Execução

Cada experimento usa um identificador exclusivo definido em
`configuracao/pipeline.json`. Todos os resultados ficam em
`execucoes/<id>`; o cache RPC fica em `cache-compartilhado` e pode ser
reutilizado pelas demais porcentagens.

### Cache RPC opcional

O [arquivo `cache-compartilhado.7z`](https://mega.nz/file/38ETGQLZ#R9Ad46DRLMitviZtZNs-9PR1ewUaBuVzviamjwP7_YQ)
contém o cache compartilhado usado nas execuções realizadas até `5%`.
Ele não faz parte do Git e não substitui os Parquets Ethereum nem os
resultados em `execucoes/<id>`. Seu uso é opcional: sem ele, o enriquecimento
precisa consultar os provedores RPC novamente.

Baixe o arquivo e extraia-o **na raiz do repositório**, em uma instalação que
ainda não tenha `cache-compartilhado/`. O arquivo compactado já contém essa
pasta no nível superior. Antes da extração, confira o SHA-256 do download:

```powershell
Get-FileHash ".\cache-compartilhado.7z" -Algorithm SHA256
```

Valor esperado: `6167D910E0DA6DA14AA9D030F995385E70528C7AEC4EB052177C96C25F24432E`.
Não extraia sobre um cache existente sem antes preservá-lo ou conferir os
arquivos que seriam substituídos.

Depois de ajustar a configuração, valide o plano sem gravar arquivos:

```powershell
uv run ".\executar_pipeline.py" --listar
uv run ".\executar_pipeline.py" --dry-run
```

Para executar ou retomar todas as etapas:

```powershell
uv run ".\executar_pipeline.py"
```

Para executar uma amostra até a etapa anterior ao enriquecimento RPC, crie um
arquivo de configuração e um identificador exclusivos. O exemplo usa `4%`:

```powershell
Copy-Item ".\configuracao\pipeline.json" ".\configuracao\pipeline-04pct.json"
notepad ".\configuracao\pipeline-04pct.json"
```

No arquivo copiado, ajuste `amostragem.percentual_blocos` para `4.0` e
`execucao.id` para `amostra-04pct`. Para `5%`, use `5.0`, `amostra-05pct` e
outro arquivo de configuração. Confira o plano e execute até a auditoria das
saídas C#, última etapa anterior ao RPC:

```powershell
uv run ".\executar_pipeline.py" --config ".\configuracao\pipeline-04pct.json" --dry-run --ate auditoria_csharp
uv run ".\executar_pipeline.py" --config ".\configuracao\pipeline-04pct.json" --ate auditoria_csharp
```

Para executar as etapas RPC e seguintes, defina os endpoints na sessão do
PowerShell e retome com a mesma configuração:

```powershell
uv run ".\executar_pipeline.py" --config ".\configuracao\pipeline-04pct.json" --de auditoria_rpc
```

Não altere a configuração após iniciar uma execução: o executor verifica sua
impressão digital na retomada. A execução até `auditoria_csharp` gera features,
modelos, análises, candidatos C# e sua auditoria, sem chamadas de
enriquecimento RPC. Neste exemplo, os resultados ficam em
`execucoes/amostra-04pct`.

Etapas concluídas são verificadas e ignoradas automaticamente. Uma execução
existente nunca é adotada apenas porque possui arquivos: ela precisa ter o
manifesto do executor e a mesma impressão digital da configuração.

O projeto C# exige .NET 8. A validação semântica exige um endpoint Ethereum
JSON-RPC informado apenas na sessão. Para execução concorrente, os provedores
e limites ficam em `configuracao/rpc_providers.json`, enquanto as URLs privadas
ficam exclusivamente nas variáveis `ETH_RPC_*`.

```powershell
uv run ".\executar_pipeline.py" --de auditoria_rpc
```

Para cada percentual, defina `execucao.id` e
`amostragem.percentual_blocos` antes da primeira execução.

## Percentuais permitidos

A amostragem seleciona blocos completos pela regra determinística:

```text
block_number % N = 0
percentual = 100 / N
```

Consequentemente, `N` precisa ser um número inteiro positivo. Não existe uma
lista finita de percentuais, mas todo valor configurado deve ser representável
como `100/N`. Alguns valores usuais são:

| Percentual | Módulo `N` | Regra aplicada |
|---:|---:|---|
| `0.1%` | 1000 | `block_number % 1000 = 0` |
| `0.2%` | 500 | `block_number % 500 = 0` |
| `0.25%` | 400 | `block_number % 400 = 0` |
| `0.5%` | 200 | `block_number % 200 = 0` |
| `1%` | 100 | `block_number % 100 = 0` |
| `1.25%` | 80 | `block_number % 80 = 0` |
| `2%` | 50 | `block_number % 50 = 0` |
| `2.5%` | 40 | `block_number % 40 = 0` |
| `4%` | 25 | `block_number % 25 = 0` |
| `5%` | 20 | `block_number % 20 = 0` |
| `10%` | 10 | `block_number % 10 = 0` |
| `20%` | 5 | `block_number % 5 = 0` |
| `25%` | 4 | `block_number % 4 = 0` |
| `50%` | 2 | `block_number % 2 = 0` |
| `100%` | 1 | todos os blocos |

Considerando apenas percentuais inteiros de `1%` a `10%`, são diretamente
permitidos `1%`, `2%`, `4%`, `5%` e `10%`. Os valores `3%`, `6%`, `7%`, `8%`
e `9%` não são aceitos porque `100/percentual` não resulta em um módulo
inteiro.

Essa restrição preserva a regra usada no experimento original, torna a seleção
reprodutível sem sorteio e mantém todas as transações de cada bloco escolhido.
Aceitar qualquer percentual exigiria substituir a regra por outro mecanismo de
amostragem, o que alteraria a metodologia e a comparabilidade com o baseline de
`1%`.
