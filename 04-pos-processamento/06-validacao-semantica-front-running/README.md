# Validação semântica de front-running

## Objetivo

Enriquecer os candidatos C# com transaction, calldata, receipt, logs e
metadados ERC-20; validar as regras; gerar o dossiê; adjudicar os eventos; e
consolidar os rótulos por transação.

## Execução concorrente das sete janelas

Os provedores são definidos em `configuracao/rpc_providers.json`. O arquivo
contém somente nomes de variáveis, limites e preferências de distribuição; as
URLs privadas nunca são gravadas no repositório ou nos manifestos.
Um [cache RPC opcional usado até a execução de 5%](../../README.md#cache-rpc-opcional)
está disponível separadamente dos Parquets e dos resultados versionados.

Defina apenas as variáveis correspondentes aos provedores disponíveis:

```powershell
$env:ETH_RPC_ALCHEMY = Read-Host "Alchemy"
$env:ETH_RPC_INFURA = Read-Host "Infura"
$env:ETH_RPC_DRPC = Read-Host "dRPC"
$env:ETH_RPC_ANKR = Read-Host "Ankr"
$env:ETH_RPC_QUICKNODE = Read-Host "QuickNode"
$env:ETH_RPC_BLASTAPI_PUBLIC = "https://eth-mainnet.public.blastapi.io"
$env:ETH_RPC_MEWAPI_PUBLIC = "https://nodes.mewapi.io/rpc/eth"
```

As variáveis existem somente no processo atual do PowerShell. Portanto, elas
devem ser definidas na mesma janela em que o pipeline será iniciado. O arquivo
de configuração define em `minimum_approved_providers` o mínimo de provedores
aprovados; o valor atual é `1`. Retire da sessão as variáveis dos provedores
com cota esgotada antes da homologação, para que não recebam novas janelas.
Antes da coleta, execute a partir da raiz do
repositório:

```powershell
uv run ".\executar_pipeline.py" --de auditoria_rpc
```

`15_testar_provedores_rpc.py` homologa `eth_chainId`, batch JSON-RPC, bloco
histórico, `eth_getBlockReceipts` e `eth_call`. Somente provedores aprovados são
entregues ao executor paralelo.

`16_executar_pipeline_janelas_paralelo.py` mantém uma fila dinâmica. As janelas
com maior quantidade de blocos ainda ausentes são entregues primeiro aos
provedores com maior velocidade efetiva. Quando um trabalhador termina, ele
recebe a próxima janela pendente. Se uma coleta RPC ou consulta de metadados
falhar, a janela é reenfileirada para outro provedor, que reutiliza o cache e
consulta somente os blocos ainda ausentes. O terminal apresenta, a cada 30
segundos, janela ativa, progresso, velocidade, ETA e fila restante.

Não existe associação fixa entre janela e provedor. O arquivo
`configuracao/rpc_providers.json` registra limites operacionais e uma estimativa
inicial de desempenho. O escalonador atualiza essa estimativa a partir dos
tempos observados e registra todas as atribuições e recuperações no manifesto
`00_manifest_pipeline_paralelo.json`.

## Execução sequencial de diagnóstico

O executor anterior permanece disponível para testes com um único endpoint:

```powershell
$env:ETH_RPC_URL = Read-Host "Endpoint RPC Ethereum"

uv run ".\04-pos-processamento\06-validacao-semantica-front-running\11_executar_pipeline_janelas.py" `
  --csharp-root ".\04-pos-processamento\05-rotulador-front-running-csharp\resultados-v3\sampled\independent_sampled" `
  --results-root ".\04-pos-processamento\06-validacao-semantica-front-running\resultados" `
  --cache-root ".\04-pos-processamento\06-validacao-semantica-front-running\cache" `
  --audit-manifest ".\04-pos-processamento\06-validacao-semantica-front-running\resultados\auditoria-csharp\02_manifest_auditoria_csharp.json" `
  --regression-manifest ".\04-pos-processamento\06-validacao-semantica-front-running\resultados-blocos\transicao_dencun_2024\04_regressao_rpc.json"
```

O executor realiza, nesta ordem:

1. enriquecimento RPC por bloco;
2. validação semântica;
3. geração do dossiê;
4. consulta de metadados dos tokens;
5. adjudicação assistida;
6. validação independente da adjudicação.

## Artefatos por janela

Diretório:
`.\04-pos-processamento\06-validacao-semantica-front-running\resultados\<janela>`.

| Artefato | Conteúdo |
|---|---|
| `01_transacoes_rpc.parquet` | transaction, calldata, receipt e logs normalizados |
| `02_erros_rpc.json` | falhas de enriquecimento |
| `03_manifest_rpc.json` | cobertura e provedor |
| `04_eventos_validados.parquet` | decisão semântica por evento |
| `05_papeis_confirmados.parquet` | hashes e papéis confirmados |
| `06_resumo_validacao.csv` | contagens por regra e decisão |
| `07_manifest_validacao.json` | regras e fontes da validação |
| `08_dossie_auditoria.csv/.parquet` | evidências completas por evento |
| `09_fluxos_tokens_atacante.csv/.parquet` | transferências ERC-20 do atacante |
| `10_resumo_dossie.csv` | resumo do dossiê |
| `11_manifest_dossie.json` | proveniência do dossiê |
| `12_metadados_tokens.csv/.parquet` | símbolo, nome e decimais dos tokens |
| `13_erros_metadados_tokens.json` | metadados incompletos |
| `14_manifest_metadados_tokens.json` | cobertura dos tokens |
| `15_decisoes_dossie.csv` | decisão, tipo, qualidade e justificativa |
| `16_manifest_adjudicacao_assistida.json` | política e contagens da adjudicação |
| `17_validacao_adjudicacoes.csv` | verificação evento a evento |
| `18_resumo_validacao_adjudicacoes.csv` | resumo da verificação |
| `19_manifest_validacao_adjudicacoes.json` | aprovação da janela |

## Consolidação

```powershell
uv run ".\04-pos-processamento\06-validacao-semantica-front-running\12_consolidar_rotulos_semanticos.py" `
  --results-root ".\04-pos-processamento\06-validacao-semantica-front-running\resultados" `
  --output-dir ".\04-pos-processamento\06-validacao-semantica-front-running\resultados\consolidado"
```

Resultados em
`.\04-pos-processamento\06-validacao-semantica-front-running\resultados\consolidado`:

| Artefato | Conteúdo |
|---|---|
| `01_eventos_adjudicados.csv/.parquet` | decisão final por evento |
| `02_papeis_evento.csv/.parquet` | atacante e vítima por evento |
| `03_rotulos_transacao.csv/.parquet` | rótulo agregado por hash e janela |
| `04_conflitos_rotulos.csv` | hashes com papéis incompatíveis |
| `05_resumo_consolidacao.csv` | totais por janela |
| `06_manifest_consolidacao.json` | definições e proveniência |
| `07_metodologia_rotulos_semanticos.md` | interpretação de `strict` e `sensitivity` |
| `08_manifest_execucao_adjudicacao_global.json` | estado das sete janelas |

`strict` usa apenas eventos confirmados. `sensitivity` acrescenta eventos de
displacement prováveis.

## Interface de consulta

```powershell
$env:SEMANTIC_RESULTS_ROOT = ".\04-pos-processamento\06-validacao-semantica-front-running\resultados"
uv run --with streamlit --with duckdb streamlit run `
  ".\04-pos-processamento\06-validacao-semantica-front-running\05_interface_dossie_semantico.py"
```

A interface é somente leitura.

## Códigos auxiliares

| Código | Função |
|---|---|
| `01_enriquecer_fila_rpc.py` | coleta RPC por transação para regressão |
| `02_validar_semantica.py` | valida insertion e displacement |
| `03_gerar_dossie_auditoria.py` | organiza evidências e fluxos |
| `04_enriquecer_tokens_rpc.py` | consulta metadados ERC-20 |
| `06_adjudicar_dossie_assistido.py` | produz decisões determinísticas |
| `08_enriquecer_fila_por_bloco.py` | coleta RPC operacional |
| `09_validar_adjudicacoes_assistidas.py` | recalcula e verifica as decisões |
| `10_validar_regressao_rpc.py` | compara os dois coletores |
| `13_executar_adjudicacao_global.py` | regenera e consolida as adjudicações |
| `14_reconstruir_cache_rpc.py` | recompõe cache a partir dos Parquets enriquecidos |
| `15_testar_provedores_rpc.py` | homologa provedores sem registrar URLs ou chaves |
| `16_executar_pipeline_janelas_paralelo.py` | distribui janelas entre provedores concorrentes |
