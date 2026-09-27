# Guia de execução do experimento

O pipeline é controlado por `configuracao/pipeline.json` e executado por
`executar_pipeline.py`. Não é necessário informar manualmente as pastas de
cada etapa.

## 1. Definir a execução

Altere estes dois campos antes de iniciar uma nova amostragem:

```json
"amostragem": { "percentual_blocos": 2.0 },
"execucao": { "id": "amostra-02pct" }
```

Não reutilize um identificador. A execução original de `1%` está preservada em
`execucoes/amostra-01pct`.

## 2. Conferir o plano

```powershell
uv run ".\executar_pipeline.py" --listar
uv run ".\executar_pipeline.py" --dry-run
```

`--dry-run` mostra todos os comandos e caminhos sem criar ou modificar
arquivos.

## 3. Executar o núcleo de aprendizado de máquina

```powershell
uv run ".\executar_pipeline.py" --ate analise_ae_if
```

Essa faixa gera features, recortes temporais, pré-processamento, drift,
Isolation Forest, Autoencoder e análise integrada.

## 4. Executar Label Cloud e candidatos C#

```powershell
uv run ".\executar_pipeline.py" `
  --de labelcloud_preparar `
  --ate auditoria_csharp
```

O Label Cloud permanece uma comparação externa e não entra no treinamento.

Para uma nova execução completa que deve parar antes do RPC, use um arquivo de
configuração próprio com outro `execucao.id` e o percentual desejado. Por
exemplo, copie `configuracao/pipeline.json` para
`configuracao/pipeline-05pct.json`, ajuste o percentual para `5.0` e o ID para
`amostra-05pct`, e execute:

```powershell
uv run ".\executar_pipeline.py" --config ".\configuracao\pipeline-05pct.json" --ate auditoria_csharp
```

O comando percorre as etapas anteriores ainda pendentes e para após auditar
as saídas C#. Quando quiser enriquecer por RPC, mantenha o arquivo idêntico,
defina os endpoints disponíveis e retome com:

```powershell
uv run ".\executar_pipeline.py" --config ".\configuracao\pipeline-05pct.json" --de auditoria_rpc
```

## 5. Executar validação semântica e avaliação final

Defina somente na sessão atual os endpoints que estiverem disponíveis:

```powershell
$env:ETH_RPC_ALCHEMY = Read-Host "Alchemy"
$env:ETH_RPC_INFURA = Read-Host "Infura"
$env:ETH_RPC_DRPC = Read-Host "dRPC"
$env:ETH_RPC_ANKR = Read-Host "Ankr"
$env:ETH_RPC_QUICKNODE = Read-Host "QuickNode"
uv run ".\executar_pipeline.py" --de auditoria_rpc
```

Os provedores são testados antes da execução. Somente os aprovados recebem
janelas; provedores ausentes ou incompatíveis são ignorados. Com apenas um
endpoint aprovado, o processamento continua válido, porém sequencial.

O cache fica em `cache-compartilhado` e é reutilizado entre execuções. Os
resultados permanecem separados em `execucoes/<id>`.

## Retomada

Repita o mesmo comando com a mesma configuração. O executor confere o marcador
de cada etapa concluída e a ignora. Se a configuração tiver sido modificada
depois do início, a retomada é recusada para preservar a comparabilidade.

O estado detalhado fica em:

```text
execucoes/<id>/00_manifest_multiexecucao.json
```

Ao finalizar a última etapa, um resumo pequeno e versionável é criado em
`historico-execucoes/<id>.json`.
