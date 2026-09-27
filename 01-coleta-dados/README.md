# Coleta dos dados Ethereum

## Objetivo

Sincronizar os Parquets públicos de transações Ethereum que alimentam todo o
experimento.

## Localização dos dados

Os Parquets são externos ao repositório. Sua localização é definida por
`dados.root_template` em `.\configuracao\pipeline.json`. O valor deve conter o
marcador obrigatório `{year}`, substituído automaticamente pelo ano processado.

Exemplo de configuração:

```json
{
  "dados": {
    "root_template": "F:\\ethereum-{year}"
  }
}
```

Não é necessário repetir esse caminho nos comandos dos scripts. Para armazenar
os dados em outra unidade, altere somente o valor no arquivo de configuração.

## Estrutura exigida

Depois de substituir `{year}`, cada diretório anual deve conter partições
diárias no seguinte formato:

```text
<diretório-2024>\date=2024-01-01\*.parquet
<diretório-2025>\date=2025-01-01\*.parquet
```

## Download com AWS CLI

Instale o AWS CLI v2 e execute os comandos abaixo no PowerShell, a partir da
raiz do repositório. O bucket público não exige credenciais; o destino de cada
ano é lido de `configuracao/pipeline.json`. Use diretórios de destino vazios:
não sincronize sobre bases das quais a coluna `input` já tenha sido removida.

```powershell
aws --version
aws s3 ls "s3://aws-public-blockchain/v1.0/eth/transactions/date=2024-01-01/" `
  --no-sign-request --region us-east-2

$configuracaoColeta = Get-Content ".\configuracao\pipeline.json" -Raw | ConvertFrom-Json
$origemColeta = "s3://aws-public-blockchain/v1.0/eth/transactions/"

foreach ($anoColeta in 2024, 2025) {
    $destinoColeta = $configuracaoColeta.dados.root_template.Replace("{year}", [string]$anoColeta)
    aws s3 sync $origemColeta $destinoColeta `
      --exclude "*" --include "date=${anoColeta}-*/*.parquet" `
      --no-sign-request --region us-east-2
    if ($LASTEXITCODE -ne 0) { throw "Falha no download do ano $anoColeta" }
}
```

O filtro baixa somente as partições de 2024 e 2025, mantendo a estrutura
`date=AAAA-MM-DD` dentro de cada diretório anual. `sync` pode ser repetido
para retomar um download interrompido, desde que o destino ainda contenha os
Parquets originais. O comando não usa `--delete` e não envia dados ao S3.

Antes de transferir tudo, acrescente `--dryrun` ao comando `aws s3 sync` para
conferir a seleção de objetos. O snapshot medido para este experimento soma
aproximadamente 392 GiB nos dois anos; reserve espaço adicional para os
resultados intermediários.

Fontes: [dataset público da AWS](https://registry.opendata.aws/aws-public-blockchain/)
e [referência do `aws s3 sync`](https://docs.aws.amazon.com/cli/latest/reference/s3/sync.html).

## Validação mínima

Antes de iniciar o pipeline, confirme:

- 366 partições diárias em 2024;
- 365 partições diárias em 2025;
- presença de arquivos Parquet em cada partição;
- leitura dos arquivos pelo DuckDB;
- unicidade de `hash` dentro de cada mês.

Esta pasta documenta a entrada externa do experimento. A remoção opcional da
coluna `input` é executada na etapa `.\02-pre-processamento`.
