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

## Validação mínima

Antes de iniciar o pipeline, confirme:

- 366 partições diárias em 2024;
- 365 partições diárias em 2025;
- presença de arquivos Parquet em cada partição;
- leitura dos arquivos pelo DuckDB;
- unicidade de `hash` dentro de cada mês.

Esta pasta documenta a entrada externa do experimento. A remoção opcional da
coluna `input` é executada na etapa `.\02-pre-processamento`.
