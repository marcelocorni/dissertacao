# Remoção da coluna `input`

## Objetivo

Reduzir o espaço ocupado pelos Parquets brutos removendo a coluna `input`. A
validação semântica recupera posteriormente a calldata apenas para os candidatos
selecionados, por JSON-RPC.

Código:
`.\02-pre-processamento\01-remover_coluna_input_parquet.py`.

## Execução

```powershell
uv run ".\02-pre-processamento\01-remover_coluna_input_parquet.py" `
  --years 2024 2025 `
  --temp-dir ".\.tmp\duckdb"
```

Para apenas listar os arquivos:

```powershell
uv run ".\02-pre-processamento\01-remover_coluna_input_parquet.py" `
  --years 2024 2025 --dry-run
```

## Funcionamento

Para cada Parquet, o script:

1. verifica se `input` existe;
2. grava um Parquet temporário sem a coluna;
3. compara esquema e número de linhas;
4. substitui o arquivo original somente após a validação;
5. ignora arquivos que já não possuem `input`.

## Resultado

Os próprios arquivos localizados por `dados.root_template` em
`.\configuracao\pipeline.json` são atualizados. Não há outro diretório de
saída.
