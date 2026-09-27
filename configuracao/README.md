# Configuração central

O arquivo `.\configuracao\pipeline.json` informa onde estão os Parquets
externos usados pelo pipeline e define a fração de blocos processada:

```json
{
  "dados": {
    "root_template": "F:\\ethereum-{year}"
  },
  "amostragem": { "percentual_blocos": 1.0 },
  "execucao": {
    "id": "amostra-01pct",
    "raiz": "execucoes",
    "historico_raiz": "historico-execucoes",
    "proteger_resultados_existentes": true
  },
  "cache_compartilhado": {
    "raiz": "cache-compartilhado",
    "rpc_blocks": "blocks",
    "rpc_transactions": "rpc",
    "token_metadata": "token_metadata"
  },
  "recursos": {
    "memory_limit": "16GB",
    "threads": 8,
    "temp_dir": ".tmp/duckdb"
  }
}
```

O marcador `{year}` é obrigatório e permite localizar os diretórios de 2024 e
2025. Para usar outro disco ou estrutura, altere somente esse valor.

`amostragem.percentual_blocos` vale `1.0` na execução original. A seleção é
determinística e mantém todas as transações dos blocos escolhidos. Exemplos:

| Percentual | Regra equivalente |
|---:|---|
| `1.0` | `block_number % 100 = 0` |
| `5.0` | `block_number % 20 = 0` |
| `10.0` | `block_number % 10 = 0` |
| `100.0` | todos os blocos |

Para preservar exatamente essa regra, o percentual deve ser representável por
`100/N`. Quanto maior o percentual, maiores serão o tempo de execução, o uso de
memória e o espaço ocupado pelos resultados. Os artefatos já existentes não são
alterados ao editar a configuração; a mudança vale para novas execuções.

Os scripts Python e o projeto C# carregam o arquivo automaticamente. Em uma
execução excepcional, `--config` seleciona outro arquivo;
`--root-template` (Python) ou `--raw-root-template` (C#) substitui apenas o
valor daquela execução.

Nos scripts de features, `--block-modulus` continua disponível como substituição
pontual do percentual configurado.

Credenciais e URLs RPC não pertencem a este arquivo.

## Provedores RPC

`rpc_providers.json` contém somente nomes de variáveis de ambiente e limites
operacionais. `blocks_per_batch`, `requests_per_second` e
`token_calls_per_second` controlam a pressão aplicada a cada serviço.
`initial_effective_blocks_per_second` é a estimativa inicial usada para ordenar
os provedores; o escalonador a atualiza com o desempenho medido durante a
execução. Não há vínculo fixo entre provedor e janela.

As maiores cargas remanescentes são entregues primeiro aos provedores mais
rápidos. Um provedor liberado assume a próxima janela da fila. Falhas nas etapas
RPC ou de metadados reenfileiram a janela para outro provedor, preservando o
cache já obtido. As URLs completas continuam existindo somente nas variáveis de
ambiente da sessão e nunca são gravadas nos manifestos.

## Execuções independentes

`execucao.id` identifica uma execução completa. Depois que ela começar, não
altere a configuração: sua impressão digital é gravada em
`execucoes/<id>/00_manifest_multiexecucao.json`. Para outra porcentagem, use
outro identificador, por exemplo:

```json
"amostragem": { "percentual_blocos": 2.0 },
"execucao": { "id": "amostra-02pct" }
```

O exemplo mostra somente os campos alterados; os demais devem permanecer no
arquivo. O executor recusa diretórios preexistentes sem manifesto válido e
retoma somente etapas concluídas com a mesma configuração.

Os percentuais continuam sujeitos à regra determinística `100/N`, em que `N`
é um inteiro positivo. No intervalo inteiro entre `1%` e `10%`, somente `1%`,
`2%`, `4%`, `5%` e `10%` são válidos. Os demais produziriam um módulo
fracionário e são recusados antes do processamento. A tabela ampliada e a
justificativa metodológica estão no README da raiz.
