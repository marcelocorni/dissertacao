# Configuração central

O arquivo `.\configuracao\pipeline.json` informa onde estão os Parquets
externos usados pelo pipeline e define a fração de blocos processada:

```json
{
  "dados": {
    "root_template": "F:\\ethereum-{year}"
  },
  "amostragem": {
    "percentual_blocos": 1.0
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
