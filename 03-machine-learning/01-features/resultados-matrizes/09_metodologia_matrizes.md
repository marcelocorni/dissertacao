# Metodologia de geração das matrizes

- Dataset lógico: `treino_2024_pos_dencun`.
- Período: 2024-04-01 a 2024-12-31.
- Amostragem: blocos completos com `block_number % 100 = 0`.
- Fonte: somente Parquets on-chain de transações Ethereum.
- Binance, Fear & Greed e TagCloud não participam das features.
- `input` não participa das features.
- O tipo 2 é a categoria de referência da codificação one-hot.
- O tipo 4 é mantido nos metadados, mas não entra nas matrizes treinadas em 2024.
- Valores de taxa não aplicáveis, representados como zero na origem, permanecem
  zero após a transformação.
- `fee_cap_utilization` recebe zero quando o teto de taxa não é aplicável.
- Criação de contrato reconhece `to_address` nulo, vazio ou igual a `0x`.
- Identificadores e endereços ficam em arquivo separado das entradas numéricas.
- `row_id` é a chave de associação entre metadados e as três matrizes.
- Os Parquets usam compressão ZSTD.

## Objetivo dos conjuntos

- Matriz A: baseline exclusivamente transacional.
- Matriz B: baseline acrescida do contexto do bloco; candidata principal.
- Matriz C: conjunto estendido para testar limite de gás e custo total por meio
  de ablação.
