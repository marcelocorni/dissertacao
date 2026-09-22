# Metodologia da auditoria de features

- Período: 2024-04-01 a 2024-12-31.
- Arquivos Parquet: 275.
- Fonte: somente transações on-chain do AWS Public Blockchain Dataset.
- Fear & Greed: não utilizado.
- Binance: não utilizado.
- TagCloud: não utilizado como feature; reservado para comparação final.
- `input`: não utilizado.
- Seleção amostral: todos os registros dos blocos cujo `block_number % 100 = 0`.
- A unidade amostral é o bloco completo, preservando posição, frequência e percentis intrabloco.
- Correlações: no máximo 500,000 registros, semente 20240916.
- Limiar de correlação potencialmente redundante: |r| >= 0.90.
- Pearson mede associação linear; Spearman mede associação monotônica.
- As correlações são diagnósticas e não determinam remoção automática de features.
- O período analisado deve pertencer exclusivamente ao conjunto de treinamento.
