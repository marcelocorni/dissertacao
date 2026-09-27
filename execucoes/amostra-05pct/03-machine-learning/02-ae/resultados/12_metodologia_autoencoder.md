# Metodologia do Autoencoder

Os modelos foram ajustados exclusivamente em `treino_2024_pos_dencun`. Dentro desse recorte,
90% das linhas treinam os pesos e 10% formam uma validação interna determinística
para early stopping. Essa divisão interna não é a validação temporal de 2025.

A arquitetura simétrica é `entrada -> 16 -> 8 -> 4 -> 8 -> 16 ->
saída`, com LeakyReLU e saída linear. O treinamento minimiza MSE com AdamW.
O escore por transação é o erro quadrático médio de reconstrução.

Os limiares Q99, Q99,5 e Q99,9 são obtidos somente em
`validacao_2025_pre_pectra`. O Q99,5 é a análise principal. As cinco configurações são
idênticas às do Isolation Forest e foram congeladas antes da avaliação.

Dispositivo usado: `cuda`; GPU: `NVIDIA GeForce RTX 3060`.
Resultados do teste não devem orientar novo ajuste de arquitetura, features ou limiar.
