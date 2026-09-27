# Metodologia do baseline Isolation Forest

## Ajuste e amostragem

Os cinco modelos foram ajustados exclusivamente em `treino_2024_pos_dencun`. A amostra
determinística usa `hash(row_id)` para cobrir todo o período de treinamento sem
ordenar ou concentrar as observações no início da janela. Cada floresta possui
200 árvores e cada árvore recebe no máximo
8192 transações.

O escore registrado é `-score_samples`: valores maiores indicam maior anomalia.
O parâmetro `contamination=auto` não define a classificação final.

## Configurações

- `a_completa`, `b_completa` e `c_completa` usam os contratos A, B e C;
- `b_sem_taxas_absolutas` e `c_sem_taxas_absolutas` removem a família de taxas
  absolutas, preservando `fee_cap_utilization` e `gas_price_rank_block`;
- as configurações foram congeladas antes da pontuação do teste.

## Limiares

Os limiares Q99, Q99,5 e Q99,9 são calculados exclusivamente em
`validacao_2025_pre_pectra`. O Q99,5 é o ponto principal e os demais formam uma análise
de sensibilidade. Por construção, aproximadamente 0,5% da validação é marcada;
isso não representa uma estimativa da prevalência real de fraude.

## Interpretação

Sem rótulos, taxa de sinalização e concordância entre modelos não medem
precisão ou recall. A escolha de uma configuração deve combinar estabilidade
temporal, inspeção dos casos, comparação externa e, posteriormente, concordância
com o Autoencoder. O teste não pode ser usado para reajustar o modelo.
