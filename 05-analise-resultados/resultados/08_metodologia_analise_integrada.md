# Análise integrada AE e Isolation Forest

O conjunto de candidatos usa a união dos casos acima do Q99,5 de cada método.
`ambos` representa consenso; `somente_ae` e `somente_if` preservam resultados
complementares. Os percentis são calculados dentro de cada recorte e configuração,
permitindo combinar escalas diferentes sem misturar os valores brutos.

Os Parquets em `candidatos/` contêm hashes, bloco, data, endereços, tipo da
transação, escores, percentis e grupo de detecção. Esses campos servem apenas
para rastreabilidade e pós-processamento; não participaram do treinamento.

As curvas de sobrevivência do IF exibem a cauda dos escores em escala logarítmica.
AUC-ROC e PR-AUC exigem rótulos externos e são tratadas pelo segundo script.
