# Metodologia da análise de drift temporal

## Referência imutável

Todas as comparações usam exclusivamente `treino_2024_pos_dencun` (2024-04-01 a
2024-12-31) como distribuição de referência. Validação, teste e janelas
de mudança de protocolo não ajustam limites, bins ou transformações.

## Métricas

- **PSI:** bins definidos pelos decis do treino. Limiares descritivos: abaixo de
  0,10, drift baixo; de 0,10 a 0,25, moderado; a partir de 0,25, alto.
- **KS aproximado:** maior diferença entre CDFs avaliada em
  101 quantis do treino. Não é calculado p-valor, pois amostras de
  milhões de linhas tornariam diferenças pequenas estatisticamente significativas.
  Limiares descritivos: abaixo de 0,10, baixo; de 0,10 a 0,20, moderado; a partir
  de 0,20, alto.
- **Mediana e IQR:** registram mudanças robustas de localização e dispersão.
- **Zeros:** registra mudanças em features esparsas e zeros estruturais.
- **Extrapolação:** proporção fora do mínimo/máximo e das cercas robustas do treino.

Esses limites são heurísticos de diagnóstico, não critérios automáticos para
eliminar features. Drift pode representar mudança de protocolo ou comportamento
relevante para detecção de anomalias.

## Transações tipo 4

O tipo 4 não entra como indicador nas matrizes treinadas em 2024, pois não existia
no período de ajuste. Ele é recuperado pelos metadados e comparado separadamente
com a distribuição do treino. Isso permite verificar se os escores futuros estão
associados à novidade de protocolo sem fornecer essa informação diretamente ao modelo.

## Uso no pipeline

O teste final continua intocado para seleção de features, hiperparâmetros e limiar.
Este relatório serve para interpretar estabilidade e planejar análises estratificadas;
não autoriza otimização orientada pelos resultados do teste.
