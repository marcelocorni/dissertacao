# Síntese da avaliação hipotética com Label Cloud

## Hipótese do experimento

Hipótese experimental: endereços MEV Bot do Label Cloud são positivos; todos os endereços ausentes são negativos. Esta hipótese viabiliza as métricas supervisionadas abaixo, mas não
transforma a ausência de tag em evidência empírica de ausência de front-running.

## População avaliada

- sete recortes cronológicos;
- cinco configurações de features;
- Autoencoder e Isolation Forest;
- avaliação sobre todos os 9.494.450 registros amostrados;
- prevalência hipotética no teste final: 1.7642%.

## Principais resultados

- melhor ROC-AUC na validação: **IF:b_completa**, 0.9125;
- melhor ROC-AUC no teste final: **IF:b_completa**, 0.9547;
- melhor PR-AUC no teste final: **IF:b_completa**, 0.2959;
- melhor F1 no teste final com limiar q995: **IF:c_completa**, 0.3655
  (precisão 0.3996, recall 0.3367,
  especificidade 0.9909, MCC 0.3564);
- melhor F1 entre os três limiares predefinidos: **IF:b_completa** em
  **q990**, 0.3897
  (precisão 0.3698,
  recall 0.4120,
  MCC 0.3788);
- maior lift no top 1% do teste: **IF:c_completa**, 26.15×
  (precisão 0.4614, recall 0.2615).

## Leitura metodológica

ROC-AUC mede ordenação global e pode parecer elevada em bases desbalanceadas.
PR-AUC, precisão, recall, F1, MCC e lift devem receber maior peso na discussão.
Os limiares q990, q995 e q999 foram calibrados anteriormente sem esses rótulos;
portanto, as matrizes de confusão avaliam a política original e não um limiar
otimizado retrospectivamente para o Label Cloud.

Os resultados do teste final devem permanecer como avaliação final. A validação
pode orientar a escolha de configuração, mas o teste não deve ser usado para
reajustar o modelo ou o limiar.
