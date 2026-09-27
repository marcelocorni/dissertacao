# Síntese da avaliação hipotética com Label Cloud

## Hipótese do experimento

Hipótese experimental: endereços MEV Bot do Label Cloud são positivos; todos os endereços ausentes são negativos. Esta hipótese viabiliza as métricas supervisionadas abaixo, mas não
transforma a ausência de tag em evidência empírica de ausência de front-running.

## População avaliada

- sete recortes cronológicos;
- cinco configurações de features;
- Autoencoder e Isolation Forest;
- avaliação sobre todos os 9.494.450 registros amostrados;
- prevalência hipotética no teste final: 1.7514%.

## Principais resultados

- melhor ROC-AUC na validação: **IF:b_completa**, 0.9033;
- melhor ROC-AUC no teste final: **IF:b_completa**, 0.9489;
- melhor PR-AUC no teste final: **IF:c_completa**, 0.3031;
- melhor F1 no teste final com limiar q995: **IF:c_completa**, 0.3556
  (precisão 0.4567, recall 0.2912,
  especificidade 0.9938, MCC 0.3558);
- melhor F1 entre os três limiares predefinidos: **IF:c_completa** em
  **q990**, 0.3741
  (precisão 0.3529,
  recall 0.3980,
  MCC 0.3629);
- maior lift no top 1% do teste: **AE:b_sem_taxas_absolutas**, 28.93×
  (precisão 0.5066, recall 0.2893).

## Leitura metodológica

ROC-AUC mede ordenação global e pode parecer elevada em bases desbalanceadas.
PR-AUC, precisão, recall, F1, MCC e lift devem receber maior peso na discussão.
Os limiares q990, q995 e q999 foram calibrados anteriormente sem esses rótulos;
portanto, as matrizes de confusão avaliam a política original e não um limiar
otimizado retrospectivamente para o Label Cloud.

Os resultados do teste final devem permanecer como avaliação final. A validação
pode orientar a escolha de configuração, mas o teste não deve ser usado para
reajustar o modelo ou o limiar.
