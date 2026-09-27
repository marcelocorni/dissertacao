# Síntese da avaliação hipotética com Label Cloud

## Hipótese do experimento

Hipótese experimental: endereços MEV Bot do Label Cloud são positivos; todos os endereços ausentes são negativos. Esta hipótese viabiliza as métricas supervisionadas abaixo, mas não
transforma a ausência de tag em evidência empírica de ausência de front-running.

## População avaliada

- sete recortes cronológicos;
- cinco configurações de features;
- Autoencoder e Isolation Forest;
- avaliação sobre todos os 9.494.450 registros amostrados;
- prevalência hipotética no teste final: 1.7740%.

## Principais resultados

- melhor ROC-AUC na validação: **IF:b_completa**, 0.9169;
- melhor ROC-AUC no teste final: **IF:b_completa**, 0.9538;
- melhor PR-AUC no teste final: **IF:c_completa**, 0.2902;
- melhor F1 no teste final com limiar q995: **IF:c_completa**, 0.3464
  (precisão 0.4047, recall 0.3028,
  especificidade 0.9920, MCC 0.3400);
- melhor F1 entre os três limiares predefinidos: **AE:b_sem_taxas_absolutas** em
  **q990**, 0.3886
  (precisão 0.4898,
  recall 0.3220,
  MCC 0.3885);
- maior lift no top 1% do teste: **AE:b_sem_taxas_absolutas**, 28.43×
  (precisão 0.5043, recall 0.2843).

## Leitura metodológica

ROC-AUC mede ordenação global e pode parecer elevada em bases desbalanceadas.
PR-AUC, precisão, recall, F1, MCC e lift devem receber maior peso na discussão.
Os limiares q990, q995 e q999 foram calibrados anteriormente sem esses rótulos;
portanto, as matrizes de confusão avaliam a política original e não um limiar
otimizado retrospectivamente para o Label Cloud.

Os resultados do teste final devem permanecer como avaliação final. A validação
pode orientar a escolha de configuração, mas o teste não deve ser usado para
reajustar o modelo ou o limiar.
