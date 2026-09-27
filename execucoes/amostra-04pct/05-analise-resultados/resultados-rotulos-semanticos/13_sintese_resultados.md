# Síntese da avaliação com rótulos semânticos

## Desenho

A avaliação usa somente transações com papel conhecido em eventos adjudicados.
Atacantes são positivos e vítimas do mesmo conjunto de eventos são negativos
contextuais. Transações fora desses papéis não são consideradas negativas.

A política estrita inclui somente eventos confirmados. A política de
sensibilidade inclui eventos confirmados ou prováveis. Transações conflitantes
são excluídas da avaliação correspondente.

As configurações abaixo foram escolhidas pela maior PR-AUC em
`validacao_2025_pre_pectra` e aplicadas sem nova seleção a
`teste_final_2025`.

## Resultado estrito no teste final

- Autoencoder `b_sem_taxas_absolutas`: ROC-AUC
  0.7689 (IC95%
  0.7560–0.7817),
  PR-AUC 0.8367, F1 em
  `q990` 0.0199
  e superioridade pareada
  0.8571.
- Isolation Forest `b_completa`: ROC-AUC
  0.7500 (IC95%
  0.7381–0.7624),
  PR-AUC 0.8408, F1 em
  `q990` 0.1633
  e superioridade pareada
  0.7811.

## Análise de sensibilidade no teste final

- Autoencoder `b_sem_taxas_absolutas`: ROC-AUC
  0.7342, PR-AUC
  0.7986, F1 em
  `q990` 0.0183
  e superioridade pareada
  0.8052.
- Isolation Forest `b_completa`: ROC-AUC
  0.7155, PR-AUC
  0.8086, F1 em
  `q990` 0.1409
  e superioridade pareada
  0.7234.

## Resultado estrito por tipo de ataque

- Em `insertion`, o AE obteve ROC-AUC
  0.7878 e
  superioridade pareada
  0.8783;
  o IF obteve ROC-AUC
  0.7857 e
  superioridade pareada
  0.8011.
- Em `displacement`, com
  92
  pares no teste, o AE obteve ROC-AUC
  0.2023 e o IF
  0.0512. A
  superioridade pareada foi
  0.0326
  nos dois modelos, indicando ordenação inversa nessa subpopulação pequena.

## Interpretação

ROC-AUC e PR-AUC medem a ordenação entre atacantes e vítimas dentro da
população condicionada aos candidatos estruturais. As métricas em q990, q995 e
q999 avaliam os limiares calibrados anteriormente, sem reajuste pelos rótulos.
A superioridade pareada mede a proporção de pares do mesmo evento nos quais o
atacante recebeu escore maior que a vítima.

Esses resultados não estimam prevalência, precisão ou recall em toda a rede.
Os rótulos cobrem apenas eventos encontrados pelas regras C# e adjudicados pelo
pipeline semântico. Essa seleção condicionada deve acompanhar qualquer uso das
métricas na dissertação.

Nos dois cenários do Autoencoder, nenhum dos limiares q990, q995 ou q999 gerou
predições positivas na validação. O q990 é exibido apenas como o menos
conservador entre os três, sem caracterizar um limiar operacional válido. A
discriminação contínua e a superioridade pareada permanecem informativas.

O resultado agregado não deve ocultar a heterogeneidade entre regras. A grande
maioria dos papéis estritos do teste pertence a `insertion`; a evidência para
`displacement` é pequena e apresenta direção oposta à esperada.
