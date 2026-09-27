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

- Autoencoder `c_sem_taxas_absolutas`: ROC-AUC
  0.6841 (IC95%
  0.6613–0.7048),
  PR-AUC 0.7537, F1 em
  `q990` 0.0206
  e superioridade pareada
  0.8036.
- Isolation Forest `b_completa`: ROC-AUC
  0.7598 (IC95%
  0.7421–0.7791),
  PR-AUC 0.8430, F1 em
  `q990` 0.1093
  e superioridade pareada
  0.7970.

## Análise de sensibilidade no teste final

- Autoencoder `c_sem_taxas_absolutas`: ROC-AUC
  0.6403, PR-AUC
  0.7041, F1 em
  `q990` 0.0248
  e superioridade pareada
  0.7440.
- Isolation Forest `b_completa`: ROC-AUC
  0.7247, PR-AUC
  0.8122, F1 em
  `q990` 0.0937
  e superioridade pareada
  0.7340.

## Resultado estrito por tipo de ataque

- Em `insertion`, o AE obteve ROC-AUC
  0.7161 e
  superioridade pareada
  0.8256;
  o IF obteve ROC-AUC
  0.8004 e
  superioridade pareada
  0.8205.
- Em `displacement`, com
  52
  pares no teste, o AE obteve ROC-AUC
  0.1646 e o IF
  0.0592. A
  superioridade pareada foi
  0.0577
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
