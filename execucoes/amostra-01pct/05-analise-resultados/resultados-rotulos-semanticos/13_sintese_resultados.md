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
  0.7350 (IC95%
  0.7061–0.7644),
  PR-AUC 0.8039, F1 em
  `q990` 0.0000
  e superioridade pareada
  0.8232.
- Isolation Forest `c_completa`: ROC-AUC
  0.7510 (IC95%
  0.7264–0.7764),
  PR-AUC 0.8366, F1 em
  `q990` 0.1079
  e superioridade pareada
  0.7657.

## Análise de sensibilidade no teste final

- Autoencoder `b_sem_taxas_absolutas`: ROC-AUC
  0.6790, PR-AUC
  0.7427, F1 em
  `q990` 0.0000
  e superioridade pareada
  0.7622.
- Isolation Forest `c_completa`: ROC-AUC
  0.7177, PR-AUC
  0.8036, F1 em
  `q990` 0.0947
  e superioridade pareada
  0.7161.

## Resultado estrito por tipo de ataque

- Em `insertion`, o AE obteve ROC-AUC 0.7692 e superioridade pareada 0.8457;
  o IF obteve ROC-AUC 0.7848 e superioridade pareada 0.7867.
- Em `displacement`, com 25 pares no teste, o AE obteve ROC-AUC 0.0192 e o IF
  0.1392. A superioridade pareada foi 0.0000 nos dois modelos, indicando
  ordenação inversa nessa subpopulação pequena.

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
