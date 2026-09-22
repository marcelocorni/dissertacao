# Metodologia da avaliação semântica

## População e classes

- Fonte: `03_rotulos_transacao.parquet` e `02_papeis_evento.parquet`.
- Positivo: `attacker_front` ou `attacker_back`.
- Negativo contextual: `victim` em evento positivo da mesma política.
- Não rotulado: excluído; não é presumido negativo.
- Conflito: excluído da política em que ocorre.

## Políticas

- Estrita: Somente eventos adjudicados como confirmado.
- Sensibilidade: Eventos adjudicados como confirmado ou provável.

## Modelos e separação temporal

São avaliadas cinco configurações de features para Autoencoder e Isolation
Forest em sete janelas. A escolha apresentada como resultado final usa a maior
PR-AUC na validação pré-Pectra, separadamente por método e política. O teste
final não participa dessa seleção. Após escolher a configuração, o limiar entre
q990, q995 e q999 é escolhido pelo maior F1 na mesma validação e aplicado sem
reajuste ao teste final.

## Métricas

- ROC-AUC e PR-AUC sobre escores contínuos;
- IC95% por bootstrap estratificado com 500 repetições;
- precisão, recall, especificidade, F1, acurácia balanceada e MCC nos limiares
  q990, q995 e q999 previamente calibrados;
- precisão, recall e lift nos top 1%, 5% e 10% dos escores;
- superioridade pareada atacante–vítima dentro do mesmo evento.

## Limitação central

A referência semântica é condicionada aos candidatos produzidos pelas regras
estruturais C#. Portanto, as métricas caracterizam a capacidade de ordenar e
separar papéis dentro dessa população e não o desempenho supervisionado sobre
todas as transações Ethereum.
