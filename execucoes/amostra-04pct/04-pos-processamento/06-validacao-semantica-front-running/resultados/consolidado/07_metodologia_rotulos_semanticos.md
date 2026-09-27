# Consolidação dos rótulos semânticos

A referência **strict** contém somente eventos confirmados. A referência **sensitivity** acrescenta os casos prováveis de displacement. Insertion e displacement confirmados são mantidos separadamente pelo campo `adjudicated_type`. Suppression não recebe rótulo positivo sem evidência de mempool.

Os negativos deste artefato são exclusivamente vítimas contextuais dos eventos positivos. Transações ausentes permanecem não rotuladas; portanto, o artefato não afirma que toda a população restante seja negativa. Quando um mesmo hash recebe papéis incompatíveis, o rótulo agregado fica nulo e o caso é registrado em `04_conflitos_rotulos.csv`.

Conflitos no cenário de sensibilidade podem representar cadeias de displacement provável, nas quais a mesma transação é vítima de uma anterior e candidata a atacante de uma posterior. Esses casos devem ser avaliados no nível do evento ou excluídos de métricas binárias por transação; não se deve escolher um papel por precedência.

A coluna `decision_source` registra `assisted_deterministic` em todas as decisões. A interface é somente leitura e não altera os rótulos produzidos pelo protocolo.
