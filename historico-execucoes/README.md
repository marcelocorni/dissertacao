# Histórico de execuções

Este diretório mantém apenas manifestos e resumos comparáveis das execuções.
Os artefatos volumosos ficam em `execucoes/<id>` e os caches imutáveis em
`cache-compartilhado`; ambos permanecem fora do Git.

Cada execução deve possuir um identificador exclusivo em
`configuracao/pipeline.json`. O identificador e o percentual configurado são
validados antes de qualquer processamento, e resultados existentes não podem
ser sobrescritos quando `proteger_resultados_existentes` estiver habilitado.
