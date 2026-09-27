# Histórico de execuções

Este diretório mantém apenas manifestos e resumos comparáveis das execuções.
Os resultados de cada percentual ficam em `execucoes/<id>`; o Git inclui seus
artefatos leves, mas exclui os Parquets. Os caches reutilizáveis ficam em
`cache-compartilhado` e permanecem fora do Git.

Cada execução deve possuir um identificador exclusivo em
`configuracao/pipeline.json`. O identificador e o percentual configurado são
validados antes de qualquer processamento, e resultados existentes não podem
ser sobrescritos quando `proteger_resultados_existentes` estiver habilitado.
