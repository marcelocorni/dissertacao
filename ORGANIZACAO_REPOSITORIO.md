# Organização do repositório

Todos os caminhos deste documento são relativos à raiz do repositório. O
identificador `<id>` vem de `execucao.id` em `configuracao/pipeline.json`;
por exemplo, `amostra-05pct`.

```text
src-multiexecucoes/
├── configuracao/                 parâmetros, percentual e provedores RPC
├── executar_pipeline.py          execução e retomada do experimento
├── 01-coleta-dados/              instruções de obtenção dos Parquets
├── 02-pre-processamento/         remoção opcional da coluna input
├── 03-machine-learning/          features, recortes, AE e IF
├── 04-pos-processamento/         Label Cloud, rotulador C# e semântica RPC
├── 05-analise-resultados/        métricas e figuras
├── 06-painel-resultados/         consulta e comparação das execuções
├── execucoes/<id>/               resultados isolados de cada percentual
├── historico-execucoes/          resumo de cada execução concluída
├── cache-compartilhado/          blocos RPC e metadados reutilizáveis
└── .tmp/                         arquivos temporários de processamento
```

Os Parquets Ethereum completos ficam fora do repositório. Sua localização é
definida em `configuracao/pipeline.json`, no campo `dados.root_template`.
Cada percentual deve ter um `<id>` próprio; assim, alterar a configuração de
uma nova execução não substitui os resultados das anteriores.

## Resultados de cada execução

Cada pasta `execucoes/<id>/` contém
`00_manifest_multiexecucao.json`, que registra as etapas e o estado geral.
Os demais resultados espelham o diretório do respectivo código-fonte:

| Conteúdo | Caminho dentro de `execucoes/<id>/` |
|---|---|
| Auditoria de features e figuras de correlação | `03-machine-learning/01-features/resultados/` |
| Matrizes de treinamento | `03-machine-learning/01-features/resultados-matrizes/` |
| Sete janelas temporais | `03-machine-learning/01-features/resultados-splits/` |
| Pré-processamento e parâmetros do scaler | `03-machine-learning/01-features/resultados-preprocessamento/` |
| Drift temporal | `03-machine-learning/01-features/resultados-drift/` |
| Modelos, escores e curvas AE | `03-machine-learning/02-ae/resultados/` |
| Modelos e escores IF | `03-machine-learning/03-if/resultados/` |
| Análise integrada AE/IF | `05-analise-resultados/resultados/` |
| Comparação exploratória Label Cloud | `04-pos-processamento/resultados-labelcloud/` e `05-analise-resultados/resultados-labelcloud-hipotetico/` |
| Candidatos C# por janela | `04-pos-processamento/05-rotulador-front-running-csharp/resultados-v3/` |
| Homologação RPC, validação semântica e dossiês | `04-pos-processamento/06-validacao-semantica-front-running/resultados/` |
| Rótulos semânticos consolidados | `04-pos-processamento/06-validacao-semantica-front-running/resultados/consolidado/` |
| Avaliação semântica final, tabelas e gráficos | `05-analise-resultados/resultados-rotulos-semanticos/` |

Na validação semântica, cada janela possui uma subpasta própria em
`.../resultados/<janela>/`. O escalonador registra a distribuição entre
provedores em `.../resultados/00_manifest_pipeline_paralelo.json`.

## Histórico, cache e versionamento

`historico-execucoes/<id>.json` resume uma execução concluída. O
`cache-compartilhado/` contém dados RPC reaproveitáveis entre percentuais e
não é uma cópia dos resultados de cada execução. A pasta `.tmp/` é temporária.

O Git inclui o código, a documentação, os manifestos e os resultados leves
das execuções. Parquet, caches, logs e arquivos temporários são excluídos pelo
`.gitignore`. Portanto, reproduzir análises que dependem de Parquet exige
acesso às bases externas ou a geração local desses arquivos.

As etapas, os códigos e os artefatos principais estão descritos no
[inventário dos códigos](INVENTARIO_CODIGOS.md); os detalhes de cada resultado
constam no README da respectiva etapa.
