# Inventário dos códigos

Os caminhos abaixo partem da raiz do repositório. Em cada experimento, `<id>`
é o identificador de `configuracao/pipeline.json` (por exemplo,
`amostra-05pct`). Os resultados são gravados em `execucoes/<id>/`; os blocos
RPC e metadados reutilizáveis ficam em `cache-compartilhado/`.

## Execução completa

| Código | Função | Resultado principal |
|---|---|---|
| `executar_pipeline.py` | executa ou retoma as etapas, conferindo a configuração e os artefatos concluídos | `execucoes/<id>/00_manifest_multiexecucao.json` e `historico-execucoes/<id>.json` |
| `configuracao/pipeline.py` | interpreta a configuração central e os caminhos de cada execução | utilizado pelos demais códigos; não gera resultado isolado |
| `configuracao/pipeline.json` | define dados externos, percentual, identificador e recursos | configuração da execução |
| `configuracao/rpc_providers.json` | define provedores e limites operacionais, sem armazenar endpoints | utilizado na homologação e no escalonamento RPC |

## Etapas executadas pelo pipeline

| Ordem | Código | Função | Resultados em `execucoes/<id>/` |
|---:|---|---|---|
| 1 | `03-machine-learning/01-features/01_auditoria_selecao_features.py` | audita dados e features | `03-machine-learning/01-features/resultados/` |
| 2 | `03-machine-learning/01-features/04_gerar_figuras_correlacao_pdf.py` | gera figuras de correlação | `03-machine-learning/01-features/resultados/` |
| 3 | `03-machine-learning/01-features/02_gerar_matrizes_features.py` | gera matrizes A, B e C | `03-machine-learning/01-features/resultados-matrizes/` |
| 4 | `03-machine-learning/01-features/03_gerar_splits_temporais.py` | define as sete janelas temporais | `03-machine-learning/01-features/resultados-splits/` |
| 5 | `03-machine-learning/01-features/05_preprocessar_matrizes.py` | ajusta o scaler no treino e transforma os recortes | `03-machine-learning/01-features/resultados-preprocessamento/` e pastas dos recortes |
| 6 | `03-machine-learning/01-features/06_analisar_drift_temporal.py` | calcula medidas de drift temporal | `03-machine-learning/01-features/resultados-drift/` |
| 7 | `03-machine-learning/03-if/01_treinar_isolation_forest.py` | treina IF e gera escores | `03-machine-learning/03-if/resultados/` |
| 8 | `03-machine-learning/02-ae/01_treinar_autoencoder.py` | treina AE e gera escores | `03-machine-learning/02-ae/resultados/` |
| 9 | `05-analise-resultados/01_analisar_ae_if.py` | integra os escores e candidatos AE/IF | `05-analise-resultados/resultados/` |
| 10–13 | `04-pos-processamento/02_preparar_labelcloud_historico.py`, `03_vincular_labelcloud_candidatos.py`, `04_gerar_rotulos_hipoteticos_labelcloud.py` e `05-analise-resultados/03_avaliar_labelcloud_hipotetico.py` | prepara e avalia a comparação exploratória com Label Cloud | `04-pos-processamento/resultados-labelcloud/` e `05-analise-resultados/resultados-labelcloud-hipotetico/` |
| 14 | `04-pos-processamento/05-rotulador-front-running-csharp/` | detecta candidatos estruturais | `04-pos-processamento/05-rotulador-front-running-csharp/resultados-v3/` |
| 15 | `04-pos-processamento/06-validacao-semantica-front-running/07_auditar_saidas_csharp.py` | audita as sete saídas C# | `04-pos-processamento/06-validacao-semantica-front-running/resultados/auditoria-csharp/` |
| 16 | `04-pos-processamento/06-validacao-semantica-front-running/15_testar_provedores_rpc.py` | homologa os endpoints configurados | `04-pos-processamento/06-validacao-semantica-front-running/resultados/auditoria-rpc/` |
| 17 | `04-pos-processamento/06-validacao-semantica-front-running/16_executar_pipeline_janelas_paralelo.py` | distribui as janelas entre RPCs e retoma blocos em cache | `04-pos-processamento/06-validacao-semantica-front-running/resultados/00_manifest_pipeline_paralelo.json` e resultados por janela |
| 18 | `04-pos-processamento/06-validacao-semantica-front-running/12_consolidar_rotulos_semanticos.py` | consolida eventos, papéis e rótulos | `04-pos-processamento/06-validacao-semantica-front-running/resultados/consolidado/` |
| 19 | `05-analise-resultados/02_avaliar_rotulos_semanticos.py` | produz métricas e gráficos finais de AE/IF | `05-analise-resultados/resultados-rotulos-semanticos/` |

O escalonador da etapa 17 chama `11_executar_pipeline_janelas.py` para cada
janela. Este executa, nessa ordem, `08_enriquecer_fila_por_bloco.py`,
`02_validar_semantica.py`, `03_gerar_dossie_auditoria.py`,
`04_enriquecer_tokens_rpc.py`, `06_adjudicar_dossie_assistido.py` e
`09_validar_adjudicacoes_assistidas.py`. Cada janela produz artefatos
`01` a `19` na sua pasta `resultados/<janela>/`, descritos no
[README da validação semântica](04-pos-processamento/06-validacao-semantica-front-running/README.md).

## Ferramentas complementares

| Código | Uso |
|---|---|
| `02-pre-processamento/01-remover_coluna_input_parquet.py` | remove `input` das bases externas; etapa preparatória opcional, fora da execução por percentual |
| `04-pos-processamento/06-validacao-semantica-front-running/05_interface_dossie_semantico.py` | consulta os dossiês sem adjudicação manual |
| `04-pos-processamento/06-validacao-semantica-front-running/10_validar_regressao_rpc.py` | testa o coletor RPC por bloco antes do processamento |
| `04-pos-processamento/06-validacao-semantica-front-running/14_reconstruir_cache_rpc.py` | reconstrói o cache em uma recuperação específica |
| `06-painel-resultados/painel_resultados.py` | compara as execuções e apresenta os resultados sem ler Parquet |
| `06-painel-resultados/01_medir_armazenamento_s3.py` | estima a diferença de armazenamento entre os objetos originais da AWS e as bases locais |

Para interpretar os arquivos produzidos em cada etapa, consulte o README do
respectivo diretório e o [guia de reprodução](GUIA_REPRODUCAO_SEGURA.md).
