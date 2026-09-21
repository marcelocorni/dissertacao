# Inventário dos códigos

| Ordem | Código | Função | Resultado principal |
|---:|---|---|---|
| 1 | `.\02-pre-processamento\01-remover_coluna_input_parquet.py` | remove `input` dos Parquets | substitui o próprio Parquet após validação |
| 2 | `.\03-machine-learning\01-features\01_auditoria_selecao_features.py` | audita esquema, features e correlações | `.\03-machine-learning\01-features\resultados` |
| 3 | `.\03-machine-learning\01-features\02_gerar_matrizes_features.py` | gera matrizes A, B e C | `.\03-machine-learning\01-features\resultados-matrizes` |
| 4 | `.\03-machine-learning\01-features\03_gerar_splits_temporais.py` | gera as sete janelas | `.\03-machine-learning\01-features\resultados-splits` |
| 5 | `.\03-machine-learning\01-features\04_gerar_figuras_correlacao_pdf.py` | gera correlações vetoriais | `.\03-machine-learning\01-features\resultados` |
| 6 | `.\03-machine-learning\01-features\05_preprocessar_matrizes.py` | ajusta e aplica RobustScaler | resultados de cada recorte e `.\03-machine-learning\01-features\resultados-preprocessamento` |
| 7 | `.\03-machine-learning\01-features\06_analisar_drift_temporal.py` | mede PSI e KS aproximado | `.\03-machine-learning\01-features\resultados-drift` |
| 8 | `.\03-machine-learning\03-if\01_treinar_isolation_forest.py` | treina IF e calcula escores | `.\03-machine-learning\03-if\resultados` |
| 9 | `.\03-machine-learning\02-ae\01_treinar_autoencoder.py` | treina AE e calcula escores | `.\03-machine-learning\02-ae\resultados` |
| 10 | `.\05-analise-resultados\01_analisar_ae_if.py` | integra AE e IF | `.\05-analise-resultados\resultados` |
| 11 | `.\04-pos-processamento\02_preparar_labelcloud_historico.py` | normaliza o Label Cloud | `.\04-pos-processamento\resultados-labelcloud` |
| 12 | `.\04-pos-processamento\03_vincular_labelcloud_candidatos.py` | vincula tags aos candidatos | `.\04-pos-processamento\resultados-labelcloud\vinculos-candidatos` |
| 13 | `.\04-pos-processamento\04_gerar_rotulos_hipoteticos_labelcloud.py` | gera o experimento binário hipotético | `.\04-pos-processamento\resultados-labelcloud` |
| 14 | `.\04-pos-processamento\05-rotulador-front-running-csharp` | gera candidatos estruturais | `.\04-pos-processamento\05-rotulador-front-running-csharp\resultados-v3` |
| 15 | `.\04-pos-processamento\06-validacao-semantica-front-running\11_executar_pipeline_janelas.py` | executa RPC, validação e adjudicação | `.\04-pos-processamento\06-validacao-semantica-front-running\resultados` |
| 16 | `.\04-pos-processamento\06-validacao-semantica-front-running\12_consolidar_rotulos_semanticos.py` | consolida rótulos | `.\04-pos-processamento\06-validacao-semantica-front-running\resultados\consolidado` |
| 17 | `.\05-analise-resultados\02_avaliar_rotulos_semanticos.py` | avalia AE/IF com rótulos semânticos | `.\05-analise-resultados\resultados-rotulos-semanticos` |
| 18 | `.\05-analise-resultados\03_avaliar_labelcloud_hipotetico.py` | calcula métricas com Label Cloud | `.\05-analise-resultados\resultados-labelcloud-hipotetico` |

A interface de consulta está em
`.\04-pos-processamento\06-validacao-semantica-front-running\05_interface_dossie_semantico.py`.
