# Estrutura do pipeline

```text
src
├── configuracao
├── 01-coleta-dados
├── 02-pre-processamento
├── 03-machine-learning
│   ├── 01-features
│   ├── 02-ae
│   └── 03-if
├── 04-pos-processamento
│   ├── 05-rotulador-front-running-csharp
│   └── 06-validacao-semantica-front-running
└── 05-analise-resultados
```

## Dados e resultados

| Conteúdo | Caminho |
|---|---|
| Configuração dos Parquets | `.\configuracao\pipeline.json` |
| Auditoria de features | `.\03-machine-learning\01-features\resultados` |
| Matrizes de treinamento | `.\03-machine-learning\01-features\resultados-matrizes` |
| Demais recortes temporais | `.\03-machine-learning\01-features\resultados-splits` |
| Parâmetros do scaler | `.\03-machine-learning\01-features\resultados-preprocessamento` |
| Drift temporal | `.\03-machine-learning\01-features\resultados-drift` |
| Modelos e escores AE | `.\03-machine-learning\02-ae\resultados` |
| Modelos e escores IF | `.\03-machine-learning\03-if\resultados` |
| Label Cloud | `.\04-pos-processamento\resultados-labelcloud` |
| Candidatos C# | `.\04-pos-processamento\05-rotulador-front-running-csharp\resultados-v3` |
| Validação semântica | `.\04-pos-processamento\06-validacao-semantica-front-running\resultados` |
| Rótulos consolidados | `.\04-pos-processamento\06-validacao-semantica-front-running\resultados\consolidado` |
| Análise AE/IF | `.\05-analise-resultados\resultados` |
| Avaliação semântica final | `.\05-analise-resultados\resultados-rotulos-semanticos` |
| Avaliação Label Cloud | `.\05-analise-resultados\resultados-labelcloud-hipotetico` |

Os diretórios `modelos`, `escores`, `cache` e `resultados*` são criados pelos
próprios artefatos. O diretório temporário recomendado é
`.\.tmp\duckdb`.
