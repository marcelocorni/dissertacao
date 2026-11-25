# Revisão experimental com seleção de hiperparâmetros

Fluxo opcional, independente de `executar_pipeline.py` e de `configuracao/pipeline.json`. Reutiliza os mesmos blocos de uma execução de 5%, mas grava novos parâmetros, modelos, escores e avaliações em outro identificador. Não consulta RPC, não altera a execução-fonte e não escreve no cache compartilhado.

## Protocolo

- AE: três arquiteturas simétricas, duas taxas de aprendizado e dois valores de weight decay, totalizando 12 combinações.
- IF: 100/200/400 árvores e 2.048/4.096/8.192/16.384 amostras por árvore, totalizando 12 combinações.
- Cada combinação é ajustada nas cinco representações. Seleção por algoritmo pela média das cinco Average Precisions (APs) na população semântica estrita da validação. Empates exatos seguem a prioridade congelada da grade, com os parâmetros históricos primeiro.
- Busca: 120 ajustes. Produção: dez ajustes finais. As alternativas não são garantia de otimalidade global. Uma semente por candidato não caracteriza a variabilidade da seleção.
- O ajuste continua não supervisionado, mas a seleção de hiperparâmetros é guiada pelos rótulos semânticos da validação. Não apresentar todo o procedimento como independente desses rótulos.
- O teste histórico já foi examinado. O fluxo congela a nova busca antes da pontuação do teste, mas isso não torna a revisão prospectiva desde a origem do experimento.

## Amostragem e transformações

As regras de SHA-256 estão em `core.py`. Aproximadamente 10% dos blocos do treinamento formam a validação interna do AE; não há blocos compartilhados com a parcela de ajuste. Outra regra seleciona cerca de 10% dos blocos para a busca do AE. O ajuste final usa a parcela completa de ajuste e monitora os blocos internos reservados.

O IF utiliza blocos completos da parcela de ajuste até atingir pelo menos 500.000 transações. A mesma população é usada na busca e no ajuste final. Esse alvo não é o tamanho por árvore.

O scaler é reajustado na parcela de ajuste, sem os blocos internos reservados, e congelado. A busca reduz os dados dos gradientes do AE, não a população usada para estimar as transformações. As matrizes brutas são entradas somente de leitura; as preprocessadas são novas.

## Planejar sem executar

Na raiz do repositório, com o ambiente científico disponível:

```powershell
uv run executar_revisao.py --source-run .\execucoes\amostra-05pct --output .\execucoes\amostra-05pct-revisao-hpo --stage plan
```

`plan` confere caminhos e presença dos arquivos, mostra o protocolo e o volume de entradas. Não lê todo o conteúdo dos Parquets, não cria a nova pasta e não valida integralmente schemas ou cobertura. A primeira execução de processamento calcula hashes dos arquivos de entrada e do código. Dependências do ambiente podem ser preparadas pelo uv; isso não constitui chamada RPC.

## Etapas de processamento

Cada chamada exige `--execute`. Por exemplo, **somente quando a execução estiver autorizada**:

```powershell
uv run executar_revisao.py --source-run .\execucoes\amostra-05pct --output .\execucoes\amostra-05pct-revisao-hpo --stage prepare --execute
```

Ordem: `prepare`, `search`, `fit`, `score`, `export`, `complements`, `descriptive`, `figures`, `reports`. `--stage all --execute` executa essa sequência. O dispositivo padrão é `auto`; usar `--device cpu` ou `--device cuda` para fixá-lo. Manter a escolha na retomada.

| Etapa | Produto |
| --- | --- |
| prepare | listas de blocos, parâmetros do scaler e matrizes de treino/validação |
| search | APs individuais, históricos e avaliações dos candidatos; seleção ao completar a grade |
| fit | dez modelos finais, checkpoint aceito, épocas e tempos |
| score | escores completos das sete janelas; quantis calibrados apenas na validação |
| export | CSVs e manifestos compatíveis com geradores históricos, escores por representação |
| complements | baselines, seleção por política, métricas semânticas, pares e ICs agrupados |
| descriptive | funil por padrão/janela, recorrência de hashes e remetentes, tipos 2 e 4 |
| figures | curvas de treinamento, ROC/PR, confusão, AP e superioridade pareada |
| reports | concordância e figuras de escores, avaliação Label Cloud, drift e correlação reexportada |

O layout novo tem seu próprio `protocol.json` e manifestos por etapa; não pressupor que painéis que exigem o layout completo de `executar_pipeline.py` reconhecerão automaticamente a execução revisada. As saídas de compatibilidade ficam em `exports/AE` e `exports/IF`.

## Avaliação e limites

As três heurísticas são `gas_price_rank_block`, `transaction_index_pct` e `gas_used_ratio`, sobre atributos brutos. A orientação (+ ou −) é escolhida pela AP na validação por política, com prioridade para + em empate. O corte é escolhido pelo maior F1 entre os três quantis da validação completa. Empates no escore são preservados.

A avaliação pareada exclui o evento inteiro se algum participante tiver conflito de papel. O relatório registra quantos eventos foram excluídos. Ausência inesperada de escore ou metadado interrompe o cálculo em vez de ser tratada como conflito. Essa regra difere da implementação pareada histórica e pode mudar especialmente os resultados de sensibilidade, sem alterar os rótulos originais.

O bootstrap utiliza 2.000 sorteios de blocos com reposição, semente 20261018, mesmos sorteios por método e contraste IF menos AE. Transações únicas, pares e eventos são estimandos separados. Eventos que abrangem múltiplos blocos interrompem a análise para revisão do agrupamento. Mais de 5% de réplicas inválidas suprime os intervalos e exige revisão metodológica. Os intervalos são condicionais aos modelos ajustados, não medem variabilidade de treinamento ou seleção.

O endereço remetente é uma unidade observável, não prova de identidade de operador. A exclusão do tipo 4 não demonstra causalidade nem simula um retreinamento. A associação Label Cloud usa os dados históricos, sem reaquisição ou correção automática de categorias. A auditoria humana é uma atividade separada e não será declarada concluída pelo fluxo.

## Sensibilidade ao remetente mais recorrente

A análise complementar da revisão 04 reutiliza os escores congelados de AE B-ST e IF B completa no teste estrito de 2025. Ela reproduz a avaliação completa e exclui os eventos inteiros nos quais o remetente mais recorrente tem papel atacante. AP e ROC-AUC usam transações únicas ainda representadas nos eventos retidos. A superioridade por par mantém apenas pares de eventos completos.

Na raiz do repositório, com DuckDB instalado e as duas execuções locais disponíveis:

```powershell
python revisao_experimental/sensibilidade_remetente.py --source-run execucoes/amostra-05pct --scores-run execucoes/amostra-05pct-revisao-hpo
```

O script lê os Parquets locais de rótulos, papéis, metadados e escores. Os caminhos padrão apontam para as execuções acima. Ele valida os denominadores e as métricas de referência; imprime JSON, ou grava em um arquivo novo se `--output` for fornecido. Não é necessário executar outra etapa do pipeline, retreinar modelos ou consultar RPC.

O [resultado versionado](resultados/sensibilidade_remetente.json) registra a exclusão de 1.515 eventos de inserção. Nos 894 eventos restantes, o IF passa à frente em AP, ROC-AUC e superioridade por par. A exclusão altera a composição por padrão, e o contraste não isola um efeito causal do endereço nem mede incerteza entre remetentes.

## Revisão humana de 1.000 eventos

A [consolidação da revisão 05](resultados/pareceres_1000_agregados.json) reúne os pareceres finalizados de um único avaliador, vinculado à orientação, em quatro campanhas (12, 100, 500 e 388 eventos do teste de 2025). A amostra foi dirigida por estratos, sem repetição de eventos ou transações.

Dos 1.000 pareceres, 460 foram compatíveis, 198 não sustentados e 342 inconclusivos. A concordância descritiva foi de 623/1.000: 447 confirmados operacionalmente com parecer compatível e 176 rejeitados com parecer não sustentado. Exigindo também o mesmo padrão nos confirmados, foram 500/1.000. Dos 448 eventos de inserção confirmados operacionalmente, 123 receberam interpretação de outro mecanismo; os 106 deslocamentos confirmados revisados ficaram inconclusivos.

Essas proporções descrevem o cruzamento entre parecer e referência operacional na amostra; não estimam acurácia populacional. A consolidação preserva os rótulos e as métricas do pipeline. O JSON contém apenas contagens agregadas por campanha e por estrato; os CSVs brutos e os registros individuais não integram a publicação.

## Complementos retrospectivos da revisão 07

Os [resultados agregados](resultados/revisao_07/resultados.json) usam os escores persistidos da execução `amostra-05pct-revisao-hpo` e os rótulos estritos da execução `amostra-05pct`. O cálculo reutilizou os modelos, os cortes e as orientações das heurísticas, sem retreino ou consultas RPC.

| Análise | População e resultados |
| --- | --- |
| Estratos da revisão humana | Comparação pareada dos papéis candidatos nos 1.000 casos. AP e ROC-AUC usam apenas os 554 eventos elegíveis com rótulos estritos originais. Os pareceres definem os grupos de análise. |
| Remetente dominante em inserção | Contraste dos escores de AE B-ST e IF B completa em 2.302 eventos de inserção, dos quais 787 permanecem após a exclusão definida na população estrita completa. |
| Orçamento de inspeção e empates | [Top-k e intervalos de recuperação devidos a empates](resultados/revisao_07/top_k_empates.csv) em três orçamentos. Na fronteira, a regra histórica dá prioridade aos hashes maiores para escores iguais. |
| Status da execução | [Distribuições de escores](resultados/revisao_07/escores_por_status.csv) nas sete janelas e dez configurações, [por papel estrito](resultados/revisao_07/status_papeis_estritos.csv) e [por padrão no teste](resultados/revisao_07/status_padrao_teste.csv). O código 0 inclui o preenchimento de valores ausentes na matriz. |

Esses contrastes descrevem a composição da referência operacional, dos grupos revisados e dos eventos retidos. Os arquivos versionados contêm métricas e contagens agregadas. Os pareceres individuais permanecem em armazenamento local.

## Retomada e segurança

Repetir a chamada da etapa reutiliza somente saídas concluídas com assinatura e hashes válidos. Mudanças de entradas, código, ambiente, dispositivo ou protocolo exigem outra saída. Uma grade incompleta não publica vencedor. Não há opção `--force` para sobrescrever a execução histórica.

Existe um bloqueio global e um por etapa. Após encerramento abrupto do sistema, um arquivo de bloqueio pode permanecer. Confirmar que não há processo ativo antes de remover exclusivamente o bloqueio da execução nova. O programa não remove bloqueios automaticamente.

Arquivos de escores e matrizes novos podem ter hardlinks internos em `exports`, evitando duplicar dados no disco. Não editar esses Parquets manualmente: uma alteração afeta os dois nomes e invalida a verificação de integridade. Não são criados hardlinks para os dados históricos.

## Testes

```powershell
python -m unittest discover -s tests -p test_revisao_experimental.py -v
```

Os testes exigem as bibliotecas científicas do fluxo, bloqueiam conexões de rede e usam dados artificiais. Por padrão, suas pastas temporárias ficam em `E:\Mestrado\revisao-01\etapa-05\testes`; a variável `REVISAO_TEST_ROOT` permite escolher outro diretório. Eles não iniciam a grade real nem alteram dados do experimento.
