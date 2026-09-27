# Painel de resultados

Interface Streamlit, somente de leitura, para apresentação do pipeline e
comparação de execuções independentes. O painel não executa modelos, não
reprocessa resultados e não abre arquivos Parquet. Ele lê manifestos JSON,
resumos CSV e figuras PNG/PDF em `execucoes/amostra-XXpct`.

## Iniciar

Na raiz `E:\Mestrado\src-multiexecucoes`:

```powershell
uv run --with streamlit --with pandas --with matplotlib --with reportlab `
  streamlit run ".\06-painel-resultados\painel_resultados.py"
```

Selecione a etapa no menu lateral e marque as execuções a comparar. Uma execução
parcial pode ser selecionada: suas etapas já concluídas aparecem normalmente;
as seções que dependem de artefatos ainda ausentes mostram um aviso. A página é
atualizada na próxima interação, sem reiniciar o painel.

## Conteúdo e interpretação

| Etapa | Fontes principais | Leitura |
|---|---|---|
| Coleta | manifesto dos splits temporais | cobertura diária, intervalos e papel das janelas |
| Remoção de `input` | código e medição S3/local opcional | justificativa, tamanho antes/depois; diferença inclui recompressão |
| Features | resumo do dataset, manifesto das janelas, correlação | blocos completos, treino/validação/teste, redundância entre features |
| Pré-processamento/drift | validação das matrizes, resumo de PSI/KS | ausência de valores inválidos e mudança de distribuição |
| AE e IF | resumos de treinamento, curvas de reconstrução do AE e taxas de anomalia | volume treinado, tempo, convergência e comportamento por janela |
| Convergência | resumo integrado AE/IF | união, interseção e Jaccard dos candidatos |
| Regras C# | resumos por detector/janela | eventos candidatos, ainda não confirmados |
| Semântica | manifestos de validação, resumo consolidado, eventos e papéis adjudicados | destino dos candidatos C#, inserção confirmada, deslocamento confirmado/provável, políticas e sobreposição entre execuções |
| Label Cloud | resumo das tags e métricas hipotéticas | comparação exploratória, não verdade-terreno exaustiva |
| Avaliação final | cobertura dos rótulos, seleção validação/teste, métricas por tipo, curvas, top-k, superioridade pareada e matrizes | população efetivamente avaliada, ROC-AUC e PR-AUC agregadas e por inserção/deslocamento, lift nos top 1%, 5% e 10%, ordenação atacante–vítima dentro do evento e resultados sob o limiar escolhido na validação |

As comparações a partir de `03-machine-learning` usam os percentuais marcados.
Contagens absolutas crescem com a amostra; compare também proporções e métricas.
O teste final não participa da escolha de modelo ou limiar. As métricas semânticas
são condicionais aos candidatos C# e negativos contextuais, não à totalidade de
transações Ethereum.

O destino dos candidatos estruturais é obtido dos manifestos por janela. O
gráfico compara apenas execuções com as sete janelas completas; janelas já
disponíveis de uma execução parcial podem ser consultadas individualmente,
sem somá-las como se representassem a execução inteira. As situações são
mutuamente exclusivas por **evento candidato**, não por transação única.

Na avaliação final, a cobertura da referência semântica distingue atacantes,
vítimas contextuais, casos sem rótulo na política e conflitos excluídos.
Esses denominadores explicam quais transações entram efetivamente nas métricas
estrita e de sensibilidade do teste final.

O lift da avaliação final compara a precisão no topo dos escores com a
prevalência de atacantes **na população semanticamente avaliada**. A referência
1× equivale a seleção aleatória dessa mesma população; o resultado não estima
a prevalência de ataques na rede inteira. O painel mostra separadamente as
políticas estrita e de sensibilidade e mantém Q99/Q99,5 fora dessa medida de
ranking, que independe de um limiar binário.

A tabela por tipo separa inserção e deslocamento no teste final para as
configurações escolhidas previamente pela validação agregada. O gráfico de
ROC-AUC permite alternar entre as políticas estrita e de sensibilidade. Cada
métrica acompanha seus denominadores; a subpopulação de deslocamento é menor,
e uma ROC-AUC inferior a 0,5 indica ordenação inversa nessa subpopulação, não
ausência de eventos confirmados.

O heatmap de superioridade pareada da política estrita compara, dentro de cada
evento confirmado, o escore do atacante com o da vítima. As colunas são janelas,
as linhas são configurações AE/IF e cada célula é a fração de pares em que o
atacante recebeu escore maior. Inserções podem contribuir com duas pernas
atacantes e, portanto, dois pares. A tabela complementar informa denominadores
e distingue a taxa por par da taxa por evento; nenhuma das duas equivale a
recall em um limiar binário.

Na validação semântica, a comparação de conjuntos usa apenas eventos
**confirmados** e hashes das transações com papel de atacante na política
estrita. Para cada par de execuções selecionadas, mostra interseção, casos
exclusivos de cada uma e percentual preservado da execução menor. Os eventos
são identificados pelos hashes de atacante(s) e vítima, não pelos IDs internos
de auditoria. Há detalhe por janela e tipo de ataque e um CSV com os hashes
comparados. Uma execução sem consolidação concluída não entra nessa comparação.

Cada tabela exibida pode ser baixada como CSV. Cada figura disponível em PDF
pode ser baixada no formato original. O botão ao fim da página exporta um PDF
com o texto, gráficos exibidos e versão resumida das tabelas; os CSVs preservam
os dados tabulares completos. Os dados e consultas permanecem locais.

## Medir o espaço poupado pelos arquivos brutos

O script de remoção não persistiu a soma dos tamanhos antes/depois. Para
documentá-la agora, `01_medir_armazenamento_s3.py` lista os tamanhos dos objetos
originais no bucket público `aws-public-blockchain` e compara com os tamanhos
atuais dos arquivos locais de 2024 e 2025. Não lê o conteúdo dos Parquets. O
prefixo contém partições `date=AAAA-MM-DD`, correspondentes às pastas locais:

```powershell
uv run ".\06-painel-resultados\01_medir_armazenamento_s3.py" `
  --years 2024 2025 --region us-east-2 --no-sign-request
```

O prefixo padrão é `s3://aws-public-blockchain/v1.0/eth/transactions/` e pode
ser substituído com `--s3-prefix` se a origem dos dados mudar.
O script grava `resultados-armazenamento/01_comparacao_s3_local.csv` e
`02_manifest_medicao.json`; o painel passa a exibir automaticamente a tabela.
Verifique `cobertura_identica=True` para cada ano antes de usar os totais.

A diferença S3/local reflete **remoção de `input` e regravação em ZSTD**, pois o
script de pré-processamento muda a compressão por padrão. Não a atribua
inteiramente à remoção da coluna. A comparação também não substitui o log
original de tamanho antes/depois caso ele esteja disponível.
