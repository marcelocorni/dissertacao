# Machine learning

Esta etapa prepara os dados e executa dois detectores não supervisionados:

- Isolation Forest, com escore de isolamento;
- Autoencoder, com erro de reconstrução.

## Ordem

1. Features e recortes:
   `.\03-machine-learning\01-features`.
2. Isolation Forest:
   `.\03-machine-learning\03-if`.
3. Autoencoder:
   `.\03-machine-learning\02-ae`.

## Desenho temporal

| Janela | Período | Uso |
|---|---|---|
| estresse pré-Dencun | 01/01/2024–12/03/2024 | comportamento anterior à Dencun |
| transição Dencun | 13/03/2024–31/03/2024 | mudança de protocolo |
| treino | 01/04/2024–31/12/2024 | scaler e ajuste dos modelos |
| validação | 01/01/2025–30/04/2025 | calibração dos limiares |
| transição Pectra | 01/05/2025–31/05/2025 | mudança de protocolo |
| teste final | 01/06/2025–30/11/2025 | avaliação final |
| estresse Fusaka | 01/12/2025–31/12/2025 | robustez futura |

Os recortes são cronológicos e não se sobrepõem. Transformações estatísticas e
modelos são ajustados apenas no treino. Os limiares Q99, Q99,5 e Q99,9 são
obtidos apenas na validação.

## Features excluídas

Binance, Fear & Greed, Label Cloud, hashes, endereços e calldata não entram nas
matrizes numéricas. Hashes, endereços, bloco e timestamp permanecem nos
metadados para rastreabilidade e pós-processamento.
