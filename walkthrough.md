# Walkthrough: Refatoração MLOps (Embraco Predictive Platform)

Nesta etapa criamos uma casca de **Engenharia de Machine Learning (MLOps)** ao redor da lógica estatística e de métricas sensoriais do modelo preditivo do compressor Embraco. 

Conseguimos resolver o problema de scripts "duros" (*hardcoded*) e pacotes desconexos, criando uma linha de produção automatizada ponta a ponta.

---

## 🏗️ 1. O Pacote do Modelo: `EmbracoAnomalyDetector`

Antes, o script isolava os cálculos da média e desvio padrão de Z-Score da estrutura do modelo (Isolation Forest). Agora, tudo reside em uma classe que segue o padrão do _scikit-learn_. 
Ela empacota as regras de negócio de filtragem por estado operativo (*Uptime Filter*).

```python
from src.ml.detector_model import EmbracoAnomalyDetector

detector = EmbracoAnomalyDetector(n_estimators=200, target_operating_state="steady_state")

# Ao treinar, ele salva médias, desvios, thresholds do Z-Score Híbrido  internamente.
detector.fit(X_train_com_janelas_healthy, Context_DF)
```

## 🎛️ 2. Arquitetura Orientada a CLI (Command Line Interface)

Todos os scripts ganharam o módulo padrão `argparse`. 
Com isso, você pode alterar janelas, ciclos em hertz, quantidade de árvores do Random Forest ou os arquivos de entrada e saída livremente no seu terminal (Bash, PowerShell) em produção ou em *docker containers*.

> [!TIP]
> Você pode rodar _qualquer_ script passando o argumento `--help` (ex: `python -m src.ml.train_em2x3125u_anomaly_model --help`) para ver todas as opções customizáveis disponíveis para ele.

## 🤖 3. Simulador de Inferência Real (`predict_em2x3125u.py`)

Para comprovar o processo de "Deploy", adicionamos um script de inferência. Ele lê os dados puramente vibro-acústicos brutos que vieram da ponta e, em tempo real, **faz a janelagem das features, chama o Pipeline do _sklearn_ treinado e devolve de quem é a anomalia (0/1)** sem treinamento no meio.

```shell
python -m src.ml.predict_em2x3125u --input_csv data/raw/novo_csv.csv --output_csv predictions.csv
```

## 🎼 4. O Maestro: `run_pipeline.py`

Com todos os scripts adaptados via terminal, criamos um mestre chamando o pacote `subprocess` do Python, permitindo que a execução encadeada ocorra sozinha!

A sequência orquestrada:
1. Geração Automática das falhas simuladas via `generate_dataset`.
2. Geração das FFTs (frequências e energia) + janelas RMS via `window_timeseries`.
3. Treino do `EmbracoAnomalyDetector` e empacotamento para `.joblib`.
4. Avaliação completa e extração de insights por matriz confusa.
5. Invocação da Inferência Real utilizando os dados das falhas produzidas para testar a aderência em produção.

Ao longo do desenvolvimento, executamos com sucesso o `run_pipeline.py`. Você pode utilizá-lo daqui em diante para prototipar hiperparâmetros rapidamente.
