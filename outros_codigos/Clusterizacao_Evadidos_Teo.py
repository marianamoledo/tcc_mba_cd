# %%
# Importando as bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import pipeline
from sklearn import cluster
from feature_engine import encoding
from yellowbrick.cluster import KElbowVisualizer
import seaborn

# %%
# Definindo opções de exibição para o Pandas
# Esta linha configura o número máximo de linhas que o Pandas exibirá ao mostrar um DataFrame
pd.set_option("display.max_rows", 1000)
# Essa linha configura o número máximo de colunas que o Pandas exibirá ao mostrar um DataFrame
pd.set_option("display.max_columns", 1000)

# %%
# Lendo os dados de um arquivo Excel e armazenando-os em um DataFrame
df = pd.read_excel("Dataset-UFF-Graduacao-Evadidos-CEPs_Validos.xlsx")
df.head()

# %%
# Definindo listas para diferentes conjuntos de características
socio_demo = ['ESTADOCIVIL', 'SEXO', 'COR', 'ACAOAFIRMATIVA']
distancia = ['DISTANCIA_NUM']
enem = ['ENEMLINGUAGEM', 'ENEMHUMANAS',
        'ENEMNATURAIS', 'ENEMMATEMATICA', 'ENEMREDACAO']
curso = ['TURNOATUAL']

# %%
# criação de uma instância do modelo KMeans para realizar a clusterização.
model_socio = cluster.KMeans(random_state=42, max_iter=1000,)
onehot_socio = encoding.OneHotEncoder(variables=socio_demo)
visualizer = KElbowVisualizer(model_socio, k=(2, 12))
visualizer.fit(onehot_socio.fit_transform(
    df[socio_demo]))
visualizer.show()

model_socio = cluster.KMeans(n_clusters=visualizer.elbow_value_)
pipe_socio = pipeline.Pipeline([('onehot', onehot_socio),
                                ('model_socio', model_socio)])  # pipeline que combina a etapa de transformação (one-hot encoding) com a etapa de modelagem (KMeans com o número de clusters sugerido)
 
# ajustando o pipeline aos dados das características socio-demográficas originais, ou seja, você está aplicando tanto a transformação one-hot quanto a modelagem de clusterização aos dados.
pipe_socio.fit(df[socio_demo])

df_socio = pipe_socio[:-1].transform(df[socio_demo])
df_socio['cluster_name'] = model_socio.labels_
df['cluster_socio_demo'] = model_socio.labels_

# %%
summary_socio = df_socio.groupby(['cluster_name']).mean()
summary_socio['qtde'] = df_socio.groupby(
    ['cluster_name'])['ESTADOCIVIL_CASADO'].count().tolist()

# %%
# Salvando o resumo dos clusters sócio-demográficos em um arquivo CSV
summary_socio.T.to_csv("summary_socio.csv", sep=";")

# %%
# Criando um mapa de calor (heatmap) para o resumo dos clusters sócio-demográficos
seaborn.heatmap(summary_socio[summary_socio.columns[:-1]])

# %%
# Agrupando com base na característica de distância usando o algoritmo KMeans
model_distancia = cluster.KMeans(random_state=42, max_iter=1000,)

visualizer = KElbowVisualizer(model_distancia, k=(2, 12))
visualizer.fit(df[distancia])  # Fit the data to the visualizer
visualizer.show()  # Finalize and render the figure

model_distancia = cluster.KMeans(
    n_clusters=visualizer.elbow_value_, max_iter=1000,)

model_distancia.fit(df[distancia])  # treinando o modelo KMeans

df_distancia = df[distancia]
df_distancia['cluster_name'] = model_distancia.labels_

# %%
summary_distancia = df_distancia.groupby(['cluster_name']).mean()
summary_distancia['qtde'] = df_distancia.groupby(
    ['cluster_name'])[distancia[0]].count().tolist()
summary_distancia

# %%
# Atribuindo os clusters de distância ao DataFrame principal
df['cluster_distancia'] = model_distancia.labels_

# Combinando clusters de diferentes características em um único cluster
df['cluster_all'] = df['cluster_distancia'].astype(
    str) + df['cluster_socio_demo'].astype(str)

# Agrupando os dados com base nos clusters combinados e contando as ocorrências
df.groupby(['cluster_all'])['ESTADOCIVIL'].count()
