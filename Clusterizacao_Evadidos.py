# %%

# Importando as bibliotecas necessárias
import pandas as pd
from sklearn import pipeline
from sklearn import cluster
from feature_engine import encoding
from yellowbrick.cluster import KElbowVisualizer
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import seaborn

# Definindo opções de exibição para o Pandas
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)

# Lendo os dados de um arquivo Excel e armazenando-os em um DataFrame
df = pd.read_excel("Dataset-UFF-Graduacao-Evadidos-CEPs_Validos.xlsx")

# %%

# Colunas que deseja excluir
columns_excluir = ['CODALUNO', 'STATUSFORMACAO','CR', 'DISTANCIA','STATUS_CURSO']
#'cep_destino', 'cep_origem','ANOINGRESSO','NOME_CURSO','IDADE','SEMESTREDESVINCULACAO','ANODESVINCULACAO','SEMESTREINGRESSO','PERIODODISC','RESULTDISC','NOTADISC','DISCIPLINA','BAIRRO','CIDADE','MOBILIDADE','TRANCAMENTOS','TEMPOPERMANENCIA','CODTURNOATUAL','CHCURSADA','CODTURNOINGRESSO']
relevante_features = [col for col in df.columns if col not in columns_excluir]
# print("Features relevantes:", relevante_features)

# %%

# Preparando dados de entrada (X) para a seleção de características
X = df[relevante_features]

# Codificação one-hot das colunas categóricas do df
X_encoded = pd.get_dummies(X)

print("Colunas one-hot:", X_encoded.columns)

# %%

# Definindo o número de características que desejamos selecionar
num_features = 12  

# Criando um array de zeros como a variável de destino fictícia
#Em muitos algoritmos de aprendizado de máquina, como regressão e classificação, é comum trabalhar com um conjunto de dados que consiste em duas partes principais: as características (também chamadas de entradas) e os rótulos (também chamados de alvos ou saídas). No entanto, em algumas situações, como quando estamos lidando com tarefas de seleção de características, não estamos realmente fazendo uma tarefa de previsão ou classificação, mas sim um processo de pré-processamento para escolher as características mais relevantes.
#O método fit_transform do seletor de características (como SelectKBest) espera receber tanto as características (X) quanto os rótulos (y) como entrada. No entanto, para cenários em que não temos um rótulo real para cada instância de dado, ainda precisamos fornecer algum tipo de rótulo para que a função possa operar.
#Aqui é onde entra o y_dummy, o array de zeros fictício:
#y_dummy é criado como um array de zeros com o mesmo número de instâncias (linhas) que o conjunto de características (X_encoded). Isso é feito usando np.zeros(X_encoded.shape[0]). Basicamente, estamos criando um array de zeros com o mesmo número de linhas que X_encoded.
#Mesmo que esses zeros não representem rótulos reais, eles servem como "placeholders" para satisfazer os requisitos do método fit_transform. O seletor de características não usará de fato esse array de zeros para fazer previsões ou qualquer cálculo significativo; ele apenas precisa dele como uma entrada de rótulo.
#Em resumo, criar um array de zeros (y_dummy) é uma prática comum quando estamos usando transformadores que esperam características e rótulos como entrada, mas estamos realizando uma tarefa onde os rótulos não são relevantes para o processo de seleção de características. O array de zeros fornece uma estrutura que permite que o transformador seja usado de maneira adequada para alcançar o objetivo de selecionar as características relevantes.
y_dummy = np.zeros(X_encoded.shape[0])

# %%
# Inicializando o seletor de características
feature_seletor = SelectKBest(score_func=f_classif, k=num_features)

# Ajustando o seletor de características aos dados
X_new = feature_seletor.fit_transform(X_encoded, y_dummy)

# Obtendo as máscaras booleanas das características selecionadas
selected_mask = feature_seletor.get_support()
print("selected_mask:", selected_mask)

# Obtendo os nomes das características selecionadas
selected_features = X_encoded.columns[selected_mask]

# Imprimindo as características selecionadas
print("Características selecionadas:", selected_features)

# %%

# Inicializando o modelo de clusterização
model_cluster = cluster.KMeans(random_state=42, max_iter=1000,)

# Criando o pipeline
cluster_pipeline = pipeline.Pipeline([
    ('feature_selector', feature_seletor),
    ('model_cluster', model_cluster)
])

# Ajustando o pipeline aos dados
model_cluster.fit(X_encoded, y_dummy)
# %%
# Adicionando os rótulos de cluster de volta ao DataFrame original
df['cluster_labels'] = model_cluster.labels_

print(df)
# %% 
# Imprimindo a contagem de amostras em cada cluster
print(df['cluster_labels'].value_counts())

# %%
df.query("cluster_labels==0")

# %%
print(df.columns)

# %%
summary = df.groupby(['cluster_labels']).mean()
print(summary)

# %%
# Salvando o resumo dos clusters sócio-demográficos em um arquivo CSV
df.to_csv("summary.csv", sep=";")

# %%
# Criando um mapa de calor (heatmap) para o resumo dos clusters sócio-demográficos
seaborn.heatmap(summary[summary.columns[:-1]])
# %%

# %%

