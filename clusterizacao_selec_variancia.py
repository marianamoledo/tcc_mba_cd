# %%
import pandas as pd
from sklearn import feature_selection
from feature_engine.encoding import OneHotEncoder
from sklearn import preprocessing
from yellowbrick.cluster import KElbowVisualizer
import seaborn
from sklearn import pipeline
from sklearn import cluster
import matplotlib.pyplot as plt

# %%
# Definição de Opções de Exibição do Pandas:

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)

# %%
# Carregamento de Dados
df = pd.read_csv("C:/Users/Mariana Moledo/Documents/GitHub/tcc_mba_cd/datasets/bd_alunos_evadidos.csv",sep=';', encoding='utf-8')
df.columns

# %%
# Exclusão de Colunas

colunas_excluir = ['CODALUNO',
                   'STATUSFORMACAO',
                   'CR',
                   'CURSO',
                   'CODTURNOINGRESSO',
                   'CODTURNOATUAL',
                   'DISCIPLINA',
                   'NOTADISC',
                   'RESULTDISC',
                   'PERIODODISC',
                   'ANODESVINCULACAO',
                   'SEMESTREDESVINCULACAO',
                   'BAIRRO',
                   'CEP',
                   'CIDADE',
                   'CHCURSADA',
                   'TRANCAMENTOS',
                   'TEMPOPERMANENCIA',
                   'NOME_CURSO',
                   'cep_destino',
                   'MOBILIDADE',
                   'Unnamed: 0'
                   ]

# %%
df = df.drop(colunas_excluir, axis=1)

# %%
df.columns

# %%
# Codificação One-Hot de Variáveis Categóricas
cat_features = ['ACAOAFIRMATIVA',
                'TURNOATUAL',
                'ANOINGRESSO',
                'SEMESTREINGRESSO',
                'COR',
                'ESTADOCIVIL',
                'SEXO',
                'AREACURSO']

df[cat_features] = df[cat_features].astype(str)

onehot = OneHotEncoder(variables=cat_features)
X_transform = onehot.fit_transform(df)

# %%
X_transform.shape

# %%
# Normalização Min-Max
# coloca todas as características numéricas no intervalo de 0 a 1
min_max = preprocessing.MinMaxScaler()
min_max.set_output(transform='pandas')


X_transform = min_max.fit_transform(X_transform)

X_transform

# %%
# Seleção de Características por Variância:

var_feature_importance = feature_selection.VarianceThreshold(0.03)
var_feature_importance.set_output(transform='pandas')
X_transform_filter = var_feature_importance.fit_transform(X_transform)
X_transform_filter

# %%
X_transform_filter.var().reset_index().T

# %%
X_transform_filter.shape

# %%
# criação de uma instância do modelo KMeans para realizar a clusterização.
model_cluster = cluster.KMeans(random_state=42, max_iter=1000,)
model_cluster


# %%
# Verifique se há NaNs no DataFrame
nan_check = X_transform_filter.isna()

# Filtre as linhas com True
rows_with_nan = X_transform_filter[nan_check.any(axis=1)]

# Exiba as linhas que contêm NaNs
print(rows_with_nan)



# %%
visualizer = KElbowVisualizer(model_cluster, k=(2, 12))
visualizer.fit(X_transform_filter)
visualizer.show()

# %%
model_cluster = cluster.KMeans(n_clusters=visualizer.elbow_value_)

# %%
model_cluster.fit(X_transform_filter)


# %%
cluster_labels = model_cluster.predict(X_transform_filter)
cluster_labels


# %%
X_transform_filter['cluster_name'] = model_cluster.labels_
X_transform_filter


# %%
# Estatísticas descritivas para cada cluster
estatisticas_clusters = X_transform_filter.groupby('cluster_name').agg(['mean', 'median', 'std', 'count'])
print(estatisticas_clusters)



