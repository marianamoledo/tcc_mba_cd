# %%
# Importar as bibliotecas necessárias
import pandas as pd
from sklearn import preprocessing
from feature_engine.encoding import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from yellowbrick.cluster import KElbowVisualizer
import seaborn
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Definição de Opções de Exibição do Pandas:
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)

# Carregamento de Dados
df = pd.read_csv(
    "C:/Users/Mariana Moledo/Documents/GitHub/tcc_mba_cd/datasets/bd_alunos_evadidos.csv", sep=';', encoding='utf-8')

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
                   'Unnamed: 0',
                   'DISTANCIA_NUM'
                   ]

df = df.drop(colunas_excluir, axis=1)

df['TURNOATUAL'] = df['TURNOATUAL'].fillna("TURNOATUAL_NAODEFINIDO")
df['ANOINGRESSO'] = df['ANOINGRESSO'].astype(str)
df['SEMESTREINGRESSO'] = df['SEMESTREINGRESSO'].astype(str)


# %%
# Separar variáveis categóricas e numéricas
# Seleciona colunas do tipo 'object'
cat_features = df.select_dtypes(include=['object'])
# Seleciona colunas que não são do tipo 'object'
num_features = df.select_dtypes(exclude=['object'])

# %%
# Codificação One-Hot de Variáveis Categóricas
onehot = OneHotEncoder(variables=cat_features.columns.tolist())
X_transform_cat = onehot.fit_transform(cat_features)
X_transform_cat

# %%
# Normalização Min-Max das variáveis numéricas
min_max_num = preprocessing.MinMaxScaler()
min_max_num.set_output(transform='pandas')

X_transform_num = min_max_num.fit_transform(num_features)
X_transform_num

# %%
# Seleção de Características por Variância para as variáveis categóricas
var_feature_importance_cat = VarianceThreshold(0.15)
var_feature_importance_cat.set_output(transform='pandas')
X_transform_cat_filtered = var_feature_importance_cat.fit_transform(
    X_transform_cat)
X_transform_cat_filtered

# %%
# Seleção de Características por Variância para as variáveis numéricas
var_feature_importance_num = VarianceThreshold(0.01)
var_feature_importance_num.set_output(transform='pandas')
X_transform_num_filtered = var_feature_importance_num.fit_transform(
    X_transform_num)
X_transform_num_filtered

# %%
# Concatenar as features categóricas e numéricas após a seleção por variância
X_transform_final = pd.concat([pd.DataFrame(
    X_transform_cat_filtered), pd.DataFrame(X_transform_num_filtered)], axis=1)
X_transform_final
# %%
# Criação de uma instância do modelo KMeans para realizar a clusterização
model_cluster = KMeans(random_state=42, max_iter=1000)
model_cluster

# %%
# Verifique se há NaNs no DataFrame
nan_check = X_transform_final.isna()


# Filtre as linhas com True
rows_with_nan = X_transform_final[nan_check.any(axis=1)]
rows_with_nan
# %%
# Visualização do gráfico de cotovelo para escolher o número ideal de clusters
visualizer = KElbowVisualizer(model_cluster, k=(2, 12))
visualizer.fit(X_transform_final)
visualizer.show()

# %%
# Criar o modelo KMeans com o número ideal de clusters
model_cluster = KMeans(n_clusters=visualizer.elbow_value_)
model_cluster.fit(X_transform_final)

# %%
# Atribuir rótulos de cluster ao DataFrame original
X_transform_final['cluster_name'] = model_cluster.labels_
X_transform_final
# %%
# Estatísticas descritivas para cada cluster
summary = X_transform_final.groupby(['cluster_name']).mean()
summary
# %%
# Visualização das estatísticas em um mapa de calor
seaborn.heatmap(summary, cmap='viridis')

# %%
# Criar subconjuntos de dados para cada cluster
clusters_data = []
for cluster_id in range(visualizer.elbow_value_):
    cluster_data = X_transform_final[X_transform_final['cluster_name'] == cluster_id].drop(
        'cluster_name', axis=1)
    clusters_data.append(cluster_data)

# Aplicar PCA para cada subconjunto de dados do cluster
pca_models = []
for cluster_data in clusters_data:
    pca = PCA(n_components=2)
    pca.fit(cluster_data)
    pca_models.append(pca)

# Plotar os resultados do PCA para cada cluster
for cluster_id, pca_model in enumerate(pca_models):
    cluster_data = clusters_data[cluster_id]
    reduced_data = pca_model.transform(cluster_data)

    plt.scatter(reduced_data[:, 0], reduced_data[:, 1],
                label=f'Cluster {cluster_id}')

plt.legend()
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Visualização dos Clusters Após PCA')
plt.show()

# %%
