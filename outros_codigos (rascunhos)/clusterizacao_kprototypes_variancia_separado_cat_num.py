# %% 
# Importar as bibliotecas necessárias
import pandas as pd
from sklearn import preprocessing
from feature_engine.encoding import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from yellowbrick.cluster import KElbowVisualizer
import seaborn
from kmodes.kprototypes import KPrototypes
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
cat_features.columns

# %% 
# Seleciona colunas que não são do tipo 'object'
num_features = df.select_dtypes(exclude=['object'])
num_features.columns

# %% 
# Codificação One-Hot de Variáveis Categóricas
onehot = OneHotEncoder(variables=cat_features.columns.tolist())
X_transform_cat = onehot.fit_transform(cat_features)

# Normalização Min-Max das variáveis numéricas
min_max_num = preprocessing.MinMaxScaler()
min_max_num.set_output(transform='pandas')
X_transform_num = min_max_num.fit_transform(num_features)

# Seleção de Características por Variância para as variáveis categóricas
var_feature_importance_cat = VarianceThreshold(0.1)
var_feature_importance_cat.set_output(transform='pandas')
X_transform_cat_filtered = var_feature_importance_cat.fit_transform(
    X_transform_cat)

# Seleção de Características por Variância para as variáveis numéricas
var_feature_importance_num = VarianceThreshold(0.01)
var_feature_importance_num.set_output(transform='pandas')
X_transform_num_filtered = var_feature_importance_num.fit_transform(
    X_transform_num)

# Concatenar as features categóricas e numéricas após a seleção por variância
X_transform_final = pd.concat([pd.DataFrame(
    X_transform_cat_filtered), pd.DataFrame(X_transform_num_filtered)], axis=1)

# Criação de uma instância do modelo K-Prototypes para realizar a clusterização
model_cluster = KPrototypes(n_clusters=5, n_init=10, verbose=2, random_state=42, max_iter=1000)

# Codificar manualmente as variáveis categóricas como numéricas
label_encoder = preprocessing.LabelEncoder()
for column in cat_features.columns:
    df[column] = label_encoder.fit_transform(df[column])

# Ajustar o modelo aos dados
model_cluster.fit(X_transform_final, categorical=list(range(len(cat_features.columns))))

# Atribuir rótulos de cluster ao DataFrame original
X_transform_final['cluster_name'] = model_cluster.labels_

# Estatísticas descritivas para cada cluster
summary = X_transform_final.groupby(['cluster_name']).mean()

# %%
# Visualização das estatísticas em um mapa de calor
seaborn.heatmap(summary, cmap='viridis')

# %%
# Criar subconjuntos de dados para cada cluster
clusters_data = []
for cluster_id in range(5):  # Defina o número de clusters desejado
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
from sklearn.metrics import silhouette_score

# Calcule o Silhouette Score para os clusters gerados pelo modelo
silhouette_avg = silhouette_score(X_transform_final.drop('cluster_name', axis=1), model_cluster.labels_)

# Imprima o valor do Silhouette Score
print(f'Silhouette Score: {silhouette_avg}')

#A interpretação do Silhouette Score pode ser feita da seguinte forma:
# Se o Silhouette Score estiver próximo de +1, os clusters estão bem separados.
# Se o Silhouette Score estiver próximo de 0, os clusters têm sobreposição.
# Se o Silhouette Score estiver próximo de -1, os pontos foram atribuídos ao cluster errado.
# %%
