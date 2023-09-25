# %% 
# Importar as bibliotecas necessárias
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from feature_engine.encoding import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Carregar os dados
file_path = "C:/Users/Mariana Moledo/Documents/GitHub/tcc_mba_cd/datasets/bd_alunos_evadidos.csv"
df = pd.read_csv(file_path, sep=';', encoding='utf-8')

# %% 
# Explorar os dados
print("Informações sobre as colunas:")
print(df.dtypes)

# %%
print("\nValores ausentes:")
print(df.isnull().sum())

# %%
# Exploração das Variáveis Categóricas
plt.figure(figsize=(20, 6))

# Frequência da variável 'AREACURSO'
frequencia_areacurso = df['AREACURSO'].value_counts()
print("\nFrequência da variável 'AREACURSO':")
print(frequencia_areacurso)

# Gráfico de Barras
sns.barplot(x=frequencia_areacurso.index, y=frequencia_areacurso.values)
plt.title('Gráfico de Barras - Contagem por AREACURSO')
plt.xlabel('AREACURSO')
plt.ylabel('Frequência')

# %%
# Função para agrupar as categorias em "Exatas" ou "Humanas"
def agrupar_categorias(categoria):
    if categoria in ['Ciências Exatas e da Terra', 'Engenharias']:
        return 'Exatas'
    elif categoria in ['Ciências Sociais Aplicadas', 'Ciências Humanas']:
        return 'Humanas'
    else:
        return 'Outras'
    
# Criar uma nova coluna 'Grupo_area_curso'
df['Grupo_area_curso'] = df['AREACURSO'].apply(agrupar_categorias)
contagem_por_grupo_area_curso = df['Grupo_area_curso'].value_counts()
print("\nContagem por grupo 'Grupo_area_curso':")
print(contagem_por_grupo_area_curso)

# %%
# Frequência da variável 'ACAOAFIRMATIVA'
frequencia_areafirmativa = df['ACAOAFIRMATIVA'].value_counts()
print("\nFrequência da variável 'ACAOAFIRMATIVA':")
print(frequencia_areafirmativa)

# %%
# Gráfico de Barras
plt.figure(figsize=(20, 6))
sns.barplot(x=frequencia_areafirmativa.index, y=frequencia_areafirmativa.values)
plt.title('Gráfico de Barras - Contagem por ACAOAFIRMATIVA')
plt.xlabel('ACAOAFIRMATIVA')
plt.ylabel('Frequência')

# %%
# Função para agrupar as categorias em "Ampla Concorrência" ou "Ações Afirmativas"
def categorias_to_grupo(categoria):
    if categoria in ['AC', 'A0']:
        return 'Ampla Concorrência'
    else:
        return 'Ações Afirmativas'

# Criar uma nova coluna 'Grupo_criterio'
df['Grupo_criterio'] = df['ACAOAFIRMATIVA'].apply(categorias_to_grupo)
contagem_por_grupo_criterio = df['Grupo_criterio'].value_counts()
print("\nContagem por grupo 'Grupo_criterio':")
print(contagem_por_grupo_criterio)

# %%
# Exclusão de Colunas
colunas_excluir = ['CODALUNO', 'STATUSFORMACAO', 'CR', 'CURSO', 'CODTURNOINGRESSO', 'CODTURNOATUAL',
                   'DISCIPLINA', 'NOTADISC', 'RESULTDISC', 'PERIODODISC', 'ANODESVINCULACAO',
                   'SEMESTREDESVINCULACAO', 'BAIRRO', 'CEP', 'CIDADE', 'CHCURSADA', 'TRANCAMENTOS',
                   'TEMPOPERMANENCIA', 'NOME_CURSO', 'cep_destino', 'MOBILIDADE', 'Unnamed: 0',
                   'DISTANCIA_NUM', 'ACAOAFIRMATIVA', 'AREACURSO', 'TURNOATUAL']

df = df.drop(colunas_excluir, axis=1)

# %%
# Codificação One-Hot de Variáveis Categóricas
cat_features = ['Grupo_criterio', 'SEMESTREINGRESSO', 'COR', 'ESTADOCIVIL', 'SEXO', 'Grupo_area_curso']
df[cat_features] = df[cat_features].astype(str)
onehot = OneHotEncoder(variables=cat_features)
X_transform = onehot.fit_transform(df)
X_transform

# %%
# Normalização Min-Max das variáveis numéricas
min_max = MinMaxScaler()
min_max.set_output(transform='pandas')
X_transform = min_max.fit_transform(X_transform)
X_transform

# %%
# Seleção de Características por Variância para as variáveis categóricas
var_feature_importance = VarianceThreshold(0.21)
var_feature_importance.set_output(transform='pandas')
X_transform_filtered = var_feature_importance.fit_transform(X_transform)
X_transform_filtered

# %%
# Encontrar o número ideal de clusters usando KElbowVisualizer
model_cluster = KMeans(random_state=42, max_iter=1000)
visualizer = KElbowVisualizer(model_cluster, k=(2, 12))
visualizer.fit(X_transform_filtered)
visualizer.show()

# %%
# Criar e ajustar o modelo KMeans com o número ideal de clusters
model_cluster = KMeans(n_clusters=visualizer.elbow_value_)
model_cluster.fit(X_transform_filtered)
cluster_labels = model_cluster.predict(X_transform_filtered)

# %%
# Adicionar os rótulos dos clusters ao DataFrame
X_transform_filtered['cluster_name'] = cluster_labels

# Estatísticas descritivas para cada cluster
summary = X_transform_filtered.groupby(['cluster_name']).mean()

# %%
# Criar um mapa de calor das estatísticas dos clusters
plt.figure(figsize=(10, 6))
sns.heatmap(summary, annot=True, cmap='viridis', fmt=".2f")
plt.xlabel('Estatísticas')
plt.ylabel('Clusters')
plt.title('Mapa de Calor das Estatísticas dos Clusters')
plt.show()

# %%
# Criar subconjuntos de dados para cada cluster
clusters_data = []
for cluster_id in range(visualizer.elbow_value_):
    cluster_data = X_transform_filtered[X_transform_filtered['cluster_name'] == cluster_id].drop('cluster_name', axis=1)
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
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], label=f'Cluster {cluster_id}')

plt.legend()
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Visualização dos Clusters Após PCA')
plt.show()

# %%
# Calcular o Silhouette Score para os clusters
silhouette_avg = silhouette_score(X_transform_filtered.drop('cluster_name', axis=1), cluster_labels)
print(f'Silhouette Score: {silhouette_avg}')

# %%
