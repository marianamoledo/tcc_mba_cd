# %%
import pandas as pd
from sklearn import feature_selection
from feature_engine.encoding import OneHotEncoder
from sklearn import preprocessing
from yellowbrick.cluster import KElbowVisualizer
import seaborn as sns
from sklearn import pipeline
from sklearn import cluster
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("C:/Users/Mariana Moledo/Documents/GitHub/tcc_mba_cd/datasets/bd_alunos_evadidos.csv",sep=';', encoding='utf-8')
df.columns

# %%
df.dtypes

# %% [markdown]
# Análise Inicial

# %%
# Exibir as primeiras linhas do DataFrame para entender a estrutura dos dados
print(df.head())

# %%
# Verificar tipos de dados e informações gerais
print(df.info())

# %%
# Estatísticas descritivas para variáveis numéricas
print(df.describe())

# %%
# Verificar a presença de valores ausentes
print(df.isnull().sum())

# %%
# Visualizar a distribuição de uma variável numérica (por exemplo, ENEMLINGUAGEM)
plt.figure(figsize=(8, 4))
plt.hist(df['IDADE'], bins=20)
plt.xlabel('IDADE')
plt.ylabel('Contagem')
plt.title('Distribuição de IDADE')
plt.show()


# %% [markdown]
# Exploração das Variáveis Categóricas

# %%
# Visualizar a distribuição de uma variável categórica (por exemplo, SEXO)
plt.figure(figsize=(6, 4))
df['SEXO'].value_counts().plot(kind='bar')
plt.xlabel('SEXO')
plt.ylabel('Contagem')
plt.title('Distribuição de SEXO')
plt.show()

# %%
#Frequência Estado Civil
frequencia_areacurso = df['AREACURSO'].value_counts()
print(frequencia_areacurso)

#Proporção Estado Civil
proporcao_areacurso = df['AREACURSO'].value_counts(normalize=True)
print(proporcao_areacurso)

#Gráfico de Barras
plt.figure(figsize=(20, 6))
sns.barplot(x=frequencia_areacurso.index, y=frequencia_areacurso.values)
plt.title('Gráfico de Barras - Contagem por AREACURSO') #Adicionando título ao gráfico
plt.xlabel('AREACURSO') #Adicionando rótulos aos eixos
plt.ylabel('Frequência') #Adicionando rótulos aos eixos

# %%
# Crie uma função para agrupar as categorias em "Exatas" ou "Humanas"
def agrupar_categorias(categoria):
    if categoria in ['Ciências Exatas e da Terra', 'Engenharias']:
        return 'Exatas'
    elif categoria in ['Ciências Sociais Aplicadas', 'Ciências Humanas']:
        return 'Humanas'
    else:
        return 'Outras'

# Aplique a função para criar uma nova coluna 'Grupo' no DataFrame
df['Grupo_area_curso'] = df['AREACURSO'].apply(agrupar_categorias)

# Visualize o DataFrame com a nova coluna 'Grupo'
print(df[['AREACURSO', 'Grupo_area_curso']])

# Use a função value_counts() para contar as ocorrências em cada grupo
contagem_por_grupo_area_curso = df['Grupo_area_curso'].value_counts()

# Exiba o resultado
print(contagem_por_grupo_area_curso)

# %%
#Frequência Estado Civil
frequencia_areafirmativa = df['ACAOAFIRMATIVA'].value_counts()
print(frequencia_areafirmativa)

#Proporção Estado Civil
proporcao_areafirmativa = df['ACAOAFIRMATIVA'].value_counts(normalize=True)
print(proporcao_areafirmativa)

#Gráfico de Barras
plt.figure(figsize=(20, 6))
sns.barplot(x=frequencia_areafirmativa.index, y=frequencia_areafirmativa.values)
plt.title('Gráfico de Barras - Contagem por AREACURSO') #Adicionando título ao gráfico
plt.xlabel('AREACURSO') # Adicionando rótulos aos eixos
plt.ylabel('Frequência') # Adicionando rótulos aos eixos

# %% [markdown]
# Vamos criar dois grupos: "Ampla Concorrência" e "Ações Afirmativas". As categorias "Ampla Concorrência" incluiriam aquelas em que não há critérios específicos de renda ou etnia, enquanto as categorias "Ações Afirmativas" incluiriam aquelas que têm critérios específicos de renda, etnia ou deficiência. 

# %%
# Crie uma função para agrupar as categorias
def categorias_to_grupo(categoria):
    if categoria in ['AC', 'A0']:
        return 'Ampla Concorrência'
    else:
        return 'Ações Afirmativas'

# Aplique a função para criar uma nova coluna 'Grupo' no DataFrame
df['Grupo_criterio'] = df['ACAOAFIRMATIVA'].apply(categorias_to_grupo)

# Visualize o DataFrame com a nova coluna 'Grupo'
print(df[['ACAOAFIRMATIVA', 'Grupo_criterio']])

# Use a função value_counts() para contar as ocorrências em cada grupo
contagem_por_grupo_criterio = df['Grupo_criterio'].value_counts()

# Exiba o resultado
print(contagem_por_grupo_criterio)

# %%

