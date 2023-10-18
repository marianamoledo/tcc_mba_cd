# %% 
import pandas as pd

# Carregando o arquivo
df = pd.read_excel('C:/Users/Mariana Moledo/Documents/GitHub/tcc_mba_cd/datasets/Dataset-UFF-Graduacao.xlsx')

# Removendo a linha que contém "ACAOAFIRMATIVA" na coluna "ACAOAFIRMATIVA"
df = df[df['ACAOAFIRMATIVA'] != "ACAOAFIRMATIVA"]

# Filtrando os alunos que possuem "STATUSFORMACAO" igual a "EVADIDO"
df_evadidos = df[df['STATUSFORMACAO'] == "EVADIDO"]

# %% 
# Removendo duplicatas com base na coluna "CODALUNO"
df_sem_duplicatas = df_evadidos.drop_duplicates(subset='CODALUNO')

df_sem_duplicatas.shape
# %% 
# Carregando o arquivo de cursos
df_cursos_iduff = pd.read_excel('C:/Users/Mariana Moledo/Documents/GitHub/tcc_mba_cd/datasets/CURSOS_IDUFF.xlsx')
df_cursos_iduff = df_cursos_iduff.rename(columns={"IDCURSO": "CURSO", "NOME": "NOME_CURSO"})
df_cursos_iduff = df_cursos_iduff[['CURSO', 'NOME_CURSO']]
df_cursos_iduff['NOME_CURSO'] = df_cursos_iduff['NOME_CURSO'].str.upper()
df_cursos_iduff.shape

# Mesclando os DataFrames com base na coluna "CURSO"
df_join_cursos_iduff = df_sem_duplicatas.merge(df_cursos_iduff, on='CURSO')

# %% 
df_join_cursos_iduff.shape
valores_nulos = df_join_cursos_iduff['NOME_CURSO'].isnull()
quantidade_nulos = valores_nulos.sum()
print(quantidade_nulos)
#%% 
# Carregando o arquivo CSV de classificação de cursos
df_curso = pd.read_csv("C:/Users/Mariana Moledo/Documents/GitHub/tcc_mba_cd/datasets/Classificacao_cursos.csv", sep=';', encoding='latin-1')

# Mesclando os DataFrames com base na coluna "NOME_CURSO"
df_join_area_curso = df_join_cursos_iduff.merge(df_curso, on='NOME_CURSO',how='outer')
df_join_area_curso.shape

#%%
df_join_area_curso = df_join_area_curso.rename(columns={"Classificação": "AREACURSO"})

#%%
# Renomeando o DataFrame para df_geral
df_geral = df_join_area_curso

df_geral.shape
# %% 
# Salvando o DataFrame em um arquivo CSV
file_path = "datasets/bd_alunos_evadidos.csv"
df_geral.to_csv(file_path, sep=";")
# %%
