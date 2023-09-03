# %%
import pandas as pd

# %%
df = pd.read_excel ('C:/Users/Mariana Moledo/Documents/Dataset-UFF-Graduacao.xlsx')
df.shape

# %%
df.columns

# %%
df = df.query('ACAOAFIRMATIVA != "ACAOAFIRMATIVA"')
df.shape

# %%
df_evadidos = df.query('STATUSFORMACAO == "EVADIDO"')
df_evadidos.shape

# %%
# Remove duplicates
df_sem_duplicatas = df_evadidos.drop_duplicates(subset='CODALUNO')
df_sem_duplicatas.shape

# %% [markdown]
# Um dos nossos objetivos é calcular a distância do cep de origem do aluno até o endereço do polo do curso. Nosso objetivo principal é entender se a distância pode também ter influenciado na evasão do aluno.
# Para isso, Precisaremos pegar o cep da localidade de cada curso. Utilizaremos o endereço do seguinte link:
# https://www.uff.br/?q=cursos-dados-completos&field_turno_value=All&field_titulacao_curso_value=All&field_tipo_de_curso_value=All&field_sigla_value=&field_c_digo_e_mec_value=

# %%
import pandas as pd

# Carregue o DataFrame df_cursos_iduff
df_cursos_iduff = pd.read_excel("CURSOS_IDUFF.xlsx")
df_cursos_iduff = df_cursos_iduff.rename(columns={"IDCURSO": "CURSO", "NOME": "NOME_CURSO"})
df_cursos_iduff = df_cursos_iduff[['CURSO', 'NOME_CURSO']]
df_cursos_iduff['NOME_CURSO'] = df_cursos_iduff['NOME_CURSO'].str.upper()

# Mesclar os DataFrames usando a coluna 'CURSO' como chave de junção
df_join_cursos_iduff = df_sem_duplicatas.merge(df_cursos_iduff, on='CURSO')

# %%
df_ceps_destino = pd.read_csv("cep_destino_cursos.csv",sep=';', encoding='latin-1')

# %%
df_join_cep_destino = df_join_cursos_iduff.merge(df_ceps_destino, on='NOME_CURSO')
df_join_cep_destino.columns
df_join_cep_destino.shape

# %%
df_curso = pd.read_csv("Classificacao_cursos.csv",sep=';', encoding='latin-1')

# %%
df_join_area_curso = df_join_cep_destino.merge(df_curso, on='NOME_CURSO')
df_join_area_curso = df_join_area_curso.rename(columns={"Classificação":"AREACURSO"})

# %%
# Use a função str.len() para contar o comprimento de cada valor na coluna
df_join_area_curso['comprimento'] = df_join_area_curso['CEP'].astype(str).str.len()

# Agora, selecione as linhas onde 'comprimento' é igual a 8
df_ceps_validos = df_join_area_curso[df_join_area_curso['comprimento'] == 8]

# Remova a coluna 'comprimento' se não for mais necessária
del df_join_area_curso['comprimento']

df_ceps_validos.shape
df_geral = df_join_area_curso 
df_geral.columns

# %% [markdown]
# Agora temos 2 Dataframes: df_ceps_validos e df_geral.
# df_ceps_validos possui somente os alunos que tem o CEP válido.
# df_geral todos os alunos que evadiram (independente do CEP).
# Vamos calcular a distância do cep de origem do aluno até o endereço do polo do curso dos alunos que tem cep válido. Isso é feito no arquivo Distancia_API_Google.py

# %%
df_com_distancia = pd.read_excel("Dataset-UFF-Graduacao-Evadidos-CEPs_Validos.xlsx")

# %%
DF_DISTANCIA_CALCULADA = df_com_distancia[['CODALUNO','DISTANCIA_NUM']]
DF_DISTANCIA_CALCULADA.columns


# %%
df_geral = df_geral.merge(DF_DISTANCIA_CALCULADA, on='CODALUNO', how='left')
df_geral.columns

# %%
# Salvando em um arquivo CSV
df_geral.to_csv("bd_alunos_evadidos.csv", sep=";")


