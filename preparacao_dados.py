# %%
import pandas as pd

# %%
df = pd.read_excel ('C:/Users/Mariana Moledo/Documents/GitHub/tcc_mba_cd/datasets/Dataset-UFF-Graduacao.xlsx')
df.shape

# %%
df.columns

# %%
# Filtragem de Dados
df = df.query('ACAOAFIRMATIVA != "ACAOAFIRMATIVA"')
df.shape

# %%
# Filtragem por Status de Formação
df_evadidos = df.query('STATUSFORMACAO == "EVADIDO"')
df_evadidos.shape

# %%
# Remoção de Duplicatas
df_sem_duplicatas = df_evadidos.drop_duplicates(subset='CODALUNO')
df_sem_duplicatas.shape

# %% [markdown]
# Um dos nossos objetivos é calcular a distância do cep de origem do aluno até o endereço do polo do curso. Nosso objetivo principal é entender se a distância pode também ter influenciado na evasão do aluno.
# Para isso, Precisaremos pegar o cep da localidade de cada curso. Utilizaremos o endereço do seguinte link:
# https://www.uff.br/?q=cursos-dados-completos&field_turno_value=All&field_titulacao_curso_value=All&field_tipo_de_curso_value=All&field_sigla_value=&field_c_digo_e_mec_value=

# %%
# Leitura de Outro Arquivo Excel (Cursos)

# Carregue o DataFrame df_cursos_iduff
df_cursos_iduff = pd.read_excel("C:/Users/Mariana Moledo/Documents/GitHub/tcc_mba_cd/datasets/CURSOS_IDUFF.xlsx")
df_cursos_iduff = df_cursos_iduff.rename(columns={"IDCURSO": "CURSO", "NOME": "NOME_CURSO"})
df_cursos_iduff = df_cursos_iduff[['CURSO', 'NOME_CURSO']]
df_cursos_iduff['NOME_CURSO'] = df_cursos_iduff['NOME_CURSO'].str.upper()

# Mesclar os DataFrames usando a coluna 'CURSO' como chave de junção
df_join_cursos_iduff = df_sem_duplicatas.merge(df_cursos_iduff, on='CURSO')

# %%
# Leitura de Arquivo CSV (CEPs de Destino)
df_ceps_destino = pd.read_csv("C:/Users/Mariana Moledo/Documents/GitHub/tcc_mba_cd/datasets/cep_destino_cursos.csv",sep=';', encoding='latin-1')

# %%
#Mesclagem com CEPs de Destino
df_join_cep_destino = df_join_cursos_iduff.merge(df_ceps_destino, on='NOME_CURSO')
df_join_cep_destino.columns
df_join_cep_destino.shape

# %%
# Leitura de Outro Arquivo CSV (Classificação de Cursos)
df_curso = pd.read_csv("C:/Users/Mariana Moledo/Documents/GitHub/tcc_mba_cd/datasets/Classificacao_cursos.csv",sep=';', encoding='latin-1')

# %%
# Mesclagem com Informações de Curso
# Utilizamos como base a classificação do arquivo http://www.coseac.uff.br/trm/2022/Arquivos/UFF-TRM2022-Anexo-14-Grupos.pdf
df_join_area_curso = df_join_cep_destino.merge(df_curso, on='NOME_CURSO')
df_join_area_curso = df_join_area_curso.rename(columns={"Classificação":"AREACURSO"})

# %%
# Manipulação de CEPs
# Contando o comprimento de cada valor na coluna
df_join_area_curso['comprimento'] = df_join_area_curso['CEP'].astype(str).str.len()

# Selecionando as linhas onde 'comprimento' é igual a 8
df_ceps_validos = df_join_area_curso[df_join_area_curso['comprimento'] == 8]

# Removendo a coluna 'comprimento' do df
del df_join_area_curso['comprimento']
df_ceps_validos.shape

#Renomeação e Criação de DataFrame Geral: 
df_geral = df_join_area_curso 
df_geral.columns

# %%
df_ceps_validos.shape

# %% [markdown]
# Agora temos 2 Dataframes: df_ceps_validos e df_geral.
# df_ceps_validos possui somente os alunos que tem o CEP válido.
# df_geral todos os alunos que evadiram (independente do CEP).
# Vamos calcular a distância do cep de origem do aluno até o endereço do polo do curso dos alunos que tem cep válido. Isso é feito no arquivo Distancia_API_Google.py

# %%
# Leitura de Outro Arquivo CSV (aqui já com as distâncias dos alunos com CEPs validos calculadas pela API do Google)
# Caso você ainda não tenha rodado a API do Google, essa etapa deve ser subsituída pelo calculo utilizando a API
df_com_distancia = pd.read_excel("C:/Users/Mariana Moledo/Documents/GitHub/tcc_mba_cd/datasets/Dataset-UFF-Graduacao-Evadidos-CEPs_Validos.xlsx")

# %%
# Selecionando colunas específicas
DF_DISTANCIA_CALCULADA = df_com_distancia[['CODALUNO','DISTANCIA_NUM']]
DF_DISTANCIA_CALCULADA.columns


# %%
# Mesclagem com Informações de distância

df_geral = df_geral.merge(DF_DISTANCIA_CALCULADA, on='CODALUNO', how='left')
df_geral.columns

# %%
# Salvando em um arquivo CSV a base principal tratada
df_geral.to_csv("bd_alunos_evadidos.csv", sep=";")


