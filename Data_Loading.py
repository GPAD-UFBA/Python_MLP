
# coding: utf-8

# In[1]:


from gpad_data_generation import *
import numpy as np
import scipy.io as sio
from Funcoes_de_EA import *
import pandas as pd
import os
import time
import pydub


# In[2]:


def convert_wav_to_mp3(filename):
    sound = AudioSegment.from_wav(filename)
    filename = filename[:-4] + ".mp3"
    soundwav = sound.export(filename, format="mp3")
    soundwav.close()
    return filename


# In[3]:


def generate_acm_mirum():

    acm_mirum_df = pd.DataFrame()

    acm_mirum = pd.read_csv("Banco de Dados/bpms_database/ACM_MIRUM.csv")

    acm_mirum = acm_mirum.rename(columns={'00000000':'filename', '00':'bpm'})


    list_of_files = os.listdir('Banco de Dados/acm_mirum_tempo')
    list_of_files = [file for file in list_of_files if file[-3:]=='mp3']

    start = time.time()
    i=0
    falhas=0
    total = len(list_of_files)
    for file in list_of_files:
        extension = file[-3:]

        if i==1410:
            continue

        if extension == 'wav':
            continue

        name = file[:-9]


        try:
            odf, pedf, ppedf, coefs = music_processor("Banco de Dados/acm_mirum_tempo/" +file)
            main_d, dicionario = ECA(pedf, wich_not=['v_pr', 'v_erf1'])
            bpm = acm_mirum.loc[acm_mirum.filename == int(name)].iloc[0][1]

            dicionario['bpm'] = bpm
            dicionario['filename'] = file
            dicionario['database'] = 'acm_mirum'

            dic_df = pd.DataFrame(dicionario, index=[i])
            acm_mirum_df = pd.concat([acm_mirum_df, dic_df], sort=False)

        except Exception as e:
            print('Erro no processamento do arquivo '+ file)
            print(e)
            print()
            falhas = falhas + 1


        i=i+1
        porcentagem = 100*i/total
        print(str(i)+'/'+str(total)+' arquivos processados, que significam: '+str(porcentagem)+'%', end='\r')

    end = time.time()
    acm_mirum_df.to_csv('Banco de Dados/atributos/acm_mirum_atributos.csv')

    print(f"{i-falhas} arquivos processados com sucesso.\nHouveram {falhas} arquivos corrompidos ou inexistentes.\nO tempo decorrido foi de {end-start} segundos.")
    
    return acm_mirum_df

def get_acm_mirum():
    return pd.read_csv('Banco de Dados/atributos/acm_mirum_atributos.csv')


# In[4]:


def generate_eball():

    extended_ballroom_df = pd.DataFrame()

    extended_ballroom = pd.read_csv("Banco de Dados/bpms_database/EBALL.csv")

    extended_ballroom = extended_ballroom.rename(columns={'00000/000000':'filename', '00.0':'bpm'})

    list_of_folders = os.listdir('Banco de Dados/Ballroom')




    i=0
    falhas=0
    total = len(extended_ballroom)
    start = time.time()

    for folder in list_of_folders:
        list_of_files = os.listdir('Banco de Dados/Ballroom/'+folder)


        for file in list_of_files:

            name = folder+'/'+file[:-4]


            try:
                odf, pedf, ppedf, coefs = music_processor("Banco de Dados/Ballroom/" + folder + '/' + file)
                main_d, dicionario = ECA(pedf, wich_not=['v_pr', 'v_erf1'])
                bpm = extended_ballroom.loc[extended_ballroom.filename == name].iloc[0][1]

                dicionario['bpm'] = bpm
                dicionario['filename'] = file
                dicionario['database'] = 'extended_ballroom'

                dic_df = pd.DataFrame(dicionario, index=[i])
                extended_ballroom_df = pd.concat([extended_ballroom_df, dic_df], sort=False)

            except Exception as e:
                #print('Erro no processamento do arquivo '+ file)
                #print(e)
                #print()
                falhas = falhas + 1
                continue


            i=i+1
            porcentagem = 100*i/total
            print(str(i)+'/'+str(total)+' arquivos processados, que significam: '+str(porcentagem)+'%', end='\r')

    end = time.time()
    extended_ballroom_df.to_csv('Banco de Dados/atributos/extended_ballroom_atributos.csv')

    print(f"{i-falhas} arquivos processados com sucesso.\nHouveram {falhas} arquivos corrompidos ou inexistentes.\nO tempo decorrido foi de {end-start} segundos.")
     
    return extended_ballroom_df

def get_eball():
    return pd.read_csv('Banco de Dados/atributos/extended_ballroom_atributos.csv')


# In[5]:


def generate_ismir():

    ismir2004_df = pd.DataFrame()

    ismir2004 = pd.read_csv('Banco de Dados/bpms_database/ISMIR2004.csv')

    ismir2004 = ismir2004.rename(columns={'0000':'filename', '000':'bpm'})

    list_of_folders = os.listdir('Banco de Dados/ismir2004_tempo')

    i=0
    falhas=0
    total = len(ismir2004)
    start = time.time()

    for folder in list_of_folders:
        list_of_files = os.listdir('Banco de dados/ismir2004_tempo/'+folder)

        for file in list_of_files:

            name = folder+'/'+file[:-4]


            try:
                new_file = convert_wav_to_mp3("Banco de Dados/ismir2004_tempo/" + folder + '/' + file)
                odf, pedf, ppedf, coefs = music_processor(new_file)
                main_d, dicionario = ECA(pedf, wich_not=['v_pr', 'v_erf1'])
                bpm = ismir2004.loc[ismir2004.filename == name].iloc[0][1]

                dicionario['bpm'] = bpm
                dicionario['filename'] = file
                dicionario['database'] = 'ismir2004'

                dic_df = pd.DataFrame(dicionario, index=[i])
                ismir2004_df = pd.concat([ismir2004_df, dic_df], sort=False)

            except Exception as e:
                #print('Erro no processamento do arquivo '+ file)
                #print(e)
                #print()
                falhas = falhas + 1
                continue


            i=i+1
            porcentagem = 100*i/total
            print(str(i)+'/'+str(total)+' arquivos processados, que significam: '+str(porcentagem)+'%', end='\r')

    end = time.time()
    ismir2004_df.to_csv('Banco de Dados/atributos/ismir2004_atributos.csv')

    print(f"{i-falhas} arquivos processados com sucesso.\nHouveram {falhas} arquivos corrompidos ou inexistentes.\nO tempo decorrido foi de {end-start} segundos.")
    
    return ismir2004_df

def get_ismir():
    return pd.read_csv('Banco de Dados/atributos/ismir2004_atributos.csv')


# In[6]:


def generate_hainsworth():

    hainsworth_df = pd.DataFrame()

    hainsworth = pd.read_csv('Banco de Dados/bpms_database/HAINSWORTH.csv')

    hainsworth = hainsworth.rename(columns={'wavs/001':'filename', '99.9563':'bpm'})


    list_of_files = os.listdir('Banco de Dados/hainsworth_tempo_mp3')
    #list_of_files = [file for file in list_of_files if file[-3:]=='mp3']

    start = time.time()
    i=0
    falhas=0
    total = len(list_of_files)
    for file in list_of_files:

        name = file[:-4]


        try:
            odf, pedf, ppedf, coefs = music_processor("Banco de Dados/hainsworth_tempo_mp3/" +file)
            main_d, dicionario = ECA(pedf, wich_not=['v_pr', 'v_erf1'])
            bpm = hainsworth.loc[hainsworth.filename == 'wavs/'+name].iloc[0][1]

            dicionario['bpm'] = bpm
            dicionario['filename'] = file
            dicionario['database'] = 'hainsworth'

            dic_df = pd.DataFrame(dicionario, index=[i])
            hainsworth_df = pd.concat([hainsworth_df, dic_df], sort=False)

        except Exception as e:
            print('Erro no processamento do arquivo '+ file)
            print(e)
            print()
            falhas = falhas + 1


        i=i+1
        porcentagem = 100*i/total
        print(str(i)+'/'+str(total)+' arquivos processados, que significam: '+str(porcentagem)+'%', end='\r')

    end = time.time()
    hainsworth_df.to_csv('Banco de Dados/atributos/hainsworth_atributos.csv')

    print(f"{i-falhas} arquivos processados com sucesso.\nHouveram {falhas} arquivos corrompidos ou inexistentes.\nO tempo decorrido foi de {end-start} segundos.")
    
    return hainsworth_df

def get_hainsworth():
    return pd.read_csv('Banco de Dados/atributos/hainsworth_atributos.csv')


# In[7]:


def generate_gtzan():

    gtzan_df = pd.DataFrame()

    gtzan = pd.read_csv('Banco de Dados/bpms_database/GTZAN_GENRES.csv')

    gtzan = gtzan.rename(columns={'00000000000':'filename', '000':'bpm'})


    list_of_files = os.listdir('Banco de Dados/gtzan_tempo')
    list_of_files = [file for file in list_of_files if file[-3:]=='wav']

    start = time.time()
    i=0
    falhas=0
    total = len(list_of_files)
    for file in list_of_files:

        name = file[:-4]

        new_file = convert_wav_to_mp3("Banco de Dados/gtzan_tempo/" + file)
        odf, pedf, ppedf, coefs = music_processor(new_file)
        main_d, dicionario = ECA(pedf, wich_not=['v_pr', 'v_erf1'])
        bpm = gtzan.loc[gtzan.filename == 'wavs/'+name].iloc[0][1]

        try:
            new_file = convert_wav_to_mp3("Banco de Dados/gtzan_tempo/" + file)
            odf, pedf, ppedf, coefs = music_processor(new_file)
            main_d, dicionario = ECA(pedf, wich_not=['v_pr', 'v_erf1'])
            bpm = gtzan.loc[gtzan.filename == 'wavs/'+name].iloc[0][1]

            dicionario['bpm'] = bpm
            dicionario['filename'] = file
            dicionario['database'] = 'gtzan'

            dic_df = pd.DataFrame(dicionario, index=[i])
            gtzan_df = pd.concat([gtzan_df, dic_df], sort=False)

        except Exception as e:
            print('Erro no processamento do arquivo '+ file)
            print(e)
            print()
            falhas = falhas + 1


        i=i+1
        porcentagem = 100*i/total
        print(str(i)+'/'+str(total)+' arquivos processados, que significam: '+str(porcentagem)+'%', end='\r')

    end = time.time()
    gtzan_df.to_csv('Banco de Dados/atributos/gtzan_atributos.csv')

    print(f"{i-falhas} arquivos processados com sucesso.\nHouveram {falhas} arquivos corrompidos ou inexistentes.\nO tempo decorrido foi de {end-start} segundos.")
    
    return gtzan_df

def get_gtzan():
    return pd.read_csv('Banco de Dados/atributos/gtzan_atributos.csv')


# In[8]:


def generate_smc_mirum():
    smc_mirum_df = pd.DataFrame()

    smc_mirum = pd.read_csv('Banco de Dados/bpms_database/SMC_MIRUM.csv')

    smc_mirum = smc_mirum.rename(columns={'0000000':'filename', '000000':'bpm'})


    list_of_files = os.listdir('Banco de Dados/smc_mirum_tempo')
    list_of_files = [file for file in list_of_files if file[-3:]=='wav']

    start = time.time()
    i=0
    falhas=0
    total = len(list_of_files)
    for file in list_of_files:

        name = file[:-4]

        try:
            new_file = convert_wav_to_mp3("Banco de Dados/smc_mirum_tempo/" + file)
            odf, pedf, ppedf, coefs = music_processor(new_file)
            main_d, dicionario = ECA(pedf, wich_not=['v_pr', 'v_erf1'])
            bpm = smc_mirum.loc[smc_mirum.filename == name].iloc[0][1]

            dicionario['bpm'] = bpm
            dicionario['filename'] = file
            dicionario['database'] = 'smc_mirum'

            dic_df = pd.DataFrame(dicionario, index=[i])
            smc_mirum_df = pd.concat([smc_mirum_df, dic_df], sort=False)

        except Exception as e:
            print('Erro no processamento do arquivo '+ file)
            print(e)
            print()
            falhas = falhas + 1


        i=i+1
        porcentagem = 100*i/total
        print(str(i)+'/'+str(total)+' arquivos processados, que significam: '+str(porcentagem)+'%', end='\r')

    end = time.time()
    smc_mirum_df.to_csv('Banco de Dados/atributos/smc_mirum_atributos.csv')

    print(f"{i-falhas} arquivos processados com sucesso.\nHouveram {falhas} arquivos corrompidos ou inexistentes.\nO tempo decorrido foi de {end-start} segundos.")
    
    return smc_mirum_df

def get_smc_mirum():
    return pd.read_csv('Banco de Dados/atributos/smc_mirum_atributos.csv')


# In[9]:


def generate_banco2():

    banco2_df = pd.DataFrame()

    banco2 = pd.read_excel('Banco de Dados/bpms_database/BPM Banco de Dados1e2.xlsx', sheet_name = 'Banco 2 Indexado')


    banco2 = pd.DataFrame(banco2['BPM Tapping'])
    banco2 = banco2.reset_index(level=1)
    banco2 = banco2[1:201]
    banco2 = banco2.rename(columns={'level_1':'filename', 'BPM Tapping': 'bpm'})


    list_of_files = os.listdir('Banco de Dados/Banco de Dados 2 V1')
    list_of_files = [file for file in list_of_files if file[-3:]=='wav']

    start = time.time()
    i=0
    falhas=0
    total = len(list_of_files)
    for file in list_of_files:

        name = file[:-4]

        try:
            new_file = convert_wav_to_mp3("Banco de Dados/Banco de Dados 2 V1/" + file)
            odf, pedf, ppedf, coefs = music_processor(new_file)
            main_d, dicionario = ECA(pedf, wich_not=['v_pr', 'v_erf1'])
            bpm = banco2.iloc[int(name)-1][1]

            dicionario['bpm'] = bpm
            dicionario['filename'] = file
            dicionario['database'] = 'banco2'

            dic_df = pd.DataFrame(dicionario, index=[i])
            banco2_df = pd.concat([banco2_df, dic_df], sort=False)

        except Exception as e:
            print('Erro no processamento do arquivo '+ file)
            print(e)
            print()
            falhas = falhas + 1


        i=i+1
        porcentagem = 100*i/total
        print(str(i)+'/'+str(total)+' arquivos processados, que significam: '+str(porcentagem)+'%', end='\r')

    end = time.time()
    banco2_df.to_csv('Banco de Dados/atributos/banco2_atributos.csv')

    print(f"{i-falhas} arquivos processados com sucesso.\nHouveram {falhas} arquivos corrompidos ou inexistentes.\nO tempo decorrido foi de {end-start} segundos.")
    
    return banco2_df

def get_banco2():
    banco2 = pd.read_csv('Banco de Dados/atributos/banco2_atributos.csv')
    return banco2


# In[10]:


def generate_banco1():

    banco1_df = pd.DataFrame()

    banco1 = pd.read_excel('Banco de Dados/bpms_database/BPM Banco de Dados1e2.xlsx', sheet_name = 'Banco 1 Corrigido')

    banco1 = pd.DataFrame(banco1['Unnamed: 1'], banco1['Unnamed: 5'])
    #banco1_bpm = pd.DataFrame(banco1['Unnamed: 5'])

    banco1 = banco1.reset_index(level=0)
    banco1 = banco1[1:308]
    banco1 = banco1.rename(columns={'Unnamed: 1':'filename', 'Unnamed: 5': 'bpm'})


    list_of_files = os.listdir('Banco de Dados/Banco de Dados 1 Corrigido')
    list_of_files = [file for file in list_of_files if file[-3:]=='wav']

    start = time.time()
    i=0
    falhas=0
    total = len(list_of_files)
    for file in list_of_files:

        name = file[:-4]

        try:
            new_file = convert_wav_to_mp3("Banco de Dados/Banco de Dados 1 Corrigido/" + file)
            odf, pedf, ppedf, coefs = music_processor(new_file)
            main_d, dicionario = ECA(pedf, wich_not=['v_pr', 'v_erf1'])
            bpm = banco1.iloc[int(name)-1][0]

            dicionario['bpm'] = bpm
            dicionario['filename'] = file
            dicionario['database'] = 'banco1'

            dic_df = pd.DataFrame(dicionario, index=[i])
            banco1_df = pd.concat([banco1_df, dic_df], sort=False)

        except Exception as e:
            print('Erro no processamento do arquivo '+ file)
            print(e)
            print()
            falhas = falhas + 1


        i=i+1
        porcentagem = 100*i/total
        print(str(i)+'/'+str(total)+' arquivos processados, que significam: '+str(porcentagem)+'%', end='\r')

    end = time.time()
    banco1_df.to_csv('Banco de Dados/atributos/banco1_atributos.csv')

    print(f"{i-falhas} arquivos processados com sucesso.\nHouveram {falhas} arquivos corrompidos ou inexistentes.\nO tempo decorrido foi de {end-start} segundos.")
    
    return banco1_df

def get_banco1():
    banco1 = pd.read_csv('Banco de Dados/atributos/banco1_atributos.csv')
    return banco1

