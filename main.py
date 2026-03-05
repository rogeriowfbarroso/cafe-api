from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

# 1. INICIALIZA A API
app = FastAPI(title="API de Predição de Café - Tese de Doutorado")

# 2. CONFIGURAÇÃO DE CORS (Permite que seu site acesse a API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rogeriowfbarroso.github.io/agroclima/"],  # Na produção, você pode trocar "*" pelo link do seu site
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. CARREGAMENTO DOS MODELOS (Executa uma vez quando a API liga)
print("Carregando modelo de Inteligência Artificial...")
try:
    modelo_rf = joblib.load('modelo_random_forest_cafe.pkl')
    colunas_corretas = joblib.load('colunas_do_modelo.pkl')
    # scaler = joblib.load('padronizador_dados.pkl') # Se usar SVM/MLP no futuro
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar arquivos .pkl: {e}")

# 4. DEFINIÇÃO DA ESTRUTURA DE ENTRADA (O que o Front-end vai enviar)
# O Pydantic garante que a API só aceite os dados se vierem no formato correto
class DadosRequisicao(BaseModel):
    t2m: list[float]          # Lista com 12 valores (Janeiro a Dezembro)
    tmax: list[float]         # Lista com 12 valores
    tmin: list[float]         # Lista com 12 valores
    rh2m: list[float]         # Lista com 12 valores
    prectotcorr: list[float]  # Lista com 12 valores
    ph_solo: float            # Valor único
    argila_solo: float        # Valor único
    Nitrogenio_solo: float    # Valor único
    OCD_solo: float           # Valor único
    OCS_solo: float           # Valor único
    SOC_solo: float           # Valor único
    
# 5. ROTA DE PREVISÃO (Onde a mágica acontece)
@app.post("/prever")
def fazer_previsao(dados: DadosRequisicao):
    try:
        # A. Converte os dados recebidos para um dicionário Python
        dados_recebidos = dados.dict()
        
        # B. Separa o que é clima (listas mensais) do que é solo (valores únicos)
        variaveis_clima = ['t2m', 'tmax', 'tmin', 'rh2m', 'prectotcorr']
        meses = ["Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho", 
                 "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"]
        
        dados_para_o_modelo = {}
        
        # C. Monta as colunas climáticas (Ex: tmax_Janeiro)
        for var in variaveis_clima:
            valores_mensais = dados_recebidos[var]
            if len(valores_mensais) != 12:
                raise HTTPException(status_code=400, detail=f"A variável {var} precisa ter exatamente 12 meses.")
            
            for i, mes in enumerate(meses):
                nome_coluna = f"{var}_{mes}"
                dados_para_o_modelo[nome_coluna] = valores_mensais[i]
                
        # D. Adiciona as colunas de solo
        dados_para_o_modelo['ph_solo'] = dados_recebidos['ph_solo']
        dados_para_o_modelo['argila_solo'] = dados_recebidos['argila_solo']
        dados_para_o_modelo['argila_solo'] = dados_recebidos['argila_solo']
        dados_para_o_modelo['Nitrogenio_solo'] = dados_recebidos['Nitrogenio_solo']
        dados_para_o_modelo['OCD_solo'] = dados_recebidos['OCD_solo']
        dados_para_o_modelo['OCS_solo'] = dados_recebidos['OCS_solo']
        dados_para_o_modelo['SOC_solo'] = dados_recebidos['SOC_solo']

        # E. Converte para Pandas e alinha as colunas com as do treinamento
        df_novo = pd.DataFrame([dados_para_o_modelo])
        df_novo = df_novo.reindex(columns=colunas_corretas, fill_value=0)
        
        # F. Faz a predição!
        previsao = modelo_rf.predict(df_novo)[0]
        
        # G. Retorna o resultado para o Front-end em formato JSON
        return {
            "status": "sucesso",
            "produtividade_estimada_kg_ha": round(previsao, 2),
            "mensagem": "Predição realizada com base em Inteligência Artificial."
        }

    except Exception as e:
        # Se algo der errado, avisa o Front-end
        raise HTTPException(status_code=500, detail=str(e))

# 6. ROTA DE TESTE (Para ver se a API está online no Render)
@app.get("/")
def raiz():
    return {"status": "Online", "projeto": "Predição de Café - API"}