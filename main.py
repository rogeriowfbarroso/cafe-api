from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

# 1. INICIALIZAÇÃO DA API
app = FastAPI(title="API de Predição de Café - Tese de Doutorado")

# 2. CONFIGURAÇÃO DE CORS (Permite que a sua interface Web acesse a API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rogeriowfbarroso.github.io/agroclima/"],  # * Aceita requisições de qualquer site
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. CARREGAMENTO DOS ARQUIVOS DO MODELO
print("A iniciar o carregamento do modelo de Inteligência Artificial...")
try:
    modelo_rf = joblib.load('modelo_random_forest_cafe.pkl')
    colunas_corretas = joblib.load('colunas_do_modelo.pkl')
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Erro Crítico ao carregar arquivos .pkl: {e}")

# 4. ESTRUTURA DE DADOS ESPERADA DO FRONT-END (Validação Pydantic)
# Aqui estão todas as variáveis exatas que o seu .pkl exige.
class DadosRequisicao(BaseModel):
    # Clima (listas de 12 meses, de Janeiro a Dezembro)
    t2m: list[float]          
    tmax: list[float]         
    tmin: list[float]         
    rh2m: list[float]         
    prectotcorr: list[float]  
    ps: list[float]           # Pressão Superficial
    ws10m: list[float]        # Velocidade do Vento a 10m
    
    # Solo (valores únicos por coordenada)
    Argila: float             
    Nitrogenio: float         
    OCD: float               
    OCS: float               
    PH: float                
    SOC: float               

# 5. ROTA PRINCIPAL: PREVISÃO
@app.post("/prever")
def fazer_previsao(dados: DadosRequisicao):
    try:
        dados_recebidos = dados.dict()
        
        # Variáveis climáticas que precisam ser desdobradas em 12 meses
        variaveis_clima = ['t2m', 'tmax', 'tmin', 'rh2m', 'prectotcorr', 'ps', 'ws10m']
        meses = ["Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho", 
                 "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"]
        
        dados_para_o_modelo = {}
        
        # Mapeando dinamicamente o clima (Ex: tmax_Janeiro, tmax_Fevereiro...)
        for var in variaveis_clima:
            valores_mensais = dados_recebidos[var]
            
            # Trava de segurança: garantir que o Front-end enviou exatamente 12 meses
            if len(valores_mensais) != 12:
                raise HTTPException(status_code=400, detail=f"A variável {var} precisa ter exatamente 12 valores.")
            
            for i, mes in enumerate(meses):
                nome_coluna = f"{var}_{mes}"
                dados_para_o_modelo[nome_coluna] = valores_mensais[i]
                
        # Inserindo diretamente os dados de Solo (Nomes exatos do seu .pkl)
        dados_para_o_modelo['Argila'] = dados_recebidos['Argila']
        dados_para_o_modelo['Nitrogenio'] = dados_recebidos['Nitrogenio']
        dados_para_o_modelo['OCD'] = dados_recebidos['OCD']
        dados_para_o_modelo['OCS'] = dados_recebidos['OCS']
        dados_para_o_modelo['PH'] = dados_recebidos['PH']
        dados_para_o_modelo['SOC'] = dados_recebidos['SOC']
        
        # Converte para Pandas
        df_novo = pd.DataFrame([dados_para_o_modelo])
        
        # REINDEX: O Segredo de Ouro. Garante que as colunas fiquem na ordem exata do treinamento
        df_novo = df_novo.reindex(columns=colunas_corretas, fill_value=0)
        
        # Executa a previsão usando o Random Forest
        previsao = modelo_rf.predict(df_novo)[0]
        
        # Retorna o resultado JSON para o Front-end
        return {
            "status": "sucesso",
            "produtividade_estimada_kg_ha": round(previsao, 2),
            "mensagem": "Predição realizada com sucesso pelo modelo Random Forest."
        }

    except Exception as e:
        # Se algo falhar na construção dos dados, avisa a interface
        raise HTTPException(status_code=500, detail=str(e))

# 6. ROTA DE STATUS (Para testar se o Render está no ar)
@app.get("/")
def raiz():
    return {
        "status": "Online", 
        "projeto": "Sistema Integrado de Suporte à Decisão - Café Arábica",
        "autor": "Rogério W F Barroso - Tese de Doutorado em Agricultura Sustentável",
        "mensagem": "Bem-vindo à API de Predição de Café! Use a rota /prever para obter estimativas de produtividade."
    }