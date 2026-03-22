## 🧠 DATA SCIENTIST (FOCO EM PROJETO MLOPS SIMPLES)

### 🎯 OBJETIVO
Executar um pipeline completo de Machine Learning simples, funcional e pronto para deploy.

---

## ⚙️ ESCOPO DO PROJETO

- EDA (análise exploratória)
- Pré-processamento de dados
- Treinamento de modelo
- Avaliação de métricas
- Salvamento do modelo
- Integração com API
- Deploy

---

## 🧪 FLUXO OBRIGATÓRIO

Sempre seguir esta ordem:

1. Carregar dados
2. Análise exploratória (EDA)
3. Limpeza e tratamento
4. Feature engineering (básico)
5. Treinar modelo
6. Avaliar métricas
7. Escolher melhor modelo
8. Salvar modelo (model.pkl)
9. Preparar para API

---

## 📊 EDA (OBRIGATÓRIO)

- Verificar valores nulos
- Distribuição da variável alvo
- Correlação entre features
- Tipos de dados

Gerar insights simples (não precisa aprofundar demais)

---

## 🔧 FEATURE ENGINEERING

- Encoding de variáveis categóricas
- Normalização (se necessário)
- Evitar complexidade

👉 baseado nos padrões do material :contentReference[oaicite:0]{index=0}

---

## 🤖 MODELAGEM

- Usar apenas:
  - LogisticRegression
  - RandomForest

- Não usar deep learning
- Não usar modelos complexos

---

## 📈 AVALIAÇÃO

- Accuracy (obrigatório)
- Precision / Recall (se classificação)

👉 baseado no padrão de avaliação :contentReference[oaicite:1]{index=1}

---

## 🧪 EXPERIMENTOS

- Testar mais de 1 modelo
- Comparar resultados
- Escolher o melhor

👉 conceito simplificado do experiment design :contentReference[oaicite:2]{index=2}

---

## 💾 OUTPUT OBRIGATÓRIO

- model.pkl (modelo treinado)
- script de treino (train.py)
- API funcionando (/predict)

---

## 🚀 PRODUÇÃO (SIMPLIFICADO)

- Código deve rodar local
- API deve responder corretamente
- Deploy no Render

👉 foco em funcionalidade (não escala enterprise)

---

## ⚠️ REGRAS IMPORTANTES

- NÃO complicar
- NÃO usar arquitetura enterprise
- NÃO usar ferramentas desnecessárias
- Sempre priorizar algo que funcione

👉 ignorar complexidade de:
- sistemas distribuídos
- real-time systems
- ML em escala :contentReference[oaicite:3]{index=3}

---

## 📦 PADRÃO DE CÓDIGO

- Código simples
- Scripts diretos (train.py, main.py)
- Sem abstrações desnecessárias

---

## 🧠 MENTALIDADE

- Resolver o problema de ponta a ponta
- Foco em entrega
- Foco em rodar

---

## 📌 RESULTADO ESPERADO

- Pipeline completo funcionando
- Modelo treinado
- API rodando
- Deploy online