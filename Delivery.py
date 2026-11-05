import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# ==========================================
# 1. CARREGAMENTO E UNIFICAÇÃO DOS DADOS
# ==========================================
# (Ajuste os caminhos conforme seus arquivos locais)
df_orders = pd.read_csv('olist_orders_dataset.csv')
df_items = pd.read_csv('olist_order_items_dataset.csv')

# Conversão de colunas de data
date_cols = ['order_purchase_timestamp', 'order_estimated_delivery_date',
             'order_delivered_customer_date', 'order_delivered_carrier_date']
for col in date_cols:
  df_orders[col] = pd.to_datetime(df_orders[col])

# Unificação das tabelas pelo order_id
df = pd.merge(df_orders, df_items, on='order_id', how='inner')

# ==========================================
# 2. DEFINIÇÃO DO TARGET E TRATAMENTO DE NULOS
# ==========================================
# O modelo prevê no momento da postagem (carrier date)
df = df.dropna(
    subset=['order_delivered_customer_date', 'order_delivered_carrier_date']
)

# Target Binário: 1 se a entrega real ultrapassou a data estimada, 0 se foi no prazo
df['atrasado'] = (
    df['order_delivered_customer_date'] > df['order_estimated_delivery_date']
).astype(int)

# ==========================================
# 3. ENGENHARIA DE FEATURES (Sem Data Leakage)
# ==========================================
# Extraindo atributos temporais da data da compra
df['dia_semana_compra'] = df['order_purchase_timestamp'].dt.dayofweek
df['hora_compra'] = df['order_purchase_timestamp'].dt.hour

# Seleção das variáveis explicativas
features_num = ['price', 'freight_value', 'dia_semana_compra', 'hora_compra']
features_cat = ['order_status']  # Adicione 'customer_state' se disponível no seu merge

# Aplicando One-Hot Encoding nas categóricas
df_encoded = pd.get_dummies(df[features_cat], drop_first=True, dtype=int)

# Matriz X e Vetor y
X = pd.concat([df[features_num], df_encoded], axis=1)
y = df['atrasado']

# ==========================================
# 4. DIVISÃO TREINO E TESTE (80/20)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 5. MODELO BASELINE: REGRESSÃO LOGÍSTICA
# ==========================================
print('--- AVALIAÇÃO: REGRESSÃO LOGÍSTICA ---')
log_model = LogisticRegression(
    class_weight='balanced', random_state=42, max_iter=1000
)
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)

print(classification_report(y_test, log_preds))

# ==========================================
# 6. MODELO AVANÇADO: RANDOM FOREST
# ==========================================
print('\n--- AVALIAÇÃO: RANDOM FOREST ---')
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

print(classification_report(y_test, rf_preds))

# ==========================================
# 7. EXTRAÇÃO DAS IMPORTÂNCIAS DAS FEATURES
# ==========================================
df_importancias = (
    pd.DataFrame(
        {'Feature': X_train.columns, 'Importancia': rf_model.feature_importances_}
    )
    .sort_values(by='Importancia', ascending=False)
    .reset_index(drop=True)
)

print('\n--- TOP FEATURES MAIS IMPORTANTES ---')
print(df_importancias.head(10))
