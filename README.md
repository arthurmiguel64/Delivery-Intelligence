Previsão de Atraso de Entregas no E-commerce (Olist Dataset)

Este projeto de Machine Learning tem como objetivo prever atrasos nas entregas de um e-commerce utilizando dados reais da plataforma Olist. A solução foca na mitigação de data leakage e no tratamento do desbalanceamento de classes inerente a problemas logísticos.

1-Objetivo do Negócio

Identificar a probabilidade de um pedido atrasar no exato momento da postagem na transportadora (order_delivered_carrier_date). Dessa forma, a equipe de operações pode atuar preventivamente em relação aos pedidos de alto risco antes da data prevista de entrega.

2- Tecnologias e Ferramentas

Linguagem: Python 3.x

Manipulação de Dados: Pandas, NumPy

Machine Learning: Scikit-learn (Logistic Regression, Random Forest Classifier)

Métricas de Avaliação: Precision, Recall, F1-Score (foco na classe minoritária: Atrasados)

3- Metodologia e Decisões Técnicas

Prevenção contra Data Leakage:

Utilização exclusiva de informações disponíveis até a postagem do produto.

Descarte de variáveis futuras ou correlacionadas pós-evento de pós-venda.

Engenharia de Atributos e Pré-processamento:

Conversão de timestamps em atributos temporais (dia da semana e hora da compra).

Aplicação de One-Hot Encoding (pd.get_dummies) em variáveis categóricas sem criar hierarquias artificiais.

Divisão do dataset em 80% para treino e 20% para teste com estratificação da variável alvo.

Tratamento do Desbalanceamento:

A maioria dos pedidos chega dentro do prazo, gerando um forte desbalanceamento de classes.

Utilização do parâmetro class_weight='balanced' para penalizar erros na identificação de atrasos e aumentar a taxa de detecção (Recall).

Comparação de Modelos:

Baseline: Regressão Logística (Modelo Linear).

Modelo Avançado: Random Forest Classifier (Captura de relações não-lineares e interações complexas entre atributos).

4- Como Executar o Projeto

Clone o repositório:

git clone https://github.com/seu-usuario/olist-delivery-delay-prediction.git
cd olist-delivery-delay-prediction


Instale as dependências:

pip install pandas numpy scikit-learn


Baixe o dataset da Olist:

Faça o download do dataset no Kaggle (Olist Brazilian E-Commerce).

Extraia os arquivos olist_orders_dataset.csv e olist_order_items_dataset.csv no diretório raiz do projeto.

Execute o script:

python main.py


5- Métricas de Avaliação

A acurácia geral não é a métrica principal devido ao desbalanceamento. A validação prioriza:

Recall (Classe 1): Proporção de atrasos reais identificados pelo modelo.

Precision (Classe 1): Assertividade dos alarmes de atraso gerados pelo modelo.

Feature Importance: Identificação dos principais fatores que causam atrasos (ex: valor do frete, preço do item e horário da compra).
