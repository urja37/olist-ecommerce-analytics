import duckdb
import pandas as pd
import re
from collections import Counter

con = duckdb.connect()

reviews_query = """
SELECT 
    o.order_id,
    CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 ELSE 0 END as is_late,
    r.review_score,
    r.review_comment_message,
    r.review_comment_title
FROM read_csv_auto('data/olist_orders_dataset.csv') o
JOIN read_csv_auto('data/olist_order_reviews_dataset.csv') r ON o.order_id = r.order_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND o.order_estimated_delivery_date IS NOT NULL
  AND r.review_score <= 2
  AND r.review_comment_message IS NOT NULL
  AND TRIM(r.review_comment_message) != ''
"""

df_rev = con.execute(reviews_query).df()
print(f"Total low review comments to analyze: {len(df_rev):,}")
print(f"Late low review comments: {(df_rev['is_late'] == 1).sum():,}")
print(f"On-time low review comments: {(df_rev['is_late'] == 0).sum():,}")

# Portuguese keywords/regex for root-cause tagging
delivery_keywords = re.compile(r'\b(atraso|atrasou|demorou|demora|prazo|entrega|recebi|chegou|chegar|ainda|esperando|aguardo|rastreio|transportadora|extravio|extraviado|dias|tempo)\b', re.I)
product_defect_keywords = re.compile(r'\b(defeito|quebrado|estragado|danificado|avariado|rasgado|falha|estragou|quebrou|funciona|funcionou|riscado|veio quebrado|veio com defeito)\b', re.I)
wrong_product_keywords = re.compile(r'\b(errado|diferente|outro produto|veio outro|incompativel|tamanho errado|modelo diferente|propaganda enganosa|anuncio)\b', re.I)
missing_incomplete_keywords = re.compile(r'\b(faltando|faltou|incompleto|veio faltando|peca faltando|veio pela metade|apenas um)\b', re.I)
quality_dissatisfaction = re.compile(r'\b(pessim[ao]|ruim|qualidade|horrivel|fraco|material ruim|porcaria|lixo|nao gostei)\b', re.I)
seller_communication = re.compile(r'\b(vendedor|atendimento|contato|resposta|responde|SAC|mensagem|suporte|reclamacao|cancelamento|reembolso|estorno)\b', re.I)

def tag_comment(text):
    text = str(text).lower()
    tags = []
    if delivery_keywords.search(text):
        tags.append('Delivery Delay / Delivery Issue')
    if product_defect_keywords.search(text):
        tags.append('Defective / Broken / Damaged')
    if wrong_product_keywords.search(text):
        tags.append('Wrong Product / Not as Described')
    if missing_incomplete_keywords.search(text):
        tags.append('Missing Parts / Incomplete')
    if quality_dissatisfaction.search(text):
        tags.append('Poor Quality / Dissatisfaction')
    if seller_communication.search(text):
        tags.append('Seller Communication / Customer Service')
    if not tags:
        tags.append('Other / Uncategorized')
    return tags

df_rev['tags'] = df_rev['review_comment_message'].apply(tag_comment)

# Explode tags to analyze proportions
exploded = df_rev.explode('tags')

print("\n=== TOPIC BREAKDOWN IN LATE LOW-REVIEW ORDERS (is_late=1) ===")
late_tags = exploded[exploded['is_late'] == 1]['tags'].value_counts(normalize=True) * 100
print(late_tags)

print("\n=== TOPIC BREAKDOWN IN ON-TIME LOW-REVIEW ORDERS (is_late=0) ===")
ontime_tags = exploded[exploded['is_late'] == 0]['tags'].value_counts(normalize=True) * 100
print(ontime_tags)

# Exact comparison table
# Reset index to avoid pandas duplicate labels error
exploded = exploded.reset_index(drop=True)
summary_tags = pd.crosstab(exploded['tags'], exploded['is_late'], normalize='columns') * 100
summary_tags.columns = ['On-Time Low Reviews (%)', 'Late Low Reviews (%)']
summary_tags['Diff (Late - OnTime)'] = summary_tags['Late Low Reviews (%)'] - summary_tags['On-Time Low Reviews (%)']
print("\n=== TAG COMPARISON TABLE ===")
print(summary_tags.sort_values('Late Low Reviews (%)', ascending=False))

