import pandas as pd
import sqlite3
csv_file = 'data/retail_sales_dataset.csv' 
df = pd.read_csv(csv_file)

conn = sqlite3.connect('sales.db')  
table_name = 'sales' 
df.to_sql(table_name, conn, if_exists='replace', index=False)

conn.commit()
conn.close()

print(f"Data from {csv_file} has been successfully inserted into {table_name} table in sales.db")
